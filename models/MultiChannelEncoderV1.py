# This is an implementation of the Encoder of the Multi-Channel Transformer from 
# Camgoz et a. (https://arxiv.org/pdf/2009.00299), repurposed for classifiaction tasks. 
# Since the authors do not provide any source code, this implementation was done 
# by closely following the papers description.

import torch
from torch import nn
from einops import rearrange

from models.BaseBenchmarkModel import BaseBenchmarkModel


class CrossChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dimension, number_of_heads, dim_head, mlp_dims, dropout, out_dim=None):
        super(CrossChannelTransformerEncoderLayer, self).__init__()
        
        self.sa = Attention(
            q_dim=input_dimension,
            heads=number_of_heads,
            dim_head=dim_head,
            dropout=dropout,
            create_heads=False
        )
        
        self.ffwd = FeedForward(
            dim=input_dimension,
            hidden_dim=sum(mlp_dims), 
            out_dim=out_dim if out_dim is not None else input_dimension
        )

    def forward(self, x_q, other_channels_output):
        # Stacking the other channels on the kernel dimension.
        # This allows the use of different numbers of kernels for the modalities.
        k_other_channels = [k for _, k, _ in other_channels_output]
        k_agg = torch.cat(k_other_channels, dim=-2)

        v_other_channels = [v for _, _, v in other_channels_output]
        v_agg = torch.cat(v_other_channels, dim=-2)

        x = x_q + self.sa(x_q, k_agg, v_agg)
        x = self.ffwd(x)
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0., end_w_dropout=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout) if end_w_dropout else nn.Identity()
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, q_dim, heads=8, dim_head=64, dropout=0., create_heads=True, out_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == q_dim)

        self.create_heads = create_heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        
        if self.create_heads:
            self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(q_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(q_dim, inner_dim, bias=False)

        if out_dim is not None:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, out_dim),
                nn.Dropout(dropout)
            )
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, q_dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        if self.create_heads:
            qkv = [
                self.to_q(q), 
                self.to_k(k),
                self.to_v(v)
            ]
        else:
            qkv = [q, k, v]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dims, num_chan, depth, heads, dim_head, mlp_dims, emb_dims,
                 out_dim, dropout=0.):
        super().__init__()
        
        modality_output_sizes = self.get_modality_output_sizes(dims, num_chan)

        self.output_ffwd = FeedForward(
            dim=sum(modality_output_sizes),
            hidden_dim=sum(emb_dims),
            out_dim=out_dim,
            end_w_dropout=False
        )

        inner_dim = dim_head * heads

        self.layers = nn.ModuleList([])
        for i in range(depth):
            depth_layers = nn.ModuleList([])

            for k in range(len(dims)):
                dim = dims[k]
                mlp_dim = mlp_dims[k]

                depth_layers.append(nn.ModuleList([
                    Attention(q_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedForward(dim, mlp_dim, dim, dropout=dropout),
                    nn.Linear(dim, inner_dim, bias=True),
                    nn.Linear(dim, inner_dim, bias=True),
                    nn.Linear(dim, inner_dim, bias=True),
                    nn.LayerNorm(inner_dim),
                    nn.LayerNorm(inner_dim),
                    nn.LayerNorm(inner_dim),
                    CrossChannelTransformerEncoderLayer(input_dimension=inner_dim, number_of_heads=heads, 
                                                        dim_head=dim_head, mlp_dims=mlp_dims, dropout=dropout, out_dim=dim)
                ]))

            self.layers.append(depth_layers)

    def forward(self, channels_output):
        for depth_layers in self.layers:
            # HCT blocks, one for each modality
            for i, (attn, ff, q_layer, k_layer, v_layer, q_norm, k_norm, v_norm, _) in enumerate(depth_layers):
                # Modality specific tensor
                x = channels_output[i]

                x_cg = attn(x, x, x) + x

                x = ff(x_cg)

                x_q = q_layer(x)
                x_q = q_norm(x_q)

                x_k = k_layer(x)
                x_k = k_norm(x_k)

                x_v = v_layer(x)
                x_v = v_norm(x_v)
                
                channels_output[i] = (x_q, x_k, x_v)

            new_channels_output = []
            # Cross attentions blocks, one for each modality
            for i, (_, _, _, _, _, _, _, _, cross_attn) in enumerate(depth_layers):
                # Modality specific tensors
                x_q, x_k, x_v = channels_output[i]
                x = cross_attn(
                    x_q, [h for n, h in enumerate(channels_output) if n != i]
                )
                new_channels_output.append(x)

            channels_output = new_channels_output

        modality_specific_emb = []
        for i, chan_out in enumerate(channels_output):
            chan_out = chan_out.view(chan_out.size(0), -1)
            modality_specific_emb.append(chan_out)

        modality_specific_emb_combined = torch.cat(modality_specific_emb, dim=-1)

        out = self.output_ffwd(modality_specific_emb_combined)

        return out

    def get_modality_output_sizes(self, dims, num_chan):
        return [dim * chan for dim, chan in zip(dims, num_chan)]


class MultiChannelEncoderV1(BaseBenchmarkModel):
    @staticmethod
    def add_model_options(parser_group):
        # These can vary between modalities
        parser_group.add_argument("--mlp_dim", default=[16, 16, 16, 16], type=int, nargs="+",
                           help="Dimensions of MLPs for the modalities")
        parser_group.add_argument("--emb_dim", default=[256, 16, 16, 16], type=int, nargs="+",
                           help="Embedding dimensions for the modalities")

        # These three must match for all modalities
        parser_group.add_argument("--depth", default=4, type=int, help="Depth of kernels")
        # The Multi-Channel Transformer form Camgoz et al. (https://arxiv.org/pdf/2009.00299) 
        # only uses a single head to reduce the number of parameters. 
        parser_group.add_argument("--heads", default=1, type=int, help="Number of heads")
        #parser_group.add_argument("--heads", default=16, type=int, help="Number of heads")
        parser_group.add_argument("--dim_head", default=16, type=int, help="Dimension of heads")

        parser_group.add_argument("--dropout", default=0.2, type=float, help="Dropout rate")
        
    def channel_embedding(self, dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def __init__(self, *, num_time, num_chan, mlp_dim,
                 emb_dim, depth, heads, dim_head, dropout, out_dim):
        super().__init__()

        self.embedders = nn.ModuleList([])
        for dim in num_time:
            embedder = self.channel_embedding(dim=dim, hidden_dim=dim)
            self.embedders.append(embedder)

        self.pos_embeddings = nn.ParameterList([])
        for dim in num_time:
            pos_embedding = nn.Parameter(torch.randn(1, dim))
            self.pos_embeddings.append(pos_embedding)

        self.transformer = Transformer(
            dims=num_time, num_chan=num_chan, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dims=mlp_dim, emb_dims=emb_dim, out_dim=out_dim, dropout=dropout
        )

        
    def forward(self, channels):
        for i, chan in enumerate(channels):
            chan = self.embedders[i](chan)
            chan += self.pos_embeddings[i]
            channels[i] = chan

        emb = self.transformer(channels)

        return emb

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_model = MultiChannelEncoderV1(
        num_time=[4 * 128, 6 * 128, 4 * 64, 10 * 32],
        num_chan=[16, 1, 1, 1],
        mlp_dim=[16, 16, 16, 16],
        emb_dim=[256, 16, 16, 16],
        depth=4,
        #heads=16,
        heads=1,
        dim_head=16,
        dropout=0.,
        out_dim=2
    )

    dummy_eeg = torch.randn(1, 16, 4 * 128)
    dummy_ppg = torch.randn(1, 1, 6 * 128)
    dummy_eda = torch.randn(1, 1, 4 * 64)
    dummy_resp = torch.randn(1, 1, 10 * 32)
    channels = [
        dummy_eeg,
        dummy_ppg, 
        dummy_eda,
        dummy_resp
    ]

    print(dummy_model)
    print(count_parameters(dummy_model))

    output = dummy_model(channels)

    print(output)
    print(output.shape)
    