# This is an implementation of the Encoder of the Multi-Channel Transformer from 
# Chang et al. (https://arxiv.org/pdf/2102.03951), repurposed for classifiaction tasks. 
# Since the authors do not provide any source code, this implementation was done by 
# following the papers description as close as possible. To produce a prediction from 
# the multiple encoder outputs (one output for each modality), we concatenate them 
# and pass the result through a multi layer perceptron. 

import torch
from torch import nn
from einops import rearrange

from models.BaseBenchmarkModel import BaseBenchmarkModel


class CrossChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dimension, dims, number_of_heads, dim_head, mlp_dims, dropout, out_dim=None):
        super(CrossChannelTransformerEncoderLayer, self).__init__()

        self.channel_weights = nn.Parameter(torch.randn(len(dims)))

        self.to_k = nn.Linear(input_dimension, input_dimension, bias=True)

        self.to_v = nn.Linear(input_dimension, input_dimension, bias=True)
        
        self.sa = Attention(
            q_dim=input_dimension,
            heads=number_of_heads,
            dim_head=dim_head,
            dropout=dropout,
            create_heads=False,
            out_dim=input_dimension
        )

        self.layer_norm = nn.LayerNorm(input_dimension)
        
        self.ffwd = FeedForward(
            dim=input_dimension,
            hidden_dim=sum(mlp_dims), 
            out_dim=out_dim if out_dim is not None else input_dimension
        )

    def forward(self, x_q, other_channels_output):
        other_channels = [c for _, c in other_channels_output]

        # Weighting the other hannels
        other_channels_weighted = [self.channel_weights[i] * c for i, c in enumerate(other_channels)]
        other_channels_weighted = torch.stack(other_channels_weighted)

        # Weighted sum of other channels 
        other_channels_weighted_sum = torch.sum(other_channels_weighted, dim=0)

        k_agg = self.to_k(other_channels_weighted_sum)

        v_agg = self.to_v(other_channels_weighted_sum)

        x = self.sa(x_q, k_agg, v_agg) + x_q
        
        x = self.layer_norm(x)
        x = self.ffwd(x) + x
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            #nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
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
            self.to_q = nn.Linear(q_dim, inner_dim, bias=True)
            self.to_k = nn.Linear(q_dim, inner_dim, bias=True)
            self.to_v = nn.Linear(q_dim, inner_dim, bias=True)

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

        qkv = [torch.nn.functional.relu(c) for c in qkv]

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
        
        inner_dim = dim_head * heads

        modality_output_sizes = self.get_modality_output_sizes(inner_dim, num_chan)

        self.layer_norm = nn.LayerNorm(sum(modality_output_sizes))
        self.output_ffwd = FeedForward(
            dim=sum(modality_output_sizes),
            hidden_dim=sum(emb_dims),
            out_dim=out_dim
        )
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            depth_layers = nn.ModuleList([])

            for k in range(len(dims)):
                mlp_dim = mlp_dims[k]

                depth_layers.append(nn.ModuleList([
                    nn.LayerNorm(inner_dim),
                    Attention(q_dim=inner_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    nn.LayerNorm(inner_dim),
                    FeedForward(inner_dim, mlp_dim, inner_dim),
                    nn.LayerNorm(inner_dim),
                    nn.Linear(inner_dim, inner_dim, bias=True),
                    CrossChannelTransformerEncoderLayer(input_dimension=inner_dim, dims=dims, number_of_heads=heads, 
                                                        dim_head=dim_head, mlp_dims=mlp_dims, dropout=dropout)
                ]))

            self.layers.append(depth_layers)

    def forward(self, channels_output):
        for depth_layers in self.layers:
            # HCT blocks, one for each modality
            for i, (x_norm_a, attn, x_norm_b, ff, x_norm_c, q_layer, _) in enumerate(depth_layers):
                # Modality specific tensor
                x = channels_output[i]
                x = x_norm_a(x)

                x = attn(x, x, x) + x

                x = x_norm_b(x)
                x = ff(x) + x

                x = x_norm_c(x)
                x_q = q_layer(x)
                
                channels_output[i] = (x_q, x)

            new_channels_output = []
            # Cross attention blocks, one for each modality
            for i, (_, _, _, _, _, _, cross_attn) in enumerate(depth_layers):
                # Modality specific tensors
                x_q, _ = channels_output[i]
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

        modality_specific_emb_combined = self.layer_norm(modality_specific_emb_combined)
        out = self.output_ffwd(modality_specific_emb_combined)

        return out

    def get_modality_output_sizes(self, inner_dim, num_chan):
        max_num_chan = max(num_chan)
        return [inner_dim * max_num_chan for _ in num_chan]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if pe.shape[1] % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else: 
            # In the case of uneven dimensionality, the last entry in the 
            # position tensor must be ignored for the uneven indices. 
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return x


class ProjectChannels(torch.nn.Module):
    def __init__(self, chan, target_chan):
        super().__init__()

        self.project_channels = nn.Linear(chan, target_chan, bias=False) \
                                    if chan != target_chan else nn.Identity()

    def forward(self, x):
        x = torch.transpose(x, -1, -2)
        x = self.project_channels(x)
        x = torch.transpose(x, -1, -2)
        return x


class MultiChannelEncoderV2(BaseBenchmarkModel):
    @staticmethod
    def add_model_options(parser_group):
        # These can vary between modalities
        parser_group.add_argument("--mlp_dim", default=[16, 16, 16, 16], type=int, nargs="+",
                           help="Dimensions of MLPs for the modalities")
        parser_group.add_argument("--emb_dim", default=[256, 16, 16, 16], type=int, nargs="+",
                           help="Embedding dimensions for the modalities")

        # These three must match for all modalities
        parser_group.add_argument("--depth", default=4, type=int, help="Depth of kernels")
        parser_group.add_argument("--heads", default=16, type=int, help="Number of heads")
        parser_group.add_argument("--dim_head", default=16, type=int, help="Dimension of heads")

        parser_group.add_argument("--dropout", default=0.2, type=float, help="Dropout rate")
        
    def channel_embedding(self, dim, hidden_dim, chan, target_chan):
        return nn.Sequential(
            nn.Linear(dim, hidden_dim), 
            # In the Multi-Chanel Transformer by Chang et al. (https://arxiv.org/pdf/2102.03951), all 
            # modalities must have the same number of channels so that their weighted sum can be calculated.
            # Therefore, we project all modalities to the largest number of channels among them. 
            ProjectChannels(chan, target_chan)
        )

    def __init__(self, *, num_time, num_chan, mlp_dim,
                 emb_dim, depth, heads, dim_head, dropout, out_dim):
        super().__init__()

        inner_dim = dim_head * heads

        target_num_chan = max(num_chan)

        self.embedders = nn.ModuleList([])
        for dim, chan in zip(num_time, num_chan):
            embedder = self.channel_embedding(
                dim=dim, 
                hidden_dim=inner_dim, 
                chan=chan, 
                target_chan=target_num_chan
            )
            self.embedders.append(embedder)

        self.pos_embeddings = nn.ParameterList([])
        for dim in num_time:
            pos_embedding = PositionalEncoding(d_model=inner_dim)
            self.pos_embeddings.append(pos_embedding)

        self.transformer = Transformer(
            dims=num_time, num_chan=num_chan, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dims=mlp_dim, emb_dims=emb_dim, out_dim=out_dim, dropout=dropout
        )

    def forward(self, channels):
        for i, chan in enumerate(channels):
            chan = self.embedders[i](chan)
            chan = self.pos_embeddings[i](chan)
            channels[i] = chan

        emb = self.transformer(channels)

        return emb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_model = MultiChannelEncoderV1(
        num_time=[4 * 128, 6 * 128, 4 * 64, 10 * 32],
        num_chan=[16, 1, 1, 1],
        mlp_dim=[16, 16, 16, 16],
        emb_dim=[256, 16, 16, 16],
        depth=4,
        heads=16,
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
    
