# This is the script of EEG-Deformer
# This is the network script
import torch
from torch import nn
import copy
from einops import rearrange
from einops.layers.torch import Rearrange

#from utils.model_util import count_parameters


class CrossChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dimension, channels_time_dims, number_of_heads, dim_head, dropout):
        super(CrossChannelTransformerEncoderLayer, self).__init__()
        # same as regular TransformerEncoderLayer, but the values and keys depend on the output of the other channels
        self.sa = Attention(
            q_dim=input_dimension,
            kv_dim=sum(channels_time_dims),
            heads=number_of_heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        """self.sa = nn.MultiheadAttention(
            input_dimension,
            number_of_heads,
            batch_first=True,
            dropout=dropout,
        )"""
        self.ffwd = FeedForward(
            dim=input_dimension,
            hidden_dim=input_dimension * 4,
            out_dim=input_dimension
        )
        
        #self.agg = nn.ParameterList(
        #    [
        #        nn.Parameter(torch.ones(d), requires_grad=True)
        #        for d in channels_time_dims
        #    ]
        #)
        #self.dropout = nn.Dropout(dropout)
        #self.ln = nn.LayerNorm(input_dimension)

    def forward(self, x, other_channels_output):
        # aggregate the output of other channels
        #x_agg = torch.stack(
        #    [torch.mul(x_i, self.agg[i]) for i, x_i in enumerate(other_channels_output)]
        #)
        #other_channels_output = [torch.mul(x_i, self.agg[i]) for i, x_i in enumerate(other_channels_output)]

        #print([o.shape for o in other_channels_output])

        x_agg = torch.cat(other_channels_output, dim=-1)

        #print(x_agg.shape)

        #print(x.shape)

        #print('####')
        #exit()

        #print(x.shape)
        #print(x_agg.shape)

        #att_out = self.sa(x, x_agg, x_agg)

        #print(att_out.shape)
        #exit()

        #print('hi')
        #exit()
        #x_agg = torch.sum(x_agg, dim=0)

        # multi-head attention + residual connection + norm
        #x = x + self.dropout(self.sa(x, x_agg, x_agg)[0]) # , need_weights=False
        x = x + self.sa(x, x_agg, x_agg)  # , need_weights=False
        #x = self.ln(x)

        # feed forward + residual connection
        #x = x + self.dropout(self.ffwd(x))
        x = x + self.ffwd(x)
        return x

####


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == q_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, q_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        #qkv = self.to_qkv(x).chunk(3, dim=-1)

        #print(q.shape)
        #print(k.shape)
        #print(v.shape)

        qkv = [
            self.to_q(q), 
            self.to_k(k),
            self.to_v(v)
        ]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def __init__(self, dims, depth, heads, dim_head, mlp_dim, in_chan, emb_dim,
                 out_dim, fine_grained_kernel=11, dropout=0.):
        super().__init__()

        self.modality_compression_layers = nn.ModuleList([])
        modality_output_sizes = self.get_modality_output_sizes(dims, depth, in_chan)
        for output_size in modality_output_sizes:
            ffwd = FeedForward(
                dim=output_size,
                hidden_dim=emb_dim,
                out_dim=emb_dim
            )
            self.modality_compression_layers.append(ffwd)

        self.output_ffwd = FeedForward(
            dim=emb_dim * len(dims),
            hidden_dim=emb_dim,
            out_dim=out_dim
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):

            depth_layers = nn.ModuleList([])

            # Number of tokens (i.e. time dimension) halves in each depth level.
            dims = [int(d * 0.5) for d in dims]

            for k in range(len(dims)):
                dims_copy = copy.deepcopy(dims)
                dims_copy.pop(k)

                dim = dims[k]

                # the in channels (i.e. the number of kernels) must be the same across
                # all layers of one depth so that the cross attention works!
                # TODO: allow support for non matching by adding linear layers in case of miss match

                depth_layers.append(nn.ModuleList([
                    Attention(q_dim=dim, kv_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedForward(dim, mlp_dim, dim, dropout=dropout),
                    self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout),
                    CrossChannelTransformerEncoderLayer(input_dimension=dim, channels_time_dims=dims_copy, 
                                                        number_of_heads=heads, dim_head=dim_head, dropout=dropout)
                ]))

            self.layers.append(depth_layers)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, channels_output):
        dense_feature = []
        for depth_layers in self.layers:
            depth_dense_feature = []  # list of fine output of each modality at that specific depth

            # HCT blocks, one for each modality
            for i, (attn, ff, cnn, cross_attn) in enumerate(depth_layers):
                # Modality specific tensor
                x = channels_output[i]

                x_cg = self.pool(x)
                x_cg = attn(x_cg, x_cg, x_cg) + x_cg
                x_fg = cnn(x)
                x_info = self.get_info(x_fg)  # (b, in_chan)
                depth_dense_feature.append(x_info)
                x = ff(x_cg) + x_fg

                channels_output[i] = x

            dense_feature.append(depth_dense_feature)

            # Cross attentions blocks, one for each modality
            for i, (attn, ff, cnn, cross_attn) in enumerate(depth_layers):
                # Modality specific tensor
                x = channels_output[i]

                # When using clone, the gradients will flow back to the input
                ##other_channels = [torch.clone(t) for n, t in enumerate(channels_output) if n != i]

                ##x = cross_attn(x, other_channels)
                x = cross_attn(
                    x, [h for n, h in enumerate(channels_output) if n != i]
                )

                channels_output[i] = x

        #print('-----')
        modality_specific_emb = []
        for i, chan_out in enumerate(channels_output):

            modality_levels = [depth[i] for depth in dense_feature]
            #print([level.shape for level in modality_levels])

            #print(chan_out.shape)
            chan_out = chan_out.view(chan_out.size(0), -1)
            #print(chan_out.shape)

            modality_levels_combined = torch.cat(modality_levels, dim=-1)
            modality_components = torch.cat((chan_out, modality_levels_combined), dim=-1)
            #print(modality_components.shape)

            modality_emb = self.modality_compression_layers[i](modality_components)

            #print(modality_emb.shape)

            modality_specific_emb.append(modality_emb)

        modality_specific_emb_combined = torch.cat(modality_specific_emb, dim=-1)
        emb = self.output_ffwd(modality_specific_emb_combined)

        #print('---')
        #print(emb.shape)

        #exit()

        #x_dense = torch.cat(dense_feature, dim=-1)  # b, in_chan*depth
        #x = x.view(x.size(0), -1)   # b, in_chan*d_hidden_last_layer
        #emd = torch.cat((x, x_dense), dim=-1)  # b, in_chan*(depth + d_hidden_last_layer)

        return emb

    def get_modality_output_sizes(self, dims, depth, in_chan):
        return [int(dim * (0.5 ** depth)) * in_chan + in_chan * depth for dim in dims]

    def get_info(self, x):
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class MultiChannelDeformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(self, *, dims, num_channels, temporal_kernel, num_kernel=64,
                 emb_dim, out_dim, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.):
        super().__init__()

        self.cnn_encoders = nn.ModuleList([])
        for chan in num_channels:
            cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=chan)
            self.cnn_encoders.append(cnn_encoder)


        #self.cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan)

        dims = [int(0.5 * d) for d in dims]  # embedding size after the first cnn encoders

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embeddings = nn.ParameterList([])
        for dim in dims:
            pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))
            self.pos_embeddings.append(pos_embedding)

        self.transformer = Transformer(
            dims=dims, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, emb_dim=emb_dim, out_dim=out_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel
        )

        #L = self.get_hidden_size(input_size=dim, num_layer=depth)

        #out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)

        #self.mlp_head = nn.Sequential(
        #    nn.Linear(out_size, emb_dim)
        #)

    def forward(self, channels):

        for i, chan in enumerate(channels):
            chan = torch.unsqueeze(chan, dim=1)  # (b, 1, channels, time)
            chan = self.cnn_encoders[i](chan)
            chan = self.to_patch_embedding(chan)
            b, n, _ = chan.shape
            chan += self.pos_embeddings[i]
            channels[i] = chan

        # eeg: (b, chan, time)
        #eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        #x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        #x = self.to_patch_embedding(x)

        #b, n, _ = x.shape
        #x += self.pos_embedding

        emb = self.transformer(channels)

        return emb  # self.mlp_head(x)

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    '''def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]'''


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_model = MultiChannelDeformer(
        dims=[4 * 128, 6 * 32, 4 * 32, 10 * 32],
        num_channels=[16, 1, 1, 1],
        depth=4,
        heads=16,
        dim_head=16,
        mlp_dim=16,
        num_kernel=64,
        emb_dim=256,
        out_dim=2,
        temporal_kernel=13,
        dropout=0.
    )

    dummy_eeg = torch.randn(1, 16, 4 * 128)
    dummy_ppg = torch.randn(1, 1, 6 * 32)
    dummy_eda = torch.randn(1, 1, 4 * 32)
    dummy_resp = torch.randn(1, 1, 10 * 32)
    channels = [
        dummy_eeg,
        dummy_ppg, dummy_eda,
        dummy_resp
    ]

    print(dummy_model)
    print(count_parameters(dummy_model))

    output = dummy_model(channels)

    print(output)
    print(output.shape)
