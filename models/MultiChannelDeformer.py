import torch
from torch import nn
import copy
from einops import rearrange
from einops.layers.torch import Rearrange

from utils.model_util import BaseBenchmarkModel
#from utils.model_util import count_parameters

# TODO
#  - think about and maybe replace random positional encoding by sin/cos encoding
#  (random does not seem like it would guarantee no repetitions)
#  - maybe the random operation also contributes to the large drop in training
#  speed when forcing deterministic behaviour dor reproducibility
#  - try sin/cos and compare if it still works and is as good as before

class CrossChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dimension, number_of_heads, dim_head, mlp_dims, dropout, out_dim=None):
        super(CrossChannelTransformerEncoderLayer, self).__init__()
        
        self.sa = Attention(
            q_dim=input_dimension,
            #kv_dim=sum(channels_time_dims),
            heads=number_of_heads,
            dim_head=dim_head,
            dropout=dropout,
            create_heads=False,
            out_dim=input_dimension if out_dim is None else out_dim
        )
        
        self.ffwd = FeedForward(
            #dim=input_dimension,
            dim=input_dimension if out_dim is None else out_dim,
            hidden_dim=sum(mlp_dims), #mlp_dim * 4, #input_dimension * 4,
            # maybe reduce this to *2. But first do num kernels and emb dim. Then see if the impact is large
            #hidden_dim=input_dimension * 2,
            out_dim=out_dim if out_dim is not None else input_dimension #input_dimension
        )

        #self.ffwd = nn.Linear(input_dimension, out_dim if out_dim is not None else input_dimension)
        

    def forward(self, x, x_q, other_channels_output):
        # Stacking the other channels on the kernel dimension.
        # This allows the use of different numbers of kernels for the modalities.

        #for o in other_channels_output:
        #    print(o.shape)

        k_other_channels = [k for x, q, k, v in other_channels_output]
        k_agg = torch.cat(k_other_channels, dim=-2)

        v_other_channels = [v for x, q, k, v in other_channels_output]
        v_agg = torch.cat(v_other_channels, dim=-2)

        #x_agg = torch.cat(other_channels_output, dim=-2) #dim=-1)
        #print(f'hi: {x.shape}')

        #print(x_agg.shape)

        #exit()

        #x = x + self.sa(x, x_agg, x_agg)
        x = x + self.sa(x_q, k_agg, v_agg)

        #print(x.shape)

        #exit()
        #x = x + self.ffwd(x)
        x = self.ffwd(x)

        #exit()
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
            # nn.Dropout(dropout)
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

        #self.q_dim = q_dim
        
        if self.create_heads:
            self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(q_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(q_dim, inner_dim, bias=False)

        if out_dim is not None:
            #("hm")
            #self.to_out = nn.Linear(inner_dim, out_dim)
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, out_dim),
                nn.Dropout(dropout)
            )
        else:
            #print("hmhm")
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, q_dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        #print(q.shape)
        #print(k.shape)
        #print(v.shape)

        #print(self.q_dim)
        if self.create_heads:
            qkv = [
                self.to_q(q), 
                self.to_k(k),
                self.to_v(v)
            ]
        else:
            qkv = [q, k, v]

        #print(qkv[0].shape)
        #print(qkv[1].shape)
        #print(qkv[2].shape)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        #print(q.shape)
        #print(k.shape)
        #print(v.shape)

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

    def __init__(self, dims, depth, heads, dim_head, mlp_dims, in_chans, emb_dims,
                 out_dim, fine_grained_kernels, dropout=0.):
        super().__init__()

        self.modality_compression_layers = nn.ModuleList([])
        modality_output_sizes = self.get_modality_output_sizes(dims, depth, in_chans)

        #print(modality_output_sizes)
        #exit()

        for output_size, emb_dim in zip(modality_output_sizes, emb_dims):
            ffwd = FeedForward(
                dim=output_size,
                hidden_dim=emb_dim,
                out_dim=emb_dim,
                end_w_dropout=False
            )
            self.modality_compression_layers.append(ffwd)

        self.output_ffwd = FeedForward(
            dim=sum(emb_dims), #emb_dim * len(dims),
            hidden_dim=sum(emb_dims), #emb_dim,
            out_dim=out_dim,
            end_w_dropout=False
        )

        inner_dim = dim_head * heads

        self.layers = nn.ModuleList([])
        for i in range(depth):

            depth_layers = nn.ModuleList([])

            # Number of tokens (i.e. time dimension) halves in each depth level.
            dims = [int(d * 0.5) for d in dims]

            for k in range(len(dims)):
                #dims_copy = copy.deepcopy(dims)
                #dims_copy.pop(k)

                dim = dims[k]
                mlp_dim = mlp_dims[k]
                fine_grained_kernel = fine_grained_kernels[k]
                in_chan = in_chans[k]

                depth_layers.append(nn.ModuleList([
                    #nn.Linear(dim, inner_dim, bias=False),
                    Attention(q_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedForward(dim, mlp_dim, dim, dropout=dropout),
                    self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout),
                    #nn.Linear(dim, inner_dim, bias=False),
                    # Version with added bias term for comparability to crossmodal deformer
                    nn.Linear(dim, inner_dim, bias=True),
                    nn.Linear(dim, inner_dim, bias=True),
                    nn.Linear(dim, inner_dim, bias=True),
                    CrossChannelTransformerEncoderLayer(input_dimension=inner_dim, number_of_heads=heads, 
                                                        dim_head=dim_head, mlp_dims=mlp_dims, dropout=dropout, out_dim=dim)
                ]))

            self.layers.append(depth_layers)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, channels_output):
        dense_feature = []
        for depth_layers in self.layers:
            # list of fine output of each modality at that specific depth
            depth_dense_feature = []  

            #depth_output_heads = []

            # HCT blocks, one for each modality
            for i, (attn, ff, cnn, q_layer, k_layer, v_layer, _) in enumerate(depth_layers):
                # Modality specific tensor
                x = channels_output[i]
                
                
                #print(x.shape)
                x_cg = self.pool(x)
                #print(x_cg.shape)
                #x_cg_heads = self_att_head_creator(x_cg)

                x_cg = attn(x_cg, x_cg, x_cg) + x_cg

                x_fg = cnn(x)

                x_info = self.get_info(x_fg)  # (b, in_chan)
                depth_dense_feature.append(x_info)
                # TODO think about if this ff can be directly used to get the heads
                x = ff(x_cg) + x_fg


                #x = head_creator(x)
                x_q = q_layer(x)
                x_k = k_layer(x)
                x_v = v_layer(x)
                
                channels_output[i] = (x, x_q, x_k, x_v)

                #print(f"x after self loop {x.shape}")
                #exit()

                
                #depth_output_heads.append(depth_heads)

            dense_feature.append(depth_dense_feature)

            new_channels_output = []
            # Cross attentions blocks, one for each modality
            for i, (_, _, _, _, _, _, cross_attn) in enumerate(depth_layers):
                # Modality specific tensor
                x, x_q, x_k, x_v = channels_output[i]
                #x = depth_output_heads[i]

                #k_agg =

                x = cross_attn(
                    x, x_q, [h for n, h in enumerate(channels_output) if n != i]
                )


                #print(x.shape)

                #print("hiii")
                #exit()

                new_channels_output.append(x)

                #channels_output[i] = x

            channels_output = new_channels_output

        #exit()

        modality_specific_emb = []
        for i, chan_out in enumerate(channels_output):
            modality_levels = [depth[i] for depth in dense_feature]
            
            chan_out = chan_out.view(chan_out.size(0), -1)
            
            modality_levels_combined = torch.cat(modality_levels, dim=-1)
            modality_components = torch.cat((chan_out, modality_levels_combined), dim=-1)
            modality_emb = self.modality_compression_layers[i](modality_components)
            modality_specific_emb.append(modality_emb)

        modality_specific_emb_combined = torch.cat(modality_specific_emb, dim=-1)
        emb = self.output_ffwd(modality_specific_emb_combined)

        return emb

    def get_modality_output_sizes(self, dims, depth, in_chans):
        return [int(dim * (0.5 ** depth)) * in_chan + in_chan * depth for dim, in_chan in zip(dims, in_chans)]

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


class MultiChannelDeformer(nn.Module, BaseBenchmarkModel):
    @staticmethod
    def add_model_options(parser_group, default_out_dim, modality=None):
        #group = parser.add_argument_group('model')

        # These can vary between modalities
        parser_group.add_argument("--num_time", default=[4 * 128, 6 * 128, 4 * 64, 10 * 32], type=int, nargs="+",
                           help="Number of time steps for the resp modality")
        parser_group.add_argument("--num_chan", default=[16, 1, 1, 1], type=int, nargs="+",
                           help="Number of channels for the modalities")
        parser_group.add_argument("--mlp_dim", default=[16, 16, 16, 16], type=int, nargs="+",
                           help="Dimensions of MLPs for the modalities")
        parser_group.add_argument("--num_kernel", default=[64, 4, 4, 4], type=int, nargs="+",
                           help="Numbers of kernels for the modalities")
        parser_group.add_argument("--temporal_kernel", default=[13, 13, 13, 13], type=int, nargs="+",
                           help="Lengths of temporal kernels for the modalities")
        parser_group.add_argument("--emb_dim", default=[256, 16, 16, 16], type=int, nargs="+",
                           help="Embedding dimensions for the modalities")

        # These must match for all modalities
        parser_group.add_argument("--depth", default=4, type=int, help="Depth of kernels")
        parser_group.add_argument("--heads", default=16, type=int, help="Number of heads")
        parser_group.add_argument("--dim_head", default=16, type=int, help="Dimension of heads")
        #parser_group.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
        # TODO: analyse what rate is better
        parser_group.add_argument("--dropout", default=0.2, type=float, help="Dropout rate")
        parser_group.add_argument("--out_dim", default=default_out_dim, type=int,
                           help="Size of the output. For classification tasks, this is the number of classes.")

    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            # Only do spatial convolution if there is more than one channel
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1),
                                 padding=0, max_norm=2) if num_chan > 1 else nn.Identity(),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            # the transformer also does pooling in the beginning
            #nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(self, *, num_time, num_chan, mlp_dim, num_kernel, temporal_kernel,
                 emb_dim, depth, heads, dim_head, dropout, out_dim):
        super().__init__()

        self.cnn_encoders = nn.ModuleList([])
        #print(num_chan, num_kernel, temporal_kernel)
        for chan, num_kern, temporal_kern in zip(num_chan, num_kernel, temporal_kernel):
            cnn_encoder = self.cnn_block(out_chan=num_kern, kernel_size=(1, temporal_kern), num_chan=chan)
            self.cnn_encoders.append(cnn_encoder)

        #dims = [int(0.5 * d) for d in dims]  # embedding size after the first cnn encoders

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embeddings = nn.ParameterList([])
        for dim, num_kern in zip(num_time, num_kernel):
            pos_embedding = nn.Parameter(torch.randn(1, num_kern, dim))
            self.pos_embeddings.append(pos_embedding)

        self.transformer = Transformer(
            dims=num_time, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dims=mlp_dim, emb_dims=emb_dim, out_dim=out_dim, dropout=dropout,
            in_chans=num_kernel, fine_grained_kernels=temporal_kernel
        )

        
    def forward(self, channels):
        for i, chan in enumerate(channels):
            chan = torch.unsqueeze(chan, dim=1)  # (b, 1, channels, time)
            chan = self.cnn_encoders[i](chan)
            chan = self.to_patch_embedding(chan)
            b, n, _ = chan.shape
            chan += self.pos_embeddings[i]
            channels[i] = chan

        emb = self.transformer(channels)

        return emb

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    '''def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]'''


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_model = MultiChannelDeformer(
        num_time=[4 * 128, 6 * 128, 4 * 64, 10 * 32],
        num_chan=[16, 1, 1, 1],
        mlp_dim=[16, 16, 16, 16],
        num_kernel=[64, 4, 4, 4],
        temporal_kernel=[13, 13, 13, 13],
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
        dummy_ppg, dummy_eda,
        dummy_resp
    ]

    print(dummy_model)
    print(count_parameters(dummy_model))

    output = dummy_model(channels)

    print(output)
    print(output.shape)

    #tensor1 = torch.randn(1, 16, 4, 16)
    #tensor2 = torch.randn(1, 16, 16, 64)
    #result = torch.matmul(tensor1, tensor2).size()

    #print("hi")
    #print(result)
    #print(count_parameters(dummy_model))
