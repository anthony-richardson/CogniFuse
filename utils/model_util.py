from models.EEGDeformer import Deformer
from models.MultiChannelDeformer import MultiChannelDeformer
from models.EarlyFusionDeformer import EarlyFusionDeformer
from models.IntermediateFusionDeformer import IntermediateFusionDeformer


def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0


def create_unimodal_deformer(args):
    deformer_model = Deformer(
        num_chan=args.num_chan,
        num_time=args.num_time,
        temporal_kernel=args.temporal_kernel,
        num_kernel=args.num_kernel,
        emb_dim=args.emb_dim,
        out_dim=args.out_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        dropout=args.dropout
    )
    return deformer_model


def create_multimodal_deformer(args):
    if args.fusion_type == 'crossmodal':
        model_cls = MultiChannelDeformer
    elif args.fusion_type == 'early':
        model_cls = EarlyFusionDeformer
    elif args.fusion_type == 'intermediate':
        model_cls = IntermediateFusionDeformer
    else:
        raise ValueError(f'Unknown fusion type: {args.fusion_type}')

    deformer_model = model_cls(
        dims=[
            args.num_time_eeg,
            args.num_time_ppg,
            args.num_time_eda,
            args.num_time_resp
        ],
        num_channels=[
            args.num_chan_eeg,
            args.num_chan_ppg,
            args.num_chan_eda,
            args.num_chan_resp
        ],
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        num_kernel=args.num_kernel,
        emb_dim=args.emb_dim,
        out_dim=args.out_dim,
        temporal_kernel=args.temporal_kernel,
        dropout=args.dropout
    )
    return deformer_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
