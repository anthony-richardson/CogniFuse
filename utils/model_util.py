from utils.parser_util import get_model_cls, get_model_arguments


def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0


def create_model(args):
    model_cls = get_model_cls(args.model_name)
    model_arguments = get_model_arguments(args, model_cls)
    model = model_cls(**model_arguments)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
