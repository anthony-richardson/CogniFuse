import importlib
from argparse import ArgumentParser
import argparse
from abc import ABC, abstractmethod
from torch import nn

#class BaseBenchmarkModel(nn.Module, ABC):
class BaseBenchmarkModel(ABC):
    # Returns a dictionary that maps
    # from task name to class number.
    @staticmethod
    @abstractmethod
    def add_model_options(parser_group, default_out_dim, modality=None):
        pass


def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0


def create_model(args):
    model_cls = get_model_cls(args.model_name)
    model_arguments = get_model_arguments(args, model_cls)
    model = model_cls(**model_arguments)
    return model


def get_model_cls(model_name):
    try:
        mod = importlib.import_module('models.' + model_name)
    except ModuleNotFoundError:
        raise ValueError(f'There is no equally named file in models.{model_name}')

    try:
        model_cls = getattr(mod, model_name)
    except ModuleNotFoundError:
        raise ValueError(f'There is no equally named model class in the model file.{model_name}')
    return model_cls


def get_model_arguments(args, model_cls):
    default_out_dim = args.out_dim,
    modality = None if args.multimodal else args.modality

    # We intentionally do not parse this parser so that the user is not required
    # to pass the arguments in the command line in cases where that is not needed.
    dummy_parser = ArgumentParser()
    dummy_parser_group = dummy_parser.add_argument_group('model')
    model_cls.add_model_options(dummy_parser_group, default_out_dim, modality)

    for group in dummy_parser._action_groups:
        if group.title == 'model':
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            arg_names = list(argparse.Namespace(**group_dict).__dict__.keys())

            model_arguments = {}
            for arg_name in arg_names:
                model_arguments[arg_name] = getattr(args, arg_name)

            return model_arguments
    return ValueError('Model group not found.')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
