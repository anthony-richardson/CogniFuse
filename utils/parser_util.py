from argparse import ArgumentParser
import argparse
import os
import json
import inspect
import datetime

import utils.tasks as tasks
from utils.model_util import get_model_cls


# TODO: is this being used or is it replaced by create_model in model_util?
def parse_and_load_from_model(parser, model_path):
    if model_path is None:
        model_path = get_model_path_from_args()

    args_path = os.path.join(os.path.dirname(model_path), '..', 'args.json')

    # Load args from model
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    add_multimodal_option(parser)
    add_model_name_option(parser)

    add_seed(parser)
    group_names = ['seed', 'model', 'multimodal', 'model_name']
    if not model_args['multimodal']:
        add_modality_option(parser)
        group_names.append('modality')

    parser_group = parser.add_argument_group('model')

    out_dim = model_args['out_dim']
    model_name = model_args['model_name']
    get_model_cls(model_name).add_model_options(
        parser_group=parser_group,
        default_out_dim=out_dim,
        modality=None if model_args['multimodal'] else model_args['modality']
    )

    args, _ = parser.parse_known_args()

    # Args according to the loaded model.
    # Do not try to specify them from cmd line since they will be overwritten.
    args_to_overwrite = []
    for group_name in group_names:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # Overwrite args from model.
    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])
        else:
            print('Warning: was not able to load [{}], '
                  'using default value [{}] instead.'.format(a, args.__dict__[a]))
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser(add_help=False)
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        model_path = dummy_args.model_path
        if model_path is None:
            raise ValueError('model_path argument must be specified.')
        else:
            return model_path
    except Exception:
        raise ValueError('model_path argument must be specified.')


def get_output_size_from_task():
    dummy_parser = ArgumentParser(add_help=False)
    add_task_option(dummy_parser)

    dummy_args, _ = dummy_parser.parse_known_args()
    task_tools = getattr(tasks, dummy_args.task)
    nr_classes = len(set(task_tools.get_mapper().values()))
    return nr_classes


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", choices=[0, 1], default=0, type=int, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")

    
def add_model_name_option(parser):
    group = parser.add_argument_group('model_name')
    model_dir = os.path.join(os.getcwd(), 'models')
    model_name_options = [name.split('.')[0] for name in os.listdir(model_dir)
                          if os.path.isfile(os.path.join(model_dir, name))
                          and name.endswith('.py')]
    if len(model_name_options) == 0:
        raise Exception('No model name options found.')

    group.add_argument("--model_name", type=str, help="The file name containing the equally named model class.",
                       choices=model_name_options, default=model_name_options[-1])


def add_data_options(parser, cross_validate=False):
    group = parser.add_argument_group('dataset')
    group.add_argument("--data_dir",
                       default=os.path.join(os.getcwd(), 'data', 'folds'),
                       type=str, help="Directory where the data splits are stored.")

    if not cross_validate:
        # Determining the fold options.
        dummy_parser = ArgumentParser(add_help=False)
        dummy_parser.add_argument("--data_dir",
                                  default=os.path.join(os.getcwd(), 'data', 'folds'),
                                  type=str, help="Directory where the data splits are stored.")
        dummy_args, _ = dummy_parser.parse_known_args()
        d_dir = dummy_args.data_dir
        fold_options = [d for d in os.listdir(d_dir)
                        if os.path.isdir(os.path.join(d_dir, d))]

        group.add_argument("--fold", type=str, help="Batch size during training.",
                           choices=fold_options, required=True)


def add_training_options(parser):
    group = parser.add_argument_group('deformer_training')
    group.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=1e-05, type=float, help="Learning rate.")
    
    group.add_argument("--weight_decay", default=0.0001,
                       type=float, help="Optimizer weight decay.")
    group.add_argument("--save_interval", default=5, type=int,
                       help="Save checkpoints and run validation each N epochs")
    group.add_argument("--num_steps", default=200_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--f1_score_variant", type=str, help="The variant of the f1-score to use.",
                       choices=["binary", "micro", "macro", "weighted", "samples"], default="micro")
    group.add_argument("--optimizer", type=str, help="The optimizer to use.",
                       choices=["Adam", "AdamW"], default="AdamW")



def add_save_dir_path(parser, default_save_dir):
    group = parser.add_argument_group('save_directory')
    group.add_argument("--save_dir", default=default_save_dir,
                    type=str, help="Directory for saving checkpoints or results.")


def add_base_dir_path(parser):
    group = parser.add_argument_group('base_directory')
    group.add_argument("--base_dir", required=True, 
                    type=str, help="Base diretory of the stored models for each fold.")


def add_seed(parser):
    group = parser.add_argument_group('seed')
    group.add_argument("--seed", default=420, type=int, help="For fixing random seed.")


def add_multimodal_option(parser):
    group = parser.add_argument_group('multimodal')
    group.add_argument("--multimodal", choices=[0, 1], default=1, type=int,
                       help="Whether the model is multimodal or not.")


def add_modality_option(parser):
    group = parser.add_argument_group('modality')
    group.add_argument("--modality", choices=['eeg', 'ppg', 'eda', 'resp'],
                       default='eeg', type=str, help="Different modalities.")


def add_evaluation_options(parser):
    group = parser.add_argument_group('evaluation')
    group.add_argument("--split", choices=['validation', 'test'],
                       required=True, type=str, help="The data split(s) used for evaluation.")
    group.add_argument("--batch_size", default=32, type=int, help="Batch size during evaluation.")


def add_task_option(parser):
    task_choices = []
    for name, obj in inspect.getmembers(tasks):
        if inspect.isclass(obj):
            cls_name = obj.__name__
            if cls_name != 'ABC' and cls_name != 'Task':
                task_choices.append(cls_name)
    if len(task_choices) == 0:
        raise Exception('No task options found.')

    group = parser.add_argument_group('task')
    group.add_argument("--task", choices=task_choices,
                       default=task_choices[0], type=str, help="Different tasks.")


def train_args(cross_validate=False):
    parser = ArgumentParser()
    add_base_options(parser)
    add_model_name_option(parser)
    add_multimodal_option(parser)
    add_seed(parser)
    add_data_options(parser, cross_validate)
    add_task_option(parser)
    add_training_options(parser)

    dummy_parser = ArgumentParser(add_help=False)
    add_model_name_option(dummy_parser)

    if not is_multimodal():
        add_modality_option(parser)
        add_modality_option(dummy_parser)

    dummy_args, _ = dummy_parser.parse_known_args()
    model_name = dummy_args.model_name

    default_out_dim = get_output_size_from_task()

    parser_group = parser.add_argument_group('model')

    get_model_cls(model_name).add_model_options(
        parser_group=parser_group,
        default_out_dim=default_out_dim,
        modality=None if is_multimodal() else dummy_args.modality
    )

    timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")

    if is_multimodal():
        default_save_dir = os.path.join(os.getcwd(), 'save', 
        				'multimodal', model_name, timestamp)
    else:
        modality = dummy_args.modality
        default_save_dir = os.path.join(os.getcwd(), 'save', 
        				'unimodal', model_name, modality, timestamp)
    add_save_dir_path(parser, default_save_dir=default_save_dir)
    return parser.parse_args()


def late_fusion_evaluation_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser, cross_validate=True)
    add_evaluation_options(parser)
    add_seed(parser)

    # TODO: Rename to modality_base_dirs
    parser.add_argument("--modality_save_dirs", type=str, nargs="+",
                        required=True, help="Save directories of the individual modalities.")

    timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    default_save_dir = os.path.join(os.getcwd(), 'save', 
        			    'multimodal', 'LateFusionDeformer', timestamp)
    add_save_dir_path(parser, default_save_dir=default_save_dir)
    return parser.parse_args()


def evaluation_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser, cross_validate=True)
    add_evaluation_options(parser)
    add_seed(parser)
    add_base_dir_path(parser)

    dummy_parser = ArgumentParser(add_help=False)
    add_base_dir_path(dummy_parser)
    dummy_args, _ = dummy_parser.parse_known_args()
    base_dir = dummy_args.base_dir
    
    add_save_dir_path(parser, default_save_dir=base_dir)
    return parser.parse_args()


def is_multimodal():
    dummy_parser = ArgumentParser(add_help=False)
    add_multimodal_option(dummy_parser)
    dummy_args, _ = dummy_parser.parse_known_args()
    multimodal = dummy_args.multimodal
    return multimodal


def model_parser(model_path=None):
    parser = ArgumentParser()
    # Args specified by the user (all other will be loaded from the model)
    add_base_options(parser)
    return parse_and_load_from_model(parser, model_path)


def get_pass_through_args(args):
    parser_args = [arg for arg in dir(args) if not arg.startswith('_')]
    pass_trough_args = []
    for arg in parser_args:
        pass_trough_args.append('--' + arg)
        val = getattr(args, arg)

        if isinstance(val, list):
            for v in val:
                v = str(v)
                pass_trough_args.append(v)
        else:
            val = str(val)
            pass_trough_args.append(val)

    return pass_trough_args
