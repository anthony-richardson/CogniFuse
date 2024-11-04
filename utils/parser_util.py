from argparse import ArgumentParser
import argparse
import os
import json
import inspect
import datetime

import utils.tasks as tasks


def parse_and_load_from_model(parser, is_unimodal, model_path):
    if model_path is None:
        model_path = get_model_path_from_args()

    args_path = os.path.join(os.path.dirname(model_path), '..', 'args.json')

    # load args from model
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    #group_names = ['dataset', model_type]

    add_seed(parser)
    group_names = ['seed']

    default_out_dim = model_args['out_dim']

    if is_unimodal:
        try:
            modality = model_args['modality']
        except KeyError:
            raise KeyError('modality argument is missing from model args.')
        add_unimodal_deformer_model_options(parser, modality, default_out_dim)
        group_names.append('unimodal_deformer_model')
    else:
        add_multimodal_deformer_model_options(parser, default_out_dim)
        group_names.append('multimodal_deformer_model')

    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    #add_data_options(parser, cross_validate=False)

    #if model_type == "deformer":
    #    add_unimodal_deformer_model_options(parser)
    #else:
    #    print(f'Warning: model type {model_type} is unknown.')

    args, _ = parser.parse_known_args()
    args_to_overwrite = []
    for group_name in group_names:
        #print(group_name)
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    #if model_path is None:
    #    model_path = get_model_path_from_args()

    '''args_path = os.path.join(os.path.dirname(model_path), 'args.json')

    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)'''

    # overwrite args from model
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
        dummy_parser = ArgumentParser()
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
    dummy_parser = ArgumentParser()
    dummy_parser.add_argument('--task')
    dummy_args, _ = dummy_parser.parse_known_args()
    task_tools = getattr(tasks, dummy_args.task)
    nr_classes = len(set(task_tools.get_mapper().values()))
    return nr_classes


def add_base_options(parser):
    group = parser.add_argument_group('base')
    #group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    #group.add_argument("--cuda", default=False, action='store_true', help="Use cuda device, otherwise use CPU.")
    group.add_argument("--cuda", choices=[0, 1], default=0, type=int, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    #group.add_argument("--seed", default=420, type=int, help="For fixing random seed.")


def add_data_options(parser, cross_validate=False):
    group = parser.add_argument_group('dataset')
    group.add_argument("--data_dir",
                       default=os.path.join(os.getcwd(), 'data', 'folds'),
                       type=str, help="Directory where the data splits are stored.")

    if not cross_validate:
        # Determining the fold options.
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("--data_dir",
                                  default=os.path.join(os.getcwd(), 'data', 'folds'),
                                  type=str, help="Directory where the data splits are stored.")
        dummy_args, _ = dummy_parser.parse_known_args()
        d_dir = dummy_args.data_dir
        fold_options = [d for d in os.listdir(d_dir)
                        if os.path.isdir(os.path.join(d_dir, d))]

        group.add_argument("--fold", type=str, help="Batch size during training.",
                           choices=fold_options, required=True)


def add_multimodal_deformer_model_options(parser, default_out_dim):
    group = parser.add_argument_group('multimodal_deformer_model')

    #group.add_argument("--num_time", default=[4 * 128, 6 * 32, 4 * 32, 10 * 32], type=int, nargs="+",
    #                   help="Number of time steps for all modalities")
    group.add_argument("--num_time_eeg", default=4 * 128, type=int,
                       help="Number of time steps for the eeg modality")
    group.add_argument("--num_time_ppg", default=6 * 32, type=int,
                       help="Number of time steps for the ppg modality")
    group.add_argument("--num_time_eda", default=4 * 32, type=int,
                       help="Number of time steps for the eda modality")
    group.add_argument("--num_time_resp", default=10 * 32, type=int,
                       help="Number of time steps for the resp modality")

    #group.add_argument("--num_chan", default=[16, 1, 1, 1], type=int, nargs="+",
    #                   help="Number of channels for all modalities")
    group.add_argument("--num_chan_eeg", default=16, type=int,
                       help="Number of channels for the eeg modality")
    group.add_argument("--num_chan_ppg", default=1, type=int,
                       help="Number of channels for the eeg modality")
    group.add_argument("--num_chan_eda", default=1, type=int,
                       help="Number of channels for the eeg modality")
    group.add_argument("--num_chan_resp", default=1, type=int,
                       help="Number of channels for the eeg modality")

    group.add_argument("--depth", default=4, type=int, help="Depth of kernels")
    group.add_argument("--heads", default=16, type=int, help="Number of heads")
    group.add_argument("--dim_head", default=16, type=int, help="Dimension of heads")
    group.add_argument("--mlp_dim", default=16, type=int, help="Dimension of MLP")
    group.add_argument("--num_kernel", default=64, type=int, help="Number of kernels")
    group.add_argument("--temporal_kernel", default=13, type=int, help="Length of temporal kernels")
    group.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    # TODO: analyse what rate is better
    #group.add_argument("--dropout", default=0.0, type=float, help="Dropout rate")
    group.add_argument("--emb_dim", default=256, type=int, help="Embedding dimension")
    group.add_argument("--out_dim", default=default_out_dim, type=int,
                       help="Size of the output. For classification tasks, this is the number of classes.")

    #group.add_argument("--out_dim", default=2, type=int,
    #                   help="Size of the output. For classification tasks, this is the number of classes.")


'''def add_crossmodal_deformer_model_options(parser):
    group = parser.add_argument_group('cross_modal_deformer_model')

    #group.add_argument("--num_time", default=[4 * 128, 6 * 32, 4 * 32, 10 * 32], type=int, nargs="+",
    #                   help="Number of time steps for all modalities")
    group.add_argument("--num_time_eeg", default=4 * 128, type=int,
                       help="Number of time steps for the eeg modality")
    group.add_argument("--num_time_ppg", default=6 * 32, type=int,
                       help="Number of time steps for the ppg modality")
    group.add_argument("--num_time_eda", default=4 * 32, type=int,
                       help="Number of time steps for the eda modality")
    group.add_argument("--num_time_resp", default=10 * 32, type=int,
                       help="Number of time steps for the resp modality")

    #group.add_argument("--num_chan", default=[16, 1, 1, 1], type=int, nargs="+",
    #                   help="Number of channels for all modalities")
    group.add_argument("--num_chan_eeg", default=16, type=int,
                       help="Number of channels for the eeg modality")
    group.add_argument("--num_chan_ppg", default=1, type=int,
                       help="Number of channels for the eeg modality")
    group.add_argument("--num_chan_eda", default=1, type=int,
                       help="Number of channels for the eeg modality")
    group.add_argument("--num_chan_resp", default=1, type=int,
                       help="Number of channels for the eeg modality")

    group.add_argument("--depth", default=4, type=int, help="Depth of kernels")
    group.add_argument("--heads", default=16, type=int, help="Number of heads")
    group.add_argument("--dim_head", default=16, type=int, help="Dimension of heads")
    group.add_argument("--mlp_dim", default=16, type=int, help="Dimension of MLP")
    group.add_argument("--num_kernel", default=64, type=int, help="Number of kernels")
    group.add_argument("--emb_dim", default=256, type=int, help="Embedding dimension")
    group.add_argument("--out_dim", default=2, type=int,
                       help="Size of the output. For classification tasks, this is the number of classes.")
    group.add_argument("--temporal_kernel", default=13, type=int, help="Length of temporal kernels")
    group.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    # TODO: analyse what rate is better
    #group.add_argument("--dropout", default=0.0, type=float, help="Dropout rate")'''
    

def add_unimodal_deformer_model_options(parser, modality, default_out_dim):
    if modality == "eeg":
        num_chan = 16
        num_kernel = 64
        num_time = 4 * 128
    else:
        num_chan = 1
        # TODO: later reduce to for example 16 for other modalities (something based on a rule)
        num_kernel = 64
        if modality == "ppg":
            num_time = 6 * 32
        elif modality == "eda":
            num_time = 4 * 32
        elif modality == "resp":
            num_time = 10 * 32
        else:
            raise ValueError(f"Unknown modality: {modality}")

    group = parser.add_argument_group('unimodal_deformer_model')
    # EEG
    #group.add_argument("--num_chan", default=16, type=int, help="Number of channels")
    #group.add_argument("--num_time", default=512, type=int, help="Number of time steps")
    # EDA
    group.add_argument("--num_chan", default=num_chan, type=int, help="Number of channels")
    group.add_argument("--num_time", default=num_time, type=int, help="Number of time steps")
    group.add_argument("--num_kernel", default=num_kernel, type=int, help="Number of kernels")
    group.add_argument("--temporal_kernel", default=13, type=int, help="Length of temporal kernels")
    group.add_argument("--depth", default=4, type=int, help="Depth of kernels")
    group.add_argument("--heads", default=16, type=int, help="Number of heads")
    group.add_argument("--mlp_dim", default=16, type=int, help="Dimension of MLP")
    group.add_argument("--dim_head", default=16, type=int, help="Dimension of heads")
    group.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    # TODO: analyse what rate is better
    #group.add_argument("--dropout", default=0.2, type=float, help="Dropout rate")
    #group.add_argument("--dropout", default=0.0, type=float, help="Dropout rate")
    #group.add_argument("--emb_dim", default=2, type=int,
    #                   help="Size of the output. For classification tasks, this is the number of classes.")
    # TODO: later reduce for other modalities than eeg
    group.add_argument("--emb_dim", default=256, type=int, help="Embedding dimension")
    group.add_argument("--out_dim", default=default_out_dim, type=int,
                       help="Size of the output. For classification tasks, this is the number of classes.")


def add_training_options(parser):
    group = parser.add_argument_group('deformer_training')
    group.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    # TODO: this lr is appropriate for unimodal
    #group.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
    # TODO: this lr is the same as in multimodal for comparison (see if it reaches the same f-score as with higher lr. If so, stick to lower)
    group.add_argument("--lr", default=0.00001, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0001,
                       type=float, help="Optimizer weight decay.")
    '''group.add_argument("--log_interval", default=500, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=5_000, type=int,
                       help="Save checkpoints and run validation each N steps")'''
    group.add_argument("--save_interval", default=5, type=int,
                       help="Save checkpoints and run validation each N epochs")
    group.add_argument("--num_steps", default=200_000, type=int,
                       help="Training will stop after the specified number of steps.")

'''def add_deformer_training_options(parser):
    group = parser.add_argument_group('deformer_training')
    group.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    # TODO: this lr is appropriate for unimodal
    #group.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
    # TODO: this lr is the same as in multimodal for comparison (see if it reaches the same f-score as with higher lr. If so, stick to lower)
    group.add_argument("--lr", default=0.00001, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0001,
                       type=float, help="Optimizer weight decay.")
    group.add_argument("--log_interval", default=500, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=5_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=200_000, type=int,
                       help="Training will stop after the specified number of steps.")'''


'''def add_cross_modal_deformer_training_options(parser):
    group = parser.add_argument_group('deformer_training')
    group.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=0.00001, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0001,
                       type=float, help="Optimizer weight decay.")
    group.add_argument("--log_interval", default=500, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=5_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=200_000, type=int,
                       help="Training will stop after the specified number of steps.")'''


'''def add_tcn_model_options(parser):
    group = parser.add_argument_group('deformer_model')
    group.add_argument("--in_chan", default=40, type=int, help="")
    group.add_argument("--n_src", default=1, type=int, help="")
    group.add_argument("--out_chan", default=2, type=int, help="")
    group.add_argument("--n_blocks", default=5, type=int, help="")
    group.add_argument("--n_repeats", default=2, type=int, help="")
    group.add_argument("--bn_chan", default=64, type=int, help="Bottleneck")
    group.add_argument("--hid_chan", default=128, type=int, help="")
    group.add_argument("--kernel_size", default=3, type=int, help="")
    group.add_argument("--cut_out", default=False, type=bool, help="")


def add_tcn_training_options(parser):
    group = parser.add_argument_group('tcn_training')
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.00001,
                       type=float, help="Optimizer weight decay.")
    group.add_argument("--log_interval", default=500, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=5_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=200_000, type=int,
                       help="Training will stop after the specified number of steps.")'''


def add_save_dir_path(parser, default_save_dir):
    group = parser.add_argument_group('save_directory')
    group.add_argument("--save_dir", default=default_save_dir,
                       type=str, help="Directory for saving checkpoints or results.")


'''def add_model_path_option(parser, model_type):
    group = parser.add_argument_group(f'{model_type}_model_path')
    group.add_argument(f'--{model_type}_model_path', required=True, type=str,
                       help="Path to model####.pt file.")'''


def add_seed(parser):
    group = parser.add_argument_group('seed')
    group.add_argument("--seed", default=420, type=int, help="For fixing random seed.")


def add_multimodal_option(parser):
    group = parser.add_argument_group('multimodal')
    #group.add_argument("--multimodal", default=False, action='store_true',
    #                   help="Whether the model is multimodal or not.")
    group.add_argument("--multimodal", choices=[0, 1], default=0, type=int,
                       help="Whether the model is multimodal or not.")
    #group.add_argument('--test', action=argparse.BooleanOptionalAction, default=False, help='Foo')
    #group.add_argument("--test", choices=[0, 1],
    #                   default=0, type=int, help="Foo.")


def add_fusion_type_option(parser):
    group = parser.add_argument_group('fusion_type')
    group.add_argument("--fusion_type", choices=['crossmodal', 'early', 'intermediate'],
                       default='crossmodal', type=str, help="Different fusion types.")


def add_modality_option(parser):
    group = parser.add_argument_group('modality')
    group.add_argument("--modality", choices=['eeg', 'ppg', 'eda', 'resp'],
                       required=True, type=str, help="Different modalities.")


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

    group = parser.add_argument_group('task')
    group.add_argument("--task", choices=task_choices,
                       required=True, type=str, help="Different tasks.")


def multimodal_deformer_train_args(cross_validate=False):
    parser = ArgumentParser()
    add_base_options(parser)
    add_multimodal_option(parser)
    add_seed(parser)
    add_data_options(parser, cross_validate)
    add_task_option(parser)
    add_training_options(parser)
    add_fusion_type_option(parser)

    default_out_dim = get_output_size_from_task()
    add_multimodal_deformer_model_options(parser, default_out_dim)

    dummy_parser = ArgumentParser()
    add_fusion_type_option(dummy_parser)
    dummy_args, _ = dummy_parser.parse_known_args()
    fusion_type = dummy_args.fusion_type

    '''if fusion_type == 'cross_modal':
        add_multimodal_deformer_model_options(parser)
        #add_cross_modal_deformer_training_options(parser)
    elif fusion_type == 'early_fusion':
        pass
    elif fusion_type == 'late_fusion':
        pass
    else:
        raise ValueError(f'Unknown fusion type: {fusion_type}')'''

    timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    default_save_dir = os.path.join(os.getcwd(), 'save', 'multimodal',
                                    f'{fusion_type}_fusion_deformer', timestamp)
    add_save_dir_path(parser, default_save_dir=default_save_dir)
    return parser.parse_args()


def unimodal_deformer_train_args(cross_validate=False):
    parser = ArgumentParser()
    add_base_options(parser)
    add_multimodal_option(parser)
    add_seed(parser)
    add_data_options(parser, cross_validate)
    #add_model_type_option(parser)
    add_modality_option(parser)
    add_task_option(parser)
    #add_deformer_training_options(parser)
    add_training_options(parser)

    dummy_parser = ArgumentParser()
    add_modality_option(dummy_parser)
    dummy_args, _ = dummy_parser.parse_known_args()
    modality = dummy_args.modality

    timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    default_save_dir = os.path.join(os.getcwd(), 'save', 'unimodal', modality, timestamp)
    add_save_dir_path(parser, default_save_dir=default_save_dir)

    default_out_dim = get_output_size_from_task()
    add_unimodal_deformer_model_options(parser, modality, default_out_dim)
    return parser.parse_args()


def late_fusion_evaluation_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser, cross_validate=True)
    add_evaluation_options(parser)
    add_seed(parser)

    parser.add_argument("--modality_save_dirs", type=str, nargs="+",
                        required=True, help="Save directories of the individual modalities.")

    timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    default_save_dir = os.path.join(os.getcwd(), 'save', 'multimodal',
                                    'late_fusion_deformer', timestamp)
    add_save_dir_path(parser, default_save_dir=default_save_dir)
    return parser.parse_args()


'''def tcn_train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    #add_model_type_option(parser)
    add_modality_option(parser)
    add_task_option(parser)
    add_tcn_model_options(parser)
    add_tcn_training_options(parser)
    default_save_dir = os.path.join(os.getcwd(), 'unimodal', 'save')
    add_save_dir_path(parser, default_save_dir=default_save_dir)
    return parser.parse_args()'''


def is_multimodal():
    dummy_parser = ArgumentParser()
    add_multimodal_option(dummy_parser)
    dummy_args, _ = dummy_parser.parse_known_args()
    multimodal = dummy_args.multimodal
    return multimodal


def model_parser(is_unimodal, model_path=None):
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    return parse_and_load_from_model(parser, is_unimodal, model_path)
