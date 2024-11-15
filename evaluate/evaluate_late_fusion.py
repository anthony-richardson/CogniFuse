# For validating and testing late fusion 
# based on unimodal cross validation results.
# Note that script this also supports using different models for the 
# different modalities as weel as fusing multiple models that use the same modality. 

import os
import json
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from utils.fixseed import fixseed
from utils.parser_util import late_fusion_evaluation_args, model_parser
from utils.eval_util import run_model_on_eval, save_dict, get_y_targets
from utils.model_util import create_model, load_model
from load.load_data import get_data_loader
import utils.tasks as tasks


def combine_unimodal_args(args):
    modality_save_dirs = args.modality_save_dirs

    task_list = []
    dir_to_modality = {}
    combined_args = {}
    parameter_sum = 0

    for mod_save_dir in modality_save_dirs:
        unimodal_args_path = os.path.join(os.getcwd(), mod_save_dir, 'args.json')

        assert os.path.exists(unimodal_args_path), 'Arguments json file was not found!'
        with open(unimodal_args_path, 'r') as fr:
            unimodal_args = json.load(fr)

        modality = unimodal_args['modality']
        dir_to_modality[mod_save_dir] = modality
        combined_args[modality] = {}
        task_list.append(unimodal_args['task'])
        parameter_sum += unimodal_args['nr_of_parameters']

        for key in unimodal_args.keys():
            combined_args[modality][key] = unimodal_args[key]

    combined_args['late_fusion'] = {}
    combined_args['late_fusion']['nr_of_parameters'] = parameter_sum

    assert len(set(task_list)) == 1, 'The tasks solved by the modalities are not matching!'
    task = task_list[0]

    assert len(set(dir_to_modality.values())) == len(dir_to_modality.values()), 'The same modality is used more than once!'

    return combined_args, dir_to_modality, task


def combine_unimodal_cross_validation_logs(args, dir_to_modality):
    modality_save_dirs = args.modality_save_dirs

    combined_logs = {}

    dummy_dir = modality_save_dirs[0]
    dummy_log_path = os.path.join(os.getcwd(), dummy_dir, 'cross_validation.json')
    assert os.path.exists(dummy_log_path), 'Cross validation json file was not found!'
    with open(dummy_log_path, 'r') as fr:
        dummy_log = json.load(fr)
    for key in dummy_log.keys():
        combined_logs[key] = {}

    folds = [d for d in os.listdir(dummy_dir)
             if os.path.isdir(os.path.join(dummy_dir, d))]
    modalities = []

    for mod_save_dir in modality_save_dirs:
        cross_validation_log_path = os.path.join(os.getcwd(), mod_save_dir, 'cross_validation.json')
        
        assert os.path.exists(cross_validation_log_path), 'Cross validation json file was not found!'
        with open(cross_validation_log_path, 'r') as fr:
            cross_validation_log = json.load(fr)

        modality = dir_to_modality[mod_save_dir] 
        modalities.append(modality)

        for key in cross_validation_log.keys():
            assert key in combined_logs.keys(), 'The keys of the modality log files are not matching!'
            combined_logs[key][modality] = cross_validation_log[key]

    return combined_logs, folds


def get_unimodal_model(args, model_path, model_args):
    unimodal_model = create_model(model_args)
    model_state_dict = torch.load(
        model_path, map_location='cpu', weights_only=False)
    load_model(unimodal_model, model_state_dict)

    if args.cuda:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    unimodal_model.to(device)

    unimodal_model.eval()
    return unimodal_model, device


def calc_modality_weights(fold, modalities, combined_logs):
    scores = {}
    for modality in modalities:
        modality_f1_score = combined_logs[fold][modality]['f1-score']
        scores[modality] = modality_f1_score
    scores_sum = sum(scores.values())
    weights = {}
    for key in scores.keys():
        weights[key] = scores[key] / scores_sum
    return weights


def run_late_fusion(args, combined_logs, combined_args, folds, task, modalities):
    task_tools = getattr(tasks, task)
    nr_classes = combined_args[list(modalities)[0]]['out_dim']

    if args.split == 'validation':
        evaluation_data = {}
        # Creating data loaders for the validation data of each fold.
        for f in folds:
            fold_data_dir = os.path.join(args.data_dir, f)
            fold_validation_data = get_data_loader(
                batch_size=args.batch_size,
                tasks=task_tools.get_mapper().keys(),
                data_dir=fold_data_dir,
                split=args.split,
                # Shuffling must be turned off for consistent labeling.
                shuffle=False
            )
            evaluation_data[f] = fold_validation_data
    elif args.split == "test":
        # Creating a data loader for the test data.
        evaluation_data = get_data_loader(
            batch_size=args.batch_size,
            tasks=task_tools.get_mapper().keys(),
            data_dir=args.data_dir,
            split=args.split,
            # Shuffling must be turned off for consistent labeling.
            shuffle=False
        )
    else:
        raise Exception('Unknown evaluation split.')

    accuracies_for_best_scores = []
    best_scores = []
    for f in tqdm(folds):
        modality_weights = calc_modality_weights(
            f, modalities, combined_logs
        )

        if args.split == 'validation':
            data_loader = evaluation_data[f]
        elif args.split == "test":
            data_loader = evaluation_data
        else:
            raise Exception('Unknown evaluation split.')

        y_targets = get_y_targets(data_loader, task_tools)

        unimodal_predictions_list = []
        for modality in modalities:
            step = combined_logs[f][modality]['step']
            model_ckpt = f"model{step:09d}.pt" 
            modality_save_dir = combined_args[modality]['save_dir']
            model_path = os.path.join(modality_save_dir, f, model_ckpt)
            model_args = model_parser(model_path=model_path)

            unimodal_model, model_device = get_unimodal_model(args, model_path, model_args)

            unimodal_predictions = run_model_on_eval(
                unimodal_model, data_loader, model_device, modality=modality)

            unimodal_predictions = torch.mul(
                unimodal_predictions, modality_weights[modality]
            )
            unimodal_predictions_list.append(unimodal_predictions)

        unimodal_predictions_tensor = torch.stack(unimodal_predictions_list, dim=0)  
        fused_predictions = torch.sum(unimodal_predictions_tensor, dim=0)

        predicted_labels = torch.argmax(fused_predictions, dim=-1)
        accuracy = accuracy_score(y_targets.cpu().detach().numpy(), predicted_labels.cpu().detach().numpy())

        # Normal f1 score like in https://dl.acm.org/doi/pdf/10.1145/2070481.2070516
        avg = 'binary'
        if nr_classes > 2:
            # Unweighted average for all classes
            avg = 'macro'

        f1 = f1_score(
            y_targets.cpu().detach().numpy(),
            predicted_labels.cpu().detach().numpy(),
            # When all predictions and labels are negative
            zero_division=1.0,
            # Unweighted average for all classes
            average=avg
        )

        accuracies_for_best_scores.append(accuracy)
        best_scores.append(f1)

        combined_logs[f]['early fusion'] = {}
        combined_logs[f]['early fusion']['accuracy'] = accuracy
        combined_logs[f]['early fusion']['f1-score'] = f1.item()

    avg_acc = np.mean(accuracies_for_best_scores)
    std_acc = np.std(accuracies_for_best_scores)
    combined_logs['accuracy mean']['early fusion'] = avg_acc.item()
    combined_logs['accuracy standard deviation']['early fusion'] = std_acc.item()

    avg_score = np.mean(best_scores)
    std_score = np.std(best_scores)
    combined_logs['f1-score mean']['early fusion'] = avg_score.item()
    combined_logs['f1-score standard deviation']['early fusion'] = std_score.item()


def main():
    args = late_fusion_evaluation_args()
    fixseed(args.seed)

    combined_args, dir_to_modality, task = combine_unimodal_args(args)
    
    combined_logs, folds = combine_unimodal_cross_validation_logs(args, dir_to_modality)
    run_late_fusion(args, combined_logs, combined_args, folds, task, dir_to_modality.values())

    args_dict = dict(vars(args))
    for key in args_dict:
        if key != 'split':
            combined_args['late_fusion'][key] = getattr(args, key)

    save_dict(combined_args, args.save_dir, 'args')
    save_dict(combined_logs, args.save_dir, args.split)
    print(f'Stored results in {args.save_dir}')
    

if __name__ == '__main__':
    main()
