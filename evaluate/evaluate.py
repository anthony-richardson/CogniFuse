# TODO: For validating and testing all models except late fusion.

# TODO: integrate into cross validation script at the end with test data as param (in case not late fusion)

import os
import json
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from utils.fixseed import fixseed
from utils.parser_util import evaluation_args, model_parser
from utils.eval_util import run_model_on_eval, save_dict, get_y_targets
from utils.model_util import create_model, load_model
from load.load_data import get_data_loader
import utils.tasks as tasks


'''def combine_unimodal_args(args):
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

    return combined_args, dir_to_modality, task'''


'''def combine_cross_validation_logs(args, dir_to_modality):
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

    return combined_logs, folds'''


def get_model(args, model_path, model_args):
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


'''def calc_modality_weights(fold, modalities, combined_logs):
    scores = {}
    for modality in modalities:
        modality_f1_score = combined_logs[fold][modality]['f1-score']
        scores[modality] = modality_f1_score
    scores_sum = sum(scores.values())
    weights = {}
    for key in scores.keys():
        weights[key] = scores[key] / scores_sum
    return weights'''


def run_evaluation(args, logs, task):
    folds = [k for k in logs.keys() if 'mean' not in k and 'deviation' not in k]

    #print(folds)

    task_tools = getattr(tasks, task)
    #nr_classes = combined_args[list(modalities)[0]]['out_dim']

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

    evaluation_results = {}

    accuracies_for_best_scores = []
    best_scores = []
    for f in tqdm(folds):
        #modality_weights = calc_modality_weights(
        #    f, modalities, combined_logs
        #)

        if args.split == 'validation':
            data_loader = evaluation_data[f]
        elif args.split == "test":
            data_loader = evaluation_data
        else:
            raise Exception('Unknown evaluation split.')

        y_targets = get_y_targets(data_loader, task_tools)

        step = logs[f]['step']
        model_ckpt = f"model{step:09d}.pt" 
        #modality_save_dir = combined_args[modality]['save_dir']
        model_path = os.path.join(args.base_dir, f, model_ckpt)
        model_args = model_parser(model_path=model_path)

        model, model_device = get_model(args, model_path, model_args)

        if model_args.multimodal:
            modality = None
        else: 
            modality = model_args.modality

        y_predictions = run_model_on_eval(
            model, data_loader, model_device, modality
        )


        #unimodal_predictions_list = []
        #for modality in modalities:
        #    step = combined_logs[f][modality]['step']
        #    model_ckpt = f"model{step:09d}.pt" 
        #    modality_save_dir = combined_args[modality]['save_dir']
        #    model_path = os.path.join(modality_save_dir, f, model_ckpt)
        #    model_args = model_parser(model_path=model_path)

        #    unimodal_model, model_device = get_unimodal_model(args, model_path, model_args)

        #    unimodal_predictions = run_model_on_eval(
        #        unimodal_model, data_loader, model_device, modality=modality)

            #unimodal_predictions = torch.mul(
            #    unimodal_predictions, modality_weights[modality]
            #)
            #unimodal_predictions_list.append(unimodal_predictions)

        #unimodal_predictions_tensor = torch.stack(unimodal_predictions_list, dim=0)  
        #fused_predictions = torch.sum(unimodal_predictions_tensor, dim=0)

        predicted_labels = torch.argmax(y_predictions, dim=-1)
        accuracy = accuracy_score(y_targets.cpu().detach().numpy(), predicted_labels.cpu().detach().numpy())

        # Normal f1 score like in https://dl.acm.org/doi/pdf/10.1145/2070481.2070516
        #avg = 'binary'
        #if nr_classes > 2:
        #    # Unweighted average for all classes
        #    avg = 'macro'

        avg = 'micro'

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

        evaluation_results[f] = {}
        evaluation_results[f]['step'] = step
        evaluation_results[f]['accuracy'] = accuracy
        evaluation_results[f]['micro f1-score'] = f1.item()

        #combined_logs[f]['late fusion'] = {}
        #combined_logs[f]['late fusion']['accuracy'] = accuracy
        #combined_logs[f]['late fusion']['micro f1-score'] = f1.item()

    avg_acc = np.mean(accuracies_for_best_scores)
    std_acc = np.std(accuracies_for_best_scores)
    evaluation_results['accuracy mean'] = avg_acc.item()
    evaluation_results['accuracy standard deviation'] = std_acc.item()

    avg_score = np.mean(best_scores)
    std_score = np.std(best_scores)
    evaluation_results['micro f1-score mean'] = avg_score.item()
    evaluation_results['micro f1-score standard deviation'] = std_score.item()

    return evaluation_results


def main():
    args = evaluation_args()
    fixseed(args.seed)

    model_args_path = os.path.join(os.path.dirname(args.base_dir), 'args.json')
    # Load args from model
    assert os.path.exists(model_args_path), 'Arguments json file was not found!'
    with open(model_args_path, 'r') as fr:
        model_args = json.load(fr)

    logs_path = os.path.join(os.path.dirname(args.base_dir), 'cross_validation.json')
    assert os.path.exists(logs_path), 'Logs json file was not found!'
    with open(logs_path, 'r') as fr:
        logs = json.load(fr)

    task = model_args['task']

    #print(model_args)
    #print(args)
    #print(logs)

    #exit()

    #combined_args, dir_to_modality, task = combine_unimodal_args(args)
    
    #combined_logs, folds = combine_unimodal_cross_validation_logs(args, dir_to_modality)
    evaluation_results = run_evaluation(args, logs, task)

    #args_dict = dict(vars(args))
    #for key in args_dict:
    #    if key != 'split':
    #        combined_args['late_fusion'][key] = getattr(args, key)

    #save_dict(combined_args, args.save_dir, 'args')
    save_dict(evaluation_results, args.base_dir, args.split)
    print(f'Stored results in {args.base_dir}')
    

if __name__ == '__main__':
    main()
