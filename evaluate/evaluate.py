# For validating and testing all models except late fusion.

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


def get_f1_score_variant(model_path):
    args_path = os.path.join(os.path.dirname(model_path), '..', 'args.json')

    # Load args from model
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)
    f1_score_variant = model_args["f1_score_variant"]

    return f1_score_variant


def run_evaluation(args, logs, task):
    folds = [k for k in logs.keys() if 'mean' not in k and 'deviation' not in k]

    task_tools = getattr(tasks, task)

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
        if args.split == 'validation':
            data_loader = evaluation_data[f]
        elif args.split == "test":
            data_loader = evaluation_data
        else:
            raise Exception('Unknown evaluation split.')

        y_targets = get_y_targets(data_loader, task_tools)

        step = logs[f]['step']
        model_ckpt = f"model{step:09d}.pt" 
        model_path = os.path.join(args.save_dir, f, model_ckpt)
        model_args = model_parser(model_path=model_path)

        model, model_device = get_model(args, model_path, model_args)

        f1_score_variant = get_f1_score_variant(model_path)

        if model_args.multimodal:
            modality = None
        else: 
            modality = model_args.modality

        y_predictions = run_model_on_eval(
            model, data_loader, model_device, modality
        )

        predicted_labels = torch.argmax(y_predictions, dim=-1)
        accuracy = accuracy_score(y_targets.cpu().detach().numpy(), predicted_labels.cpu().detach().numpy())

        f1 = f1_score(
            y_targets.cpu().detach().numpy(),
            predicted_labels.cpu().detach().numpy(),
            # When all predictions and labels are negative
            zero_division=1.0,
            average=f1_score_variant
        )

        accuracies_for_best_scores.append(accuracy)
        best_scores.append(f1)

        evaluation_results[f] = {}
        evaluation_results[f]['step'] = step
        evaluation_results[f]['accuracy'] = accuracy
        evaluation_results[f][f'{f1_score_variant} f1-score'] = f1.item()

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

    model_args_path = os.path.join(args.save_dir, 'args.json')
    # Load args from model
    assert os.path.exists(model_args_path), 'Arguments json file was not found!'
    with open(model_args_path, 'r') as fr:
        model_args = json.load(fr)

    logs_path = os.path.join(args.save_dir, 'cross_validation.json')
    assert os.path.exists(logs_path), 'Logs json file was not found!'
    with open(logs_path, 'r') as fr:
        logs = json.load(fr)

    task = model_args['task']
    
    print(f"Creating {args.split} results:")
    evaluation_results = run_evaluation(args, logs, task)

    save_dict(evaluation_results, args.save_dir, args.split)
    print(f'Stored results in {args.save_dir}')
    

if __name__ == '__main__':
    main()
