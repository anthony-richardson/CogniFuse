import os
import numpy as np
import torch
import torch.nn.functional as F
import json
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.model_util import count_parameters


def log2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def cross_validate(folds, save_dir: str, f1_score_variant: str):
    best_fold_checkpoints = {}
    best_scores = []
    accuracies_for_best_scores = []

    for f in folds:
        log_dir = os.path.join(save_dir, f)
        log_file_name = [p for p in os.listdir(log_dir)
                         if os.path.isfile(os.path.join(log_dir, p)) and
                         'events' in p][0]
        log_path = os.path.join(log_dir, log_file_name)

        df = log2pandas(log_path)
        df_f1 = df[df.metric == f'Loss/validation_{f1_score_variant}_f1_score']

        checkpoint_steps = [
            int(p.strip('model').strip('.pt')) for p in os.listdir(log_dir)
            if os.path.isfile(os.path.join(log_dir, p)) and 'model' in p
        ]

        df_f1 = df_f1[df_f1['step'].isin(checkpoint_steps)]

        best_checkpoint = df_f1.loc[df_f1['value'].idxmax()]
        value = best_checkpoint.value.item()
        step = int(best_checkpoint.step.item())

        df_acc = df[df.metric == 'Loss/validation_accuracy']
        df_acc = df_acc[df_acc.step == step]
        if df_acc.shape[0] > 1:
            df_acc = df_acc.iloc[0]
            raise Warning("A step has been logged more than once.")

        acc = df_acc.value.item()

        best_fold_checkpoints[f] = {
            'step': step,
            'accuracy': acc,
            f'{f1_score_variant} f1-score': value
        }

        best_scores.append(value)
        accuracies_for_best_scores.append(acc)

    avg_score = np.mean(best_scores)
    std_score = np.std(best_scores)
    best_fold_checkpoints[f'{f1_score_variant} f1-score mean'] = avg_score
    best_fold_checkpoints[f'{f1_score_variant} f1-score standard deviation'] = std_score

    avg_acc = np.mean(accuracies_for_best_scores)
    std_acc = np.std(accuracies_for_best_scores)
    best_fold_checkpoints['accuracy mean'] = avg_acc
    best_fold_checkpoints['accuracy standard deviation'] = std_acc

    return best_fold_checkpoints


def save_dict(dictionary, save_dir, name):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not name.endswith('.json'):
        name += '.json'
    save_path = os.path.join(save_dir, name)
    with open(save_path, 'w') as fw:
        json.dump(dictionary, fw, indent=4)


def save_args(args, create_model_fn):
    dummy_model = create_model_fn(args)
    num_parameters = count_parameters(dummy_model)
    args_dict = dict(vars(args))
    args_dict['nr_of_parameters'] = num_parameters
    save_dict(args_dict, args.save_dir, 'args')


def get_y_targets(data_loader, task_tools):
    targets = []
    for modality_data, meta_info in data_loader:
        y_target = task_tools.map_meta_info_to_class(
            task_tools, meta_info=meta_info)
        targets.append(y_target)
    targets = torch.cat(targets, dim=0)
    return targets


def run_model_on_eval(model, data_loader, device, modality=None):
    predictions = []
    with torch.no_grad():
        for modality_data, meta_info in data_loader:
            if modality is None:
                x = [modality_data[modality_name].type(torch.float).to(device) for
                        modality_name in ['eeg', 'ppg', 'eda', 'resp']]
            else:
                x = modality_data[modality].type(torch.float).to(device)

            output = model(x)
            y_prediction = F.softmax(output, dim=-1)
            predictions.append(y_prediction)

    predictions = torch.cat(predictions, dim=0)
    return predictions
