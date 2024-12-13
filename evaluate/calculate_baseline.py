import numpy as np
import os
import inspect
from sklearn.metrics import accuracy_score, f1_score

import utils.tasks as tasks


def get_task_data(complete_data, task_list=None):
    filtered_data = []
    # Filtering
    for d in complete_data:
        d_task = d['task']

        if d['difficulty'] > 0:
            d_task += f'_{d['difficulty']}'

            #print(d['task'])

        if (task_list is None or d_task in task_list):

            #print(f'in: {d['task']}')

            filtered_data.append((
                d['participant_id'],
                d_task,
                d['eeg'],
                d['ppg'],
                d['eda'],
                d['resp'],
            ))
        
        #else:
        #    if 'SwitchBackAuditive' in d['task']:
        #        print(d['task'])
            #else:
            #    print(f'in: {d['task']}')

    # TODO: change preprocessing of data so that it is already structured as a scenario
    #   Then remove the logic above accordingly.
    filtered_data = np.array(filtered_data, dtype=[
            ('participant_id', 'i4'),
            ('scenario', 'U30'),
            ('eeg', 'O'),
            ('ppg', 'O'),
            ('eda', 'O'),
            ('resp', 'O')
        ]
    )
    return filtered_data


def calc_baseline(targets):
    class_count = {int(c):0 for c in np.unique(targets)}

    for t in targets:
        class_count[int(t)] += 1

    print(f'Class count: {class_count}')


    values, counts = np.unique(targets, return_counts=True)
    ind = np.argmax(counts)
    most_frequent_cls = values[ind]
    #print(values[ind])
    #most_frequent_cls = max(targets, key=lambda key: targets[key])

    print(most_frequent_cls)
    

    predictions = np.array([most_frequent_cls] * len(targets))

    accuracy = accuracy_score(
        targets,
        predictions
    )

    # Normal f1 score like in https://dl.acm.org/doi/pdf/10.1145/2070481.2070516
    """avg = 'binary'
    if len(class_count) > 2:
        # Unweighted average for all classes
        avg = 'macro'"""

    # micro f1 score is the same as accuracy on binary task.
    # The cognifit paper uses accuracy: https://dl.acm.org/doi/pdf/10.1145/2070481.2070516
    avg = 'micro'

    #avg = 'macro'

    f1 = f1_score(
        targets,
        predictions,
        # When all predictions and labels are negative
        zero_division=1.0,
        # Unweighted average for all classes
        average=avg
    )

    baseline_metrics = {
        'accuracy': accuracy,
        'f1-score': f1
    }

    return baseline_metrics


def main():
    train_data = np.load(os.path.join(os.getcwd(), 'data', 'folds', '1', 'train.npy'), allow_pickle=True)
    validation_data = np.load(os.path.join(os.getcwd(), 'data', 'folds', '1', 'validation.npy'), allow_pickle=True)
    complete_data = np.append(train_data, validation_data)

    '''task_choices = [
        'SwitchingTaskPresence', 
        'SwitchBackAuditivePresenceRelax', 
        'SwitchBackAuditivePresence',
        'VisualSearchTaskPresence'
    ]'''
    task_choices = []
    for name, obj in inspect.getmembers(tasks):
        if inspect.isclass(obj):
            cls_name = obj.__name__
            if cls_name != 'ABC' and cls_name != 'Task':
                task_choices.append(cls_name)

    for t in task_choices:
        task_tools = getattr(tasks, t)
        task_list = task_tools.get_mapper().keys()

        task_data = get_task_data(complete_data, task_list=task_list)
        print(t + f': {len(task_data)}')

        if len(task_data) > 0:   
            class_count = {t:0 for t in task_list}
            for d in task_data:
                class_count[d[f'scenario']] += 1
            print(class_count)

            targets = task_tools.map_meta_info_to_class(
                task_tools, meta_info={'scenario': task_data['scenario']}
            ).numpy()

            baseline = calc_baseline(targets)
            print(f'{t} Baseline: {baseline}')


if __name__ == '__main__':
    main()