import numpy as np
import os
import inspect
from sklearn.metrics import accuracy_score, f1_score

import utils.tasks as tasks


# TODO: change so that max is only calculated based on the training data count not also validation. And then baseline caluclated bsed on the validation or test data. 
# TODO: add caluclation of test baseline (should be the same but the same data for all folds, meaning the test data)


def get_task_data(data, task_list=None):
    filtered_data = []
    # Filtering
    for d in data:
        #d_task = d['task']
        d_task = d['scenario']

        #if d['difficulty'] > 0:
        #    d_task += f'_{d['difficulty']}'

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


def calc_baseline(targets, most_frequent_train_cls):
    #class_count = {int(c):0 for c in np.unique(targets)}

    #for t in targets:
    #    class_count[int(t)] += 1

    #print(f'Class count: {class_count}')


    #values, counts = np.unique(targets, return_counts=True)
    #ind = np.argmax(counts)
    #most_frequent_cls = values[ind]
    #print(values[ind])
    #most_frequent_cls = max(targets, key=lambda key: targets[key])

    #print(most_frequent_cls)
    

    predictions = np.array([most_frequent_train_cls] * len(targets))

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

    test_data = np.load(os.path.join(os.getcwd(), 'data', 'folds', 'test.npy'), allow_pickle=True)

    task_choices = []
    for name, obj in inspect.getmembers(tasks):
        if inspect.isclass(obj):
            cls_name = obj.__name__
            if cls_name != 'ABC' and cls_name != 'Task':
                task_choices.append(cls_name)

    validation_baselines = {}
    test_baselines = {}

    for t in task_choices:
        validation_baselines[t] = []
        test_baselines[t] = []

    for i in range(1, 11):
        print(f'Calculating baselines for fold {i} ...')

        #validation_baselines[f'{i}'] = {}
        #test_baselines[f'{i}'] = {}

        train_data = np.load(os.path.join(os.getcwd(), 'data', 'folds', f'{i}', 'train.npy'), allow_pickle=True)
        validation_data = np.load(os.path.join(os.getcwd(), 'data', 'folds', f'{i}', 'validation.npy'), allow_pickle=True)
        #complete_data = np.append(train_data, validation_data)

        '''task_choices = [
            'SwitchingTaskPresence', 
            'SwitchBackAuditivePresenceRelax', 
            'SwitchBackAuditivePresence',
            'VisualSearchTaskPresence'
        ]'''


        #task_validation_baselines = []
        #task_test_baselines = []
        for t in task_choices:
            task_tools = getattr(tasks, t)
            task_list = task_tools.get_mapper().keys()

            task_data_train = get_task_data(train_data, task_list=task_list)
            task_data_validation = get_task_data(validation_data, task_list=task_list)
            task_data_test = get_task_data(test_data, task_list=task_list)

            #print(t + f': {len(task_data)}')

            #if len(task_data) > 0:
            #class_count = {t:0 for t in task_list}
            #for d in task_data:
            #    class_count[d[f'scenario']] += 1
            #print(class_count)

            if len(task_data_train) > 0:
                targets_train = task_tools.map_meta_info_to_class(
                    task_tools, meta_info={'scenario': task_data_train['scenario']}
                ).numpy()

                values, counts = np.unique(targets_train, return_counts=True)
                ind = np.argmax(counts)
                most_frequent_train_cls = values[ind]

                targets_validation = task_tools.map_meta_info_to_class(
                    task_tools, meta_info={'scenario': task_data_validation['scenario']}
                ).numpy()

                validation_baseline = calc_baseline(targets_validation, most_frequent_train_cls)
                print(f'{t} Validation Baseline: {validation_baseline}')
                validation_baselines[t].append(validation_baseline)
                #task_validation_baselines.append(validation_baseline)

                targets_test = task_tools.map_meta_info_to_class(
                    task_tools, meta_info={'scenario': task_data_test['scenario']}
                ).numpy()

                test_baseline = calc_baseline(targets_test, most_frequent_train_cls)
                print(f'{t} Test Baseline: {test_baseline}')
                test_baselines[t].append(test_baseline)
                #task_test_baselines.append(test_baseline)

        #validation_baselines.append(validation_baseline)
        #test_baselines.append(test_baseline)

    print("---------------")

    for t in task_choices:

        # Validation mean and standard deviation over folds for f1-score
        val_f1_scores = [e['f1-score'] for e in validation_baselines[t]]
        val_baseline_f1_avg = np.mean(val_f1_scores)
        val_baseline_f1_std = np.std(val_f1_scores)
        print(f'{t} Validation basline f1-score avg: {val_baseline_f1_avg}, '
              f'{t} Validation basline f1-score std: {val_baseline_f1_std}')

        # Validation mean and standard deviation over folds for accuraccies
        #val_acc_scores = [e['accuracy'] for e in validation_baselines]
        val_acc_scores = [e['accuracy'] for e in validation_baselines[t]]
        val_baseline_acc_avg = np.mean(val_acc_scores)
        val_baseline_acc_std = np.std(val_acc_scores)
        print(f'{t} Validation basline accuracy avg: {val_baseline_acc_avg}, '
              f'{t} Validation basline accuracy std: {val_baseline_acc_std}')

        # Test mean and standard deviation over folds for f1-score
        #test_f1_scores = [e['f1-score'] for e in test_baselines]
        test_f1_scores = [e['f1-score'] for e in test_baselines[t]]
        test_baseline_f1_avg = np.mean(test_f1_scores)
        test_baseline_f1_std = np.std(test_f1_scores)
        print(f'{t} Test basline f1-score avg: {test_baseline_f1_avg}, '
              f'{t} Test basline f1-score std: {test_baseline_f1_std}')

        # Test mean and standard deviation over folds for accuraccies
        #test_acc_scores = [e['accuracy'] for e in test_baselines]
        test_acc_scores = [e['accuracy'] for e in test_baselines[t]]
        test_baseline_acc_avg = np.mean(test_acc_scores)
        test_baseline_acc_std = np.std(test_acc_scores)
        print(f'{t} Test basline accuracy avg: {test_baseline_acc_avg}, '
              f'{t} Test basline accuracy std: {test_baseline_acc_std}')

        print("------")

        ##

        # Test mean and standard deviation over folds
        #test_baseline_avg = np.mean(test_baselines)
        #test_baseline_std = np.std(test_baselines)
        #print(f'Test basline avg: {test_baseline_avg}, Test basline std: {test_baseline_std}')
    print("---------------")

if __name__ == '__main__':
    main()
