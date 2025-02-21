# CogniFuse

short introduction to the benchmarkl 

reprodcuboiliy .. etc. multimodal; biosioginal fusion etc. accerlate reserch etc. 

also show table of current test results. 

## Setup
requirements txt 

data descriptin and download and number of samples . 

## Adding custom models
To add a custom model to the benchmark, three steps are required:
- Creating a pytorch model class that inherits from [BaseBenchmarkModel](utils/model_util.py)
- Placing the created model inside the [models](models) folder
- Running the model on a benchmark task by executing [run_benchmark.py](train/train_cross_validation.py) 

All other aspects, including parameter loading, optimizer setup, data loading, as well as model training and evaluation, are done automatically, reproducibly and in compliance with the already existing benchmark results.

### Creating a custom model class
To become usable for the benchmarking system, each created model must inherit from [BaseBenchmarkModel](utils/model_util.py). This forces the created model class to overwrite the `add_model_options` function, where it is possible to define of custom command line arguments that will be automatically passed to the models `init` function. These arguments may then be used to configure the models architecture. One argument that is by default already passed to the models `init` function is the number of output classes. This number depends on the selected task and must be used to define the output dimension of the model. The name of the model can be chosen freely. Once a custom model has been added to the [models](models) folder, the benchmarking system will automatically integrate it and add its name to the list of available models. 

In addition to the models introduced in our [paper](), we provide two minimal examples of creating custom model classes. One of these example is for the [unimodal](models/UnimodalDummy.py) and the other one for the [multimodal](models/MultimodalDummy.py) case. They provide detailed documentation and may be used as starting points when creating custom models. 

### Running a model on the benchmark
The [run_benchmark.py](train/train_cross_validation.py) script allows users to run any model, whether unimodal or multimodal, on any task, and can be configured through its command line line argments. The most important arguments are:
- `model_name`: The name of the model. This is where a custom model becomes available as option once added to the [models](models) folder
- `multimodal`: The choice between `1` for multimodal and `0` for unimodal
- `modality`: In the case of an unimodal model, this decides which modality to use. The options are `eeg`, `ppg`, `eda` and `resp`
- `task`: The task on which the model should be trained and evaluated. The available tasks can be found [here](utils/tasks.py) or in the script help menu 
- `cuda`: The choice between `1` for using a cuda device and `0` for using the cpu
- `device`: The device id, in the case of using a cuda device. For Nvidia grpahics cards, this id may be viewed using `nvidia-smi`
- All arguments that have been added by the user when overwriting the `add_model_options` function of the custom model class

More details and a list of all available options for the specific model are provided when adding `--help` or `-h` to the end of the script execution. After executing the [run_benchmark.py](train/train_cross_validation.py) script, the results will be stored in the [save](save) directory. This includes the model configuration, all model checkpoints as well as the validation and test scores of a 10-fold-cross-validation.

#### Examples
If for intance, a user adds a custom multimodal model called `MyMultimodalModel`, it can be run on the `SwitchBackAuditivePresence` task by executing the following command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.run_benchmark --model_name MyMultimodalModel --multimodal 1 --task SwitchBackAuditivePresence --cuda 1 --device 0
```
If, on the other hand, a user adds a custom unimodal model for EEG data called `MyUnimodalModel`, it can be run on the `SwitchBackAuditivePresence` task by executing the following command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.run_benchmark --model_name MyUnimodalModel --multimodal 0 --modality eeg --task SwitchBackAuditivePresence --cuda 1 --device 0
```

## Reproducability
The benchmarking system is configured to assure full reproducability, by replacing all non-deterministic algorithms, such as dropout during training, with deterministic alternatives. An exact guide for reprocuding the benchmark results from our [paper]() can be found [here](save/README.md). 

## License
This code is distributed under the [CC-By Attribution 4.0 International Public License](LICENSE).

