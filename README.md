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
- Placing that model inside the [models](models) folder
- Running the model on a benchmark task by executing [run_benchmark.py](train/train_cross_validation.py) 

All other aspects, including parameter loading, optimizer setup, data loading, as well as model training and evaluation, are done automatically, reproducibly and in compliance with the already existing benchmark results.

### Creating a custom model class
TODO: explain the overall procedeuce and link to unimodel and multimodal examples. 

The names of the models can be chosen freely. Once a custom model has been added to the [models](models) folder, the benchmarking system will automatically import it and add its name to the list of available models. 

### Running the model on the benchmark
The [run_benchmark.py](train/train_cross_validation.py) script allows users to run any model, whether unimodal or multimodal, on any task, and can be configured through its command line line argments. The most important arguments are:
- `model_name`: The name of the model. This is where a custom model becomes available as option once added to the [models](models) folder
- `multimodal`:
- `modality`:
- `cuda`:
- `device`:
- All arguments that have been added by the user when overwriting the `add_model_options` function of the custom model class

More details and a list of all available options for the specific model are provided when adding --help or -h to the end of the script execution. After executing the [run_benchmark.py](train/train_cross_validation.py) script, the results will be stored in the [save](save) directory. This includes the model configuration, all model checkpoints as well as the validation and test scores of a 10-fold-cross-validation. 

#### Examples
If for intance, a user adds a custom multimodal model called `MyMultimodalModel`, it can be run on the `SwitchBackAuditivePresence` task by executing the following command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.run_benchmark --multimodal 1 --model_name MyMultimodalModel --task SwitchBackAuditivePresence --cuda 1 --device 0
```
If, on the other hand, a user adds a custom unimodal model called `MyUnimodalModel`, it can be run with any single modality (here EEG) on the `SwitchBackAuditivePresence` task by executing the following command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.run_benchmark --multimodal 0 --modality eeg --model_name MyUnimodalModel --task SwitchBackAuditivePresence --cuda 1 --device 0
```

## Reproducability
The benchmarking system is configured to assure full reproducablity, by replacing all non-deterministic algorithms, such as dropout during training, with deterministic alternatives. An exact guide for reprocuding the benchmark results from our [paper]() can be found [here](save/README.md). 

## License
This code is distributed under the [CC-By Attribution 4.0 International Public License](LICENSE).

