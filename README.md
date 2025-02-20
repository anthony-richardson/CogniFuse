# CogniFuse

short introduction to the benchmarkl 

reprodcuboiliy .. etc. multimodal; biosioginal fusion etc. accerlate reserch etc. 

also show table of current test results. 

## Setup
requirements txt 

## Adding custom models
To add a custom model to the benchmark, three steps are required:
- Creating a pytorch model class that inherits from [BaseBenchmarkModel](utils/model_util.py)
- Adding that model in a separate python script to the [models](models) folder
- Running the model on a benchmark task by executing [run_benchmark.py](train/train_cross_validation.py) 

All other aspects, including parameter loading, optimizer setup, data loading, as well as model training and evaluation, are done automatically, reproducibly and in compliance with the already existing benchmark results.

### Creating a custom model class
TODO: explain the overall procedeuce and link to unimodel and multimodal examples. 

### Running the model on the benchmark
TODO: exaplain the exectuin of the run_benchmark.py script and its parameters. 
once added to the .. folder, the model automatically becomes availalabele as choice for the ..argument. If for intance, a user adds a custom model called MyModel, it can be run on the SwitchingTask3Presence task by executing the following command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.run_benchmark --multimodal 1 --model_name MyModel --task SwitchingTask3Presence --cuda 1 --device 0
```

## Reproducability
The benchmarking system is configured to assure full reproducablity, by replacing all non-deterministic algorithms, such as dropout during training, with deterministic alternatives. An exact guide for reprocuding the benchmark results from our [paper]() can be found [here](save/README.md). 

## License
This code is distributed under the [CC-By Attribution 4.0 International Public License](LICENSE).

