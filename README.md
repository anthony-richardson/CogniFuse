# CogniFuse
reprodcuboiliy .. etc. multimodal; biosioginal fusion etc. accerlate reserch etc. 

## Adding custom models
To add a custom model to the benchmark three steps need to be executed:
- Creating a custom pytorch model class that inherits from [BaseBenchmarkModel](utils/model_util.py)
- Adding that model in a separate python script to the [models](models) folder
- Running the model on the benchmark by executing [run_model.py](train/train_cross_validation.py) 

All other aspects .. 

### Creating a custom model class
hi

### Running the model on the benchmark

## Reproducablity
The benchmarking system is configured to assure full reproducablity, by replacing all non-deterministic algorithms, such as dropout during training, with deterministic alternatives. A guide for reprocuding the benchmark results from our [paper]() can be found [here](save/README.md). 

## License
This code is distributed under the [CC-By Attribution 4.0 International Public License](LICENSE).

