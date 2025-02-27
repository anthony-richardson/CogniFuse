# CogniFuse

This is the official code base of the paper [CogniFuse and Multimodal Deformers: A Unified Approach for Benchmarking and Modeling Biosignal Fusion](). Please cite it when using this code or benchmark in orignal or modified form. 
```
TODO
```

Continuously monitored physiological signals, often referred to as biosignals, carry rich information about the human body and the biological processes happening within. Extracting this information from casually collected data in activities of daily living holds great potential to revolutionize real-time monitoring of physical and mental states outside of highly controlled clinical conditions. However, this potential comes with a number of difficulties: 
- Data recorded during activities of daily living contains an increased amount of noise and artifacts, making the extraction of relevant aspects more difficult
- The importance of each biosignal and the impact it should have on the prediction can vary between tasks
- The relevant aspects in the data can differ between tasks
- Difficulty scales with the number of tasks and modalities

Therefore, we provide a public dataset and benchmarking system for multi-task multimodal biosignal fusion during activities of daily living. To accelerate future research on biosinal fusion, this benchmarking system was developed with careful attention to comparability, robustness, reproducibility and accessibility. In particular, the process of adding custom models in a comparible and reproducible way is highly simplified and does not require any modifications of the underlying code base.

## Getting started 

The dataset needed to run the benchmark can be accessed [here](). A detailed description of the data is provided in our [paper](). The dataset contains 119.435 samples from 134 participants. Each sample is a collection of simultaneously starting chunks of electroencephalogram (`eeg`), photoplethysmography (`ppg`), electrodermal activity (`eda`) and respiration (`resp`) data.  

We recommend using a conda environment for running the benchmark. The official conda installation guide can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). After the installation, users can set up and activate a new conda environment by running:

```
conda create -n cognifuse python=3.12.4
conda activate cognifuse
```

The specified python version is required, even when not using a conda environment. To install the remaining dependencies for this code base, run:

```
pip install -r requirements.txt
```

## Adding custom models
To produce benchmark scores for a custom model, three steps are required:
- Creating a pytorch model class that inherits from [BaseBenchmarkModel](models/BaseBenchmarkModel.py)
- Placing the created model class anywhere inside the [models](models) folder
- Running the model on a benchmark task by executing [run_benchmark.py](run_benchmark.py) 

All other aspects, including parameter loading, optimizer setup, data loading, as well as model training and evaluation, are done automatically, reproducibly and in compliance with the already existing benchmark results.

### Creating a custom model class
To become usable for the benchmarking system, each created model must inherit from [BaseBenchmarkModel](models/BaseBenchmarkModel.py). This forces the created model class to overwrite the `add_model_options` function, where it is possible to define custom command line arguments that will be automatically passed to the models `init` function. These arguments may then be used to configure the models architecture. The name of the model can be chosen freely. Once a custom model has been added to the [models](models) folder, the benchmarking system will automatically integrate it and add its name to the list of available models. This also works when using subdirectories or when placing multiple models in one script. 

In addition to the models introduced in our [paper](), we provide two minimal examples of creating custom model classes. One of these example is for the [unimodal](models/dummies/UnimodalDummy.py) and the other one for the [multimodal](models/dummies/MultimodalDummy.py) case. They provide detailed documentation and serve as starting points when creating custom models. 

### Running a model on the benchmark
The [run_benchmark.py](run_benchmark.py) script allows users to run any model, whether unimodal or multimodal, on any task, and can be configured through its command line line argments. The most important arguments are:
- `model_name`: The class name of the model. This is where a custom model becomes available as option once added to the [models](models) folder
- `multimodal`: The choice between `1` for multimodal and `0` for unimodal
- `modality`: In the case of an unimodal model, this decides which modality to use. The options are `eeg`, `ppg`, `eda` and `resp`
- `task`: The task on which the model should be trained and evaluated. The available tasks can be found [here](utils/tasks.py) or in the script help menu 
- `cuda`: The choice between `1` for using a cuda device and `0` for using the cpu
- `device`: The device id, in the case of using a cuda device. For Nvidia graphics cards, this id may be viewed using `nvidia-smi`
- All arguments that have been added by the user when overwriting the `add_model_options` function of the custom model class

More details and a list of all available options are provided when adding `--help` or `-h` to the end of the script execution. After executing the [run_benchmark.py](run_benchmark.py) script, the results will be stored in the [save](save) directory. This includes the model configuration, all model checkpoints as well as the validation and test scores of a 10-fold-cross-validation.

#### Examples
If for intance, a user adds a custom class for a multimodal model called `MyMultimodalModel` inside a script called `MyScript.py`, the model can be run on the `SwitchBackAuditivePresence` task by executing:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.run_benchmark --model_name MyScript.MyMultimodalModel --multimodal 1 --task SwitchBackAuditivePresence --cuda 1 --device 0
```

If, on the other hand, a user adds a custom class for a unimodal model for EEG data called `MyUnimodalModel` inside a script called `MyScript.py`, the model can be run on the `SwitchBackAuditivePresence` task by executing:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.run_benchmark --model_name MyScript.MyUnimodalModel --multimodal 0 --modality eeg --task SwitchBackAuditivePresence --cuda 1 --device 0
```

## Reproducibility
The benchmarking system is configured to assure full reproducability, by replacing all non-deterministic algorithms, such as dropout during training, with deterministic alternatives. An exact guide for reprocuding the benchmark results from our [paper]() can be found [here](save/README.md). 

## License
This code is distributed under the [CC-By Attribution 4.0 International Public License](LICENSE).

