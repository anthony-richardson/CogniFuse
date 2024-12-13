# Recreating the benchmark

For the next run of the benchmark change the following:
- redo the dataset creation with new/shuffled subject ids
- also scale emb_dim and redo the scaling
- prior to benchmark test if increasing the mlp_dim for EfficientMultiChannelDeformer improves results (and does not add a lot of parameters)
- Observation: learning rate might still be too high. Test lower lr out on examples
- find an optimization scheme or schedule that works well for all the models


Unimodal:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality eeg  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 0
(Parameters: 1417218)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality ppg  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 1
(Parameters: 193778)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality eda  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 2
(Parameters: 129634)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality resp  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 3
(Parameters: 322066)

Added up unimodal parameters: 2062696 

-----------

Efficient Crossmodal deformer:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 1  --model_name EfficientMultiChannelDeformer --task SwitchingTaskPresence --cuda 1 --device 4
(Parameters: 3893338)

-----------

Scaling procedure: 
+- 0.1% (3893) Parameters compared to efficient cross modal deformer (3893338 * 0.001 = 3893.338)
--> lower limit: 3889445, upper limit: 3897231
Scaling factor applied to mlp_dim, dim_head, heads, num_kernel (all hyperparameters that are not fixed, e.g. number of channels, or would drastically change the architecture, e.g. depth)
(TODO: add emb_dim to the scaled hyperparameters)

-----------

Unimodal (scaled up): 

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality eeg  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 0 --mlp_dim 30 --dim_head 30 --heads 30 --num_kernel 119
(Parameters: 3892209, scaling factor: 1.86)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality ppg --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 1 --mlp_dim 73 --dim_head 73 --heads 73 --num_kernel 18
(Parameters: 3890214, scaling factor: 4.55)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality eda --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 2 --mlp_dim 90 --dim_head 90 --heads 88 --num_kernel 33
(Parameters: 3892994, scaling factor: 5.65)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality resp --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 3 --mlp_dim 57 --dim_head 57 --heads 56 --num_kernel 15
(Parameters: 3889478, scaling factor: 3.55)

-----------

Crossmodal deformer (scaled up):

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 1  --model_name MultiChannelDeformer --task SwitchingTaskPresence --cuda 1 --device 0 --mlp_dim 10 --dim_head 12 --heads 11 --num_kernel 45
(Parameters: 3889519, scaling factor: 0.71)

-----------

Early fusion deformer (scaled up):

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 1  --model_name EarlyFusionDeformer --task SwitchingTaskPresence --cuda 1 --device 1 --mlp_dim 26 --dim_head 20 --heads 20 --num_kernel 80
(Parameters: 3891514, scaling factor: 1.25)

-----------

Intermediate fusion deformer (scaled up):

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 1  --model_name IntermediateFusionDeformer --task SwitchingTaskPresence --cuda 1 --device 2 --mlp_dim 24 24 24 24 --dim_head 23 --heads 23 --num_kernel 91 6 6 6
(Parameters: 3890597, scaling factor: 1.42)

-----------

Late fusion deformer: 

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evaluate.evaluate_late_fusion --split validation --cuda 1 --device 0 --modality_save_dirs [LIST_OF_UNIMODAL_DIRECTORIES]

-----------

The above commands create the results or the task 'SwitchingTaskPresence'

Repeat everything for the following tasks: 
(as these four tasks have the same number of classes we can use the same hyperparameters)
- 'SwitchBackAuditivePresenceRelax'
- 'SwitchBackAuditivePresence'
- 'VisualSearchTaskPresence'




