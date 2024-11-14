# Recreating the benchmark

Unimodal:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality eeg  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 0
(Parameters: 1417218)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality ppg  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 1
(Parameters: 193778)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality eda  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 2
(Parameters: 129634)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 0 --modality resp  --model_name UnimodalDeformer --task SwitchingTaskPresence --cuda 1 --device 3
(Parameters: 322066)

Added up unimdal paramaters: 2062696 

-----------

Scaling procedure: 
+- 0.1% (3893) Parameters compared to cross modal deformer (3893338 * 0.001 = 3893.338)
--> lower limit: 3889445, upper limit: 3897231
Scaling factor applied to mlp_dim, dim_head, heads, num_kernel (all hyper parameters that are not fixed, e.g. number of channels, or would drastically change the architecture, e.g. depth)

-----------

Unimodal (scaled up): 

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m unimodal.train.cross_validate_unimodal_deformer --task ControlledSwitchingLowHigh --device 3 --modality eeg --mlp_dim 34 --dim_head 41 --heads 41 --num_kernel 163
(Parameters: 4326173, scaling factor: 2.55)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m unimodal.train.cross_validate_unimodal_deformer --task ControlledSwitchingLowHigh --device 4 --modality ppg --mlp_dim 83 --dim_head 55 --heads 55 --num_kernel 222
(Parameters: 4317964, scaling factor: 3.465)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m unimodal.train.cross_validate_unimodal_deformer --task ControlledSwitchingLowHigh --device 5 --modality eda --mlp_dim 60 --dim_head 58 --heads 59 --num_kernel 237
(Parameters: 4317896, scaling factor: 3.7)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m unimodal.train.cross_validate_unimodal_deformer --task ControlledSwitchingLowHigh --device 6 --modality resp --mlp_dim 53 --dim_head 50 --heads 49 --num_kernel 198
(Parameters: 4317892, scaling factor: 3.09)

-----------

Efficient Crossmodal deformer:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_cross_validation --multimodal 1  --model_name EfficientMultiChannelDeformer --task SwitchingTaskPresence --cuda 1 --device 4
(Parameters: 3893338)

-----------

Crossmodal deformer:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m multimodal.train.cross_validate_cross_modal_deformer --task ControlledSwitchingLowHigh --device 2 --fusion_type crossmodal
(Parameters: 4322194)

-----------

Multimodal early fusion deformer (scaled up):

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m multimodal.train.cross_validate_cross_modal_deformer --task ControlledSwitchingLowHigh --device 1 --fusion_type early --mlp_dim 37 --dim_head 29 --heads 30 --num_kernel 122
(Parameters: 4318074, scaling factor: 1.9)

-----------

Multimodal intermediate fusion deformer (scaled up):

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m multimodal.train.cross_validate_cross_modal_deformer --task ControlledSwitchingLowHigh --device 5 --fusion_type intermediate --mlp_dim 23 --dim_head 24 --heads 24 --num_kernel 89
(Parameters: 4325977, scaling factor: 1.45)

-----------

Multimodal late fusion deformer: 



-----------

Late fusion command:

python -m multimodal.train.cross_validate_late_fusion --modality_save_dirs unimodal/save/eeg/2024.10.15-07:30:11 unimodal/save/ppg/2024.10.15-07:32:17

ControlledSwitchingLowHigh




