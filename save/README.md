# Recreating the benchmark

Unimodal:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 0 --modality eeg --model_name UnimodalDeformer.UnimodalDeformer --task SwitchingTask3Presence --cuda 1 --device 0
(Parameters: 1417218)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 0 --modality ppg --model_name UnimodalDeformer.UnimodalDeformer --task SwitchingTask3Presence --cuda 1 --device 1
(Parameters: 771074)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 0 --modality eda --model_name UnimodalDeformer.UnimodalDeformer --task SwitchingTask3Presence --cuda 1 --device 2
(Parameters: 257922)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 0 --modality resp --model_name UnimodalDeformer.UnimodalDeformer --task SwitchingTask3Presence --cuda 1 --device 3
(Parameters: 322066)

Added up unimodal parameters: 2768280

-----------

multi channel deformer:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 1  --model_name MultiChannelDeformer.MultiChannelDeformer --task SwitchingTask3Presence --cuda 1 --device 4
(Parameters: 4952962)

-----------

Scaling procedure: 
+- 0.1% (4952) Parameters compared to multi channel deformer (4952962 * 0.001 = 4952.962)
--> lower limit: 4948010, upper limit: 4957914
Linear scaling factor applied to mlp_dim, dim_head, heads, num_kernel, emb_dim (all hyperparameters that are not fixed, e.g. number of channels, or would drastically change the architecture, e.g. depth or temporal kernel)

-----------

Unimodal (scaled): 

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 0 --modality eeg --model_name UnimodalDeformer.UnimodalDeformer --task SwitchingTask3Presence --cuda 1 --device 0 --mlp_dim 31 --dim_head 30 --heads 30 --num_kernel 121 --emb_dim 486
(Parameters: 4948041, scaling factor: 1.90)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 0 --modality ppg --model_name UnimodalDeformer.UnimodalDeformer --task SwitchingTask3Presence --cuda 1 --device 1 --mlp_dim 42 --dim_head 41 --heads 41 --num_kernel 11 --emb_dim 47
(Parameters: 4948027, scaling factor: 2.65)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 0 --modality eda --model_name UnimodalDeformer.UnimodalDeformer --task SwitchingTask3Presence --cuda 1 --device 2 --mlp_dim 71 --dim_head 71 --heads 71 --num_kernel 24 --emb_dim 74
(Parameters: 4948156, scaling factor: 4.45)

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 0 --modality resp --model_name UnimodalDeformer.UnimodalDeformer --task SwitchingTask3Presence --cuda 1 --device 3 --mlp_dim 63 --dim_head 63 --heads 64 --num_kernel 23 --emb_dim 64
(Parameters: 4949790, scaling factor: 4.00)

-----------

Early fusion deformer (scaled):

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 1  --model_name EarlyFusionDeformer.EarlyFusionDeformer --task SwitchingTask3Presence --cuda 1 --device 0 --mlp_dim 18 --dim_head 17 --heads 17 --num_kernel 70 --emb_dim 284
(Parameters: 4952876, scaling factor: 1.10)

-----------

Intermediate fusion deformer (scaled):

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m run_benchmark --multimodal 1  --model_name IntermediateFusionDeformer.IntermediateFusionDeformer --task SwitchingTask3Presence --cuda 1 --device 0 --mlp_dim 21 21 21 21 --dim_head 21 --heads 21 --num_kernel 86 5 5 5 --emb_dim 344
(Parameters: 4948560, scaling factor: 1.34)

-----------

Late fusion deformer: 

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evaluate.evaluate_late_fusion --split validation --cuda 1 --device 0 --modality_save_dirs [LIST_OF_UNIMODAL_DIRECTORIES]

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evaluate.evaluate_late_fusion --split test --cuda 1 --device 0 --modality_save_dirs [LIST_OF_UNIMODAL_DIRECTORIES]

-----------

The above commands create the results or the task 'SwitchingTask3Presence'

Repeat everything for the following tasks: 
(as these four tasks have the same number of classes we can use the same hyperparameters)
- SwitchBackAuditive3PresenceRelax
- SwitchBackAuditive3Presence
- VisualSearchTask3Presence




