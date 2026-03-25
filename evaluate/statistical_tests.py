import numpy as np
import os
import json
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, shapiro
from statsmodels.stats.multitest import multipletests


def paired_statistical_test_subset(models_dict, selected_model_names, normalize_within_tasks=False):
    """
    Performs a paired t-test on a subset of models.
    Optionally, each task is normalized across all models. Then folds are flattened.
    Pairwise t-tests are computed between the specified best model and the rest
    of the subset, with Holm correction for multiple comparisons.

    Parameters:
    - models_dict: dict of {model_name: np.array of shape (num_tasks, num_folds)}
    - best_model_name: str, key of the model in models_dict to compare against others
    """

    model_names = list(models_dict.keys())
    num_tasks, num_folds = next(iter(models_dict.values())).shape
    #print(num_tasks, num_folds)

    # Step 1: Task-wise normalization across models
    models = {}
    for t in range(num_tasks):
        # Stack all models' folds for this task
        if normalize_within_tasks:
            task_data = np.array([models_dict[m][t] for m in model_names])  # shape: num_models x num_folds
            task_mean = task_data.mean()
            task_std = task_data.std(ddof=1)

            for i, m in enumerate(model_names):
                if m not in models:
                    models[m] = np.zeros_like(models_dict[m])
                models[m][t] = (models_dict[m][t] - task_mean) / task_std
        else:
            models = models_dict

    # Step 2: Flatten folds per model
    models_flat = {m: data.flatten() for m, data in models.items()}

    print("-------")

    # Optional: Friedman test on flattened normalized data (for reference)
    stat, p_friedman = friedmanchisquare(*models_flat.values())
    #print(f"Friedman statistic: {stat:.3f}, p-value: {p_friedman}\n")
    print(f"Friedman p-value: {p_friedman}\n")

    p_values = []
    differences = []
    competitors = [m for m in model_names if m not in selected_model_names]
    for m_name in selected_model_names:
        # Step 3: Paired t-tests vs selected model
        model_scores = models_flat[m_name]
        
        for comp in competitors:
            #_, p_val = ttest_rel(model_scores, models_flat[comp])
            _, p_val = wilcoxon(model_scores, models_flat[comp])

            diff = np.array(model_scores) - np.array(models_flat[comp])
            differences.extend(diff)
            p_values.append(p_val)

    # This informs us whether to do Wilcoxon or t-test. 
    stat, p_shapiro = shapiro(differences)
    #print(f"Shapiro statistics: {stat:.3f}, p-value: {p_shapiro}, is normal: {'Yes' if p_shapiro > 0.05 else 'No'}\n")
    print(f"Shapiro p-value: {p_shapiro}, is normal: {'Yes' if p_shapiro > 0.05 else 'No'}\n")

    # Step 4: Holm correction
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')

    # Step 5: Print results
    for l, m_name in enumerate(selected_model_names):
        print('\n')
        #print(f"Pairwise t-tests (Holm-corrected) vs {m_name}:\n")
        print(f"Pairwise Wilcoxon signed-rank (Holm-corrected) vs {m_name}:\n")
        for i, comp in enumerate(competitors):
            print(f"{m_name} vs {comp}: "
                #f"Wilcoxon signed-rank p={p_values[(l * len(competitors)) + i]}, "
                f"Holm-corrected p={p_corrected[(l * len(competitors)) + i]}, "
                f"Significant? {'Yes' if reject[(l * len(competitors)) + i] else 'No'}")


def load_model_results(model_base_dir, scaled=None):
    scaled_threshold=4952962
    tolerance=4952

    required_tasks = [
        'SwitchingTask3Presence',
        'SwitchBackAuditive3PresenceRelax', 
        'SwitchBackAuditive3Presence',
        'VisualSearchTask3Presence'
    ]

    if not os.path.isdir(model_base_dir):
        raise ValueError(f"{model_base_dir} is not a valid directory.")

    # List only directories
    task_dirs = [
        d for d in os.listdir(model_base_dir)
        if os.path.isdir(os.path.join(model_base_dir, d))
    ]

    num_tasks = len(task_dirs)
    selected = {}

    if num_tasks == 8 and scaled is None:
        raise ValueError(
            "8 task folders found. You must specify scaled=True or scaled=False."
        )

    for d in task_dirs:
        args_path = os.path.join(model_base_dir, d, "args.json")
        test_results_path = os.path.join(model_base_dir, d, "test.json")

        if not os.path.isfile(args_path):
            raise ValueError(f"Missing args.json in {test_results_path}")

        with open(args_path, "r") as f:
            args = json.load(f)

        if "LateFusionDeformer" in model_base_dir:
            if "nr_of_parameters" not in args["late_fusion"]:
                raise ValueError(f"'nr_of_parameters' missing in {args_path}")
            nr_params = args["late_fusion"]["nr_of_parameters"]
            task = args["eeg"]["task"]
        else: 
            if "nr_of_parameters" not in args:
                raise ValueError(f"'nr_of_parameters' missing in {args_path}")
            nr_params = args["nr_of_parameters"]
            task = args["task"]

        if num_tasks == 4:
            selected[task] = test_results_path
        elif num_tasks == 8:
            # Determine scaled vs unscaled
            is_scaled = abs(nr_params - scaled_threshold) <= tolerance
            if scaled and is_scaled:
                selected[task] = test_results_path
            elif not scaled and not is_scaled:
                selected[task] = test_results_path
        else:
            raise ValueError(
                f"Expected 4 or 8 task folders, but found {num_tasks}."
            )

    if len(selected) != 4:
        raise ValueError(
            f"Expected 4 {'scaled' if scaled else 'unscaled'} tasks "
            f"but found {len(selected)}. Check parameter thresholds."
        )
    
    for req_t in required_tasks:
        if req_t not in selected.keys():
            raise ValueError(f"Missing task: {req_t}")
    
    print("----")
    print(model_base_dir)
    print("-")
    test_results = []
    test_standard_deviations = []
    for req_t in required_tasks:
        results_path = selected[req_t]
        with open(results_path, "r") as f:
            results = json.load(f)
        scores = []
        for i in range(1, 11):
            if "LateFusionDeformer" in model_base_dir:
                fold_score = results[f"{i}"]["late fusion"]["micro f1-score"]
            else:
                fold_score = results[f"{i}"]["micro f1-score"]
            scores.append(fold_score)
        test_results.append(scores)
        standard_deviation = np.std(scores)
        test_standard_deviations.append(standard_deviation)
        print(f"{req_t}: {(np.mean(scores)*100):.2f} +- {(standard_deviation*100):.2f}")
    print(f"Average: {(np.mean(np.concatenate(test_results)) * 100):.2f} +- {(np.mean(test_standard_deviations) * 100):.2f}")
    print("----")

    test_results = np.array(test_results)

    if len(np.concatenate(test_results).tolist()) != 40:
        raise ValueError("Not exactly 40 scores")
    
    return test_results

cwd = os.getcwd()
base_save_dir = os.path.join(cwd, "../../../../data/antmen/save")


eeg_deformer_base_dir = os.path.join(base_save_dir, "unimodal", "UnimodalDeformer", "eeg")
ppg_deformer_base_dir = os.path.join(base_save_dir, "unimodal", "UnimodalDeformer", "ppg")
eda_deformer_base_dir = os.path.join(base_save_dir, "unimodal", "UnimodalDeformer", "eda")
resp_deformer_base_dir = os.path.join(base_save_dir, "unimodal", "UnimodalDeformer", "resp")

multi_channel_encoder_v1_base_dir = os.path.join(base_save_dir, "multimodal", "MultiChannelEncoderV1.MultiChannelEncoderV1")
multi_channel_encoder_v2_base_dir = os.path.join(base_save_dir, "multimodal", "MultiChannelEncoderV2.MultiChannelEncoderV2")
phemonet_base_dir = os.path.join(base_save_dir, "multimodal", "PHemoNet.HyperFuseNet")

late_fusion_deformer_base_dir = os.path.join(base_save_dir, "multimodal", "LateFusionDeformer")
intermediate_fusion_deformer_base_dir = os.path.join(base_save_dir, "multimodal", "IntermediateFusionDeformer")
early_fusion_deformer_base_dir = os.path.join(base_save_dir, "multimodal", "EarlyFusionDeformer")
multi_channel_deformer_base_dir = os.path.join(base_save_dir, "multimodal", "MultiChannelDeformer")

multi_channel_deformer_w_o_dip_and_ftl_base_dir = os.path.join(base_save_dir, "multimodal", "ablation.MultiChannelDeformerWithoutDIPandFTL.MultiChannelDeformerWithoutDIPandFTL")
multi_channel_deformer_w_o_dip_base_dir = os.path.join(base_save_dir, "multimodal", "ablation.MultiChannelDeformerWithoutDIP.MultiChannelDeformerWithoutDIP")
multi_channel_deformer_w_o_ftl_base_dir = os.path.join(base_save_dir, "multimodal", "ablation.MultiChannelDeformerWithoutFTL.MultiChannelDeformerWithoutFTL")
multi_channel_deformer_w_o_sfe_base_dir = os.path.join(base_save_dir, "multimodal", "ablation.MultiChannelDeformerWithoutSFE.MultiChannelDeformerWithoutSFE")


eeg_deformer_results = load_model_results(eeg_deformer_base_dir, scaled=False)
eeg_deformer_scaled_results = load_model_results(eeg_deformer_base_dir, scaled=True)
ppg_deformer_results = load_model_results(ppg_deformer_base_dir, scaled=False)
ppg_deformer_scaled_results = load_model_results(ppg_deformer_base_dir, scaled=True)
eda_deformer_results = load_model_results(eda_deformer_base_dir, scaled=False)
eda_deformer_scaled_results = load_model_results(eda_deformer_base_dir, scaled=True)
resp_deformer_results = load_model_results(resp_deformer_base_dir, scaled=False)
resp_deformer_scaled_results = load_model_results(resp_deformer_base_dir, scaled=True)

multi_channel_encoder_v1_results = load_model_results(multi_channel_encoder_v1_base_dir, scaled=False)
multi_channel_encoder_v1_scaled_results = load_model_results(multi_channel_encoder_v1_base_dir, scaled=True)
multi_channel_encoder_v2_results = load_model_results(multi_channel_encoder_v2_base_dir, scaled=False)
multi_channel_encoder_v2_scaled_results = load_model_results(multi_channel_encoder_v2_base_dir, scaled=True)
phemonet_results = load_model_results(phemonet_base_dir, scaled=False)
phemonet_scaled_results = load_model_results(phemonet_base_dir, scaled=True)

late_fusion_deformer_results = load_model_results(late_fusion_deformer_base_dir)
intermediate_fusion_deformer_results = load_model_results(intermediate_fusion_deformer_base_dir)
early_fusion_deformer_results = load_model_results(early_fusion_deformer_base_dir)
multi_channel_deformer_results = load_model_results(multi_channel_deformer_base_dir)

multi_channel_deformer_results = load_model_results(multi_channel_deformer_base_dir)
multi_channel_deformer_results = load_model_results(multi_channel_deformer_base_dir)
multi_channel_deformer_results = load_model_results(multi_channel_deformer_base_dir)
multi_channel_deformer_results = load_model_results(multi_channel_deformer_base_dir)

multi_channel_deformer_w_o_dip_and_ftl_results = load_model_results(multi_channel_deformer_w_o_dip_and_ftl_base_dir)
multi_channel_deformer_w_o_dip_results = load_model_results(multi_channel_deformer_w_o_dip_base_dir)
multi_channel_deformer_w_o_ftl_results = load_model_results(multi_channel_deformer_w_o_ftl_base_dir)
multi_channel_deformer_w_o_sfe_results = load_model_results(multi_channel_deformer_w_o_sfe_base_dir)



all_unimodal_vs_late_fusion_deformer = {
    "EEG-Deformer-Scaled": eeg_deformer_scaled_results,
    "PPG-Deformer-Scaled": ppg_deformer_scaled_results,
    "EDA-Deformer-Scaled": eda_deformer_scaled_results,
    "RESP-Deformer-Scaled": resp_deformer_scaled_results,
    "Late-Fusion-Deformer": late_fusion_deformer_results,
}

paired_statistical_test_subset(
    all_unimodal_vs_late_fusion_deformer,
    selected_model_names=[
        "Late-Fusion-Deformer", 
    ]
)

all_unimodal_vs_intermediate_fusion_deformer = {
    "EEG-Deformer-Scaled": eeg_deformer_scaled_results,
    "PPG-Deformer-Scaled": ppg_deformer_scaled_results,
    "EDA-Deformer-Scaled": eda_deformer_scaled_results,
    "RESP-Deformer-Scaled": resp_deformer_scaled_results,
    "Intermediate-Fusion-Deformer": intermediate_fusion_deformer_results
}

paired_statistical_test_subset(
    all_unimodal_vs_intermediate_fusion_deformer,
    selected_model_names=[
        "Intermediate-Fusion-Deformer"
    ]
)

all_unimodal_vs_early_fusion_deformer = {
    "EEG-Deformer-Scaled": eeg_deformer_scaled_results,
    "PPG-Deformer-Scaled": ppg_deformer_scaled_results,
    "EDA-Deformer-Scaled": eda_deformer_scaled_results,
    "RESP-Deformer-Scaled": resp_deformer_scaled_results,
    "Early-Fusion-Deformer": early_fusion_deformer_results
}

paired_statistical_test_subset(
    all_unimodal_vs_early_fusion_deformer,
    selected_model_names=[
        "Early-Fusion-Deformer"
    ]
)

all_unimodal_vs_multi_channel_deformer = {
    "EEG-Deformer-Scaled": eeg_deformer_scaled_results,
    "PPG-Deformer-Scaled": ppg_deformer_scaled_results,
    "EDA-Deformer-Scaled": eda_deformer_scaled_results,
    "RESP-Deformer-Scaled": resp_deformer_scaled_results,
    "Multi-Channel-Deformer": multi_channel_deformer_results
}

paired_statistical_test_subset(
    all_unimodal_vs_multi_channel_deformer,
    selected_model_names=[
        "Multi-Channel-Deformer"
    ]
)

all_prior_multimodal_vs_multi_channel_deformer = {
    "Multi-Channel-Encoder-V1": multi_channel_encoder_v1_results,
    "Multi-Channel-Encoder-V1-Scaled": multi_channel_encoder_v1_scaled_results,
    "Multi-Channel-Encoder-V2": multi_channel_encoder_v2_results,
    "Multi-Channel-Encoder-V2-Scaled": multi_channel_encoder_v2_scaled_results,
    "PhemoNet": phemonet_results,
    "PhemoNet-Scaled": phemonet_scaled_results,
    "Late-Fusion-Deformer": late_fusion_deformer_results,
    "Intermediate-Fusion-Deformer": intermediate_fusion_deformer_results,
    "Early-Fusion-Deformer": early_fusion_deformer_results,
    "Multi-Channel-Deformer": multi_channel_deformer_results
}

paired_statistical_test_subset(
    all_prior_multimodal_vs_multi_channel_deformer,
    selected_model_names=[
        "Late-Fusion-Deformer", 
        "Intermediate-Fusion-Deformer", 
        "Early-Fusion-Deformer", 
        "Multi-Channel-Deformer"
    ]
)


all_multimodal_deformers_vs_multi_channel_deformer = {
    "Late-Fusion-Deformer": late_fusion_deformer_results,
    "Intermediate-Fusion-Deformer": intermediate_fusion_deformer_results,
    "Early-Fusion-Deformer": early_fusion_deformer_results,
    "Multi-Channel-Deformer": multi_channel_deformer_results
}

paired_statistical_test_subset(
    all_multimodal_deformers_vs_multi_channel_deformer,
    selected_model_names=["Multi-Channel-Deformer"]
)


ablations_vs_multi_channel_deformer = {
    "Multi-Channel-Deformer w/o DIP and FTL": multi_channel_deformer_w_o_dip_and_ftl_results,
    "Multi-Channel-Deformer w/o DIP": multi_channel_deformer_w_o_dip_results,
    "Multi-Channel-Deformer w/o FTL": multi_channel_deformer_w_o_ftl_results,
    "Multi-Channel-Deformer w/o SFE": multi_channel_deformer_w_o_sfe_results,
    "Multi-Channel-Deformer w/o MCE layers": intermediate_fusion_deformer_results,
    "Multi-Channel-Deformer": multi_channel_deformer_results
}

paired_statistical_test_subset(
    ablations_vs_multi_channel_deformer,
    selected_model_names=["Multi-Channel-Deformer"]
)

