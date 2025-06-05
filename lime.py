import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import lime
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import Counter
from lime.lime_tabular import LimeTabularExplainer

BASE_PROJECT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Malicious URL Detection")
DATA_DIR = BASE_PROJECT_DIR / "data"
RESULTS_LIME_DIR = BASE_PROJECT_DIR / "results"
SCRIPTS_DIR = BASE_PROJECT_DIR / "scripts"

PIPELINE_FILENAME = "preprocess.joblib"
MODEL_FILENAME = "model.pt"
X_TRAIN_FILENAME = "x_train_processed.npy"
Y_TRAIN_FILENAME = "y_train.npy"
X_TEST_FILENAME = "x_test_processed.npy"
Y_TEST_FILENAME = "y_test.npy"

SEED = 42                              # 재현성
N_BG_PER_CLASS = 100                   # 배경 데이터 샘플 개수
NUM_LIME_FEATURES_GLOBAL = 10          # 전역 설명 특성 개수
N_GLOBAL_PER_CLASS = 100               # 전역 설명 샘플 개수 by 클래스
NUM_LIME_SAMPLES_GLOBAL = 1000         # 전역 설명 샘플 개수
NUM_LIME_FEATURES_LOCAL = 10           # 지역 설명 특성 개수
NUM_LIME_SAMPLES_LOCAL = 1000          # 지역 설명 샘플 개수
DISTANCE_METRIC_LIME = 'euclidean'     # 인스턴스 간 거리 계산 방법
TARGET_CLASS_IDX_EXPLAIN = 1           # Malicious
TOP_N_FOR_GLOBAL_VISUALIZATION = 20    # 전역 설명 특성 룰 개수
CLASS_NAMES = ['Benign', 'Malicious']  # 클래스 라벨

class MLP(nn.Module):
    def __init__(self, in_dim: int, p_drop: float = 0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def setup_directories():
    RESULTS_LIME_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_LIME_DIR / "global_individual_explanations").mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {DATA_DIR.resolve()}")
    print(f"LIME results directory: {RESULTS_LIME_DIR.resolve()}")
    print(f"Scripts directory: {SCRIPTS_DIR.resolve()}")
    print(f"Global individual explanations directory: {(RESULTS_LIME_DIR / 'global_individual_explanations').resolve()}")

def set_random_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    print(f"\nRandom seed set to: {seed_value}")

def predict_proba_for_lime(numpy_array, model_instance, device_instance):
    if model_instance is None or device_instance is None:
        raise ValueError("Model or device not initialized for predictor.")

    tensor_data = torch.from_numpy(numpy_array).float().to(device_instance)       # NumPy 배열을 PyTorch 텐서로 변환
    model_instance.eval()

    with torch.no_grad():
        logits = model_instance(tensor_data)                                      # 순전파에 의한 logits
        prob_malicious = torch.sigmoid(logits)                                    # 악성 확률 계산
        prob_benign = 1 - prob_malicious                                          # 정상 확률 계산
        probabilities_for_lime = torch.cat((prob_benign, prob_malicious), dim=1)  # (Batch, 2)

    return probabilities_for_lime.cpu().numpy()

def load_preprocessing_pipeline(data_dir, filename):
    pipeline_path = data_dir / filename

    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)                                        # 전처리 파이프라인 로드
        print(f"Preprocessing pipeline loaded from: {pipeline_path}")

        try:
            feature_names_out = pipeline.named_steps['pre'].get_feature_names_out()  # 특성 이름 추출
            print(f"Number of features after preprocessing: {len(feature_names_out)}")
            return pipeline, list(feature_names_out)

        except Exception as e:
            print(f"Error extracting feature names: {e}")
            return pipeline, None

    else:
        print(f"Error: Preprocessing pipeline not found at {pipeline_path}")
        return None, None

def load_numpy_data(data_dir, x_filename, y_filename, data_type="Training"):
    x_path = data_dir / x_filename
    y_path = data_dir / y_filename

    if x_path.exists() and y_path.exists():
        x_data = np.load(x_path)
        y_data = np.load(y_path)
        print(f"Preprocessed {data_type.lower()} data loaded: X shape {x_data.shape}, y shape {y_data.shape}")
        return x_data, y_data

    else:
        print(f"Error: Preprocessed {data_type.lower()} data files not found ({x_filename}, {y_filename}).")
        return None, None

def get_class_indices(y_data, data_type="Data"):
    if y_data is None:
        print(f"Warning: y_data for {data_type} is None.")
        return np.array([]), np.array([])

    idx_pos = np.where(y_data == 1.0)[0]  # 전체 악성 클래스 인덱스 추출
    idx_neg = np.where(y_data == 0.0)[0]  # 전체 정상 클래스 인덱스 추출
    print(f"{data_type}: Malicious: {len(idx_pos)}, Benign: {len(idx_neg)}")
    return idx_pos, idx_neg               # (len(idx_pos),), (len(idx_neg),)

def load_pytorch_model(model_cls, input_dim, model_weights_path, device_instance, p_drop=0.5):
    if input_dim is None:
        print("Error: Input dimension not available for model initialization.")
        return None

    if not model_weights_path.exists():
        print(f"Error: Model weights not found at {model_weights_path}")
        return None

    model_instance = model_cls(input_dim, p_drop=p_drop)

    try:
        model_instance.load_state_dict(torch.load(model_weights_path, map_location=device_instance))
        model_instance.to(device_instance)
        model_instance.eval()
        print(f"Model loaded from {model_weights_path} and set to eval mode on {device_instance}.")
        return model_instance

    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

def create_lime_background_data(x_train_data, idx_train_pos_data, idx_train_neg_data, n_bg_per_class_lime):
    if x_train_data is None or idx_train_pos_data.size == 0 or idx_train_neg_data.size == 0:
        print("Error: Training data not available or not enough samples for LIME background data.")
        return None

    num_pos_samples = min(n_bg_per_class_lime, len(idx_train_pos_data))
    num_neg_samples = min(n_bg_per_class_lime, len(idx_train_neg_data))

    if num_pos_samples == 0 or num_neg_samples == 0:
        print("Warning: Not enough samples in one or both classes to create balanced background data.")

        if len(x_train_data) > 0 :
            print(f"Using all available {len(x_train_data)} training samples as background.")
            return x_train_data

        else:
            return None

    bg_pos_indices = np.random.choice(idx_train_pos_data, size=num_pos_samples, replace=False)
    x_bg_pos = x_train_data[bg_pos_indices]
    bg_neg_indices = np.random.choice(idx_train_neg_data, size=num_neg_samples, replace=False)
    x_bg_neg = x_train_data[bg_neg_indices]
    x_background_data = np.vstack((x_bg_pos, x_bg_neg))  # 두 클래스 샘플 결합
    np.random.shuffle(x_background_data)                 # 셔플
    print(f"LIME background data created with shape: {x_background_data.shape}")
    return x_background_data

def generate_and_save_lime_explanation(
    instance_vector,             # 1차원 NumPy 배열: 설명 대상 샘플의 특성 값 벡터
    instance_idx_str,            # 샘플 ID를 나타내는 문자열
    instance_actual_label_idx,   # 샘플의 실제 클래스 인덱스 (0 또는 1)
    explainer_obj,               # LIME Explainer 객체
    model_pred_fn,               # 예측 함수: 확률 배열 반환
    model_instance,              # PyTorch 모델 인스턴스
    device_instance,             # CPU 또는 CUDA
    num_lime_features,           # LIME 설명 시 포함할 최대 특성 개수
    num_lime_samples_lime,       # LIME 설명 시 생성할 섭동 샘플 개수
    distance_metric_lime_param,  # LIME에서 사용할 거리 파라미터
    target_class_idx,            # 설명 대상 클래스 인덱스
    save_dir_path,               # 저장 경로
    feature_names_list_lime,     # 특성 이름 리스트 (instance_vector와 1:1 대응)
    class_names_list_lime        # 클래스 이름 리스트 (모델 출력 인덱스와 1:1 대응)
):
    # 필수 컴포넌트 유효성 검사
    if explainer_obj is None or instance_vector is None or save_dir_path is None or feature_names_list_lime is None:
        print(f"Skipping LIME for instance {instance_idx_str} due to missing components.")
        return None, None

    actual_class_name = class_names_list_lime[instance_actual_label_idx]  # 실제 클래스 인덱스
    target_class_name = class_names_list_lime[target_class_idx]           # 설명 대상 클래스 인덱스
    print(f"\nGenerating LIME explanation for instance: {instance_idx_str} (Actual: {actual_class_name})")
    print(f"Explaining for class: {target_class_name} (Index: {target_class_idx})")
    prob_malicious_title = -1.0

    if model_instance and device_instance:
        instance_tensor = torch.from_numpy(instance_vector).float().unsqueeze(0).to(device_instance)

        with torch.no_grad():
            logit = model_instance(instance_tensor)
            prob_malicious_title = torch.sigmoid(logit).item()

        print(f"Model's predicted probability of Malicious: {prob_malicious_title:.4f}")

    # 단일 인스턴스에 대한 LIME 분석
    try:
        explanation = explainer_obj.explain_instance(
            data_row=instance_vector,
            predict_fn=lambda x: model_pred_fn(x, model_instance, device_instance),
            num_features=num_lime_features,
            num_samples=num_lime_samples_lime,
            distance_metric=distance_metric_lime_param,
            labels=(target_class_idx,)
        )

    except Exception as e:
        print(f"Error during LIME explain_instance for {instance_idx_str}: {e}")
        return None, None

    # 경로 문자열 검사
    if not isinstance(save_dir_path, Path):
        save_dir_path = Path(save_dir_path)

    # 경로 생성
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # 특성 기여도 결과
    explanation_list = explanation.as_list(label=target_class_idx)
    explanation_df = pd.DataFrame(explanation_list, columns=['feature_rule', 'weight'])

    # LIME 분석 과정에서 생성된 조건식 문자열에서 특성 이름 추출
    def get_original_feature_from_rule(rule, all_features):
        parts = rule.split()          # 공백 기준 분리
        potential_feature = parts[0]  # 가장 첫 번째 요소 선택

        if potential_feature in all_features:
            return potential_feature

        for fn in all_features:
            if rule.startswith(fn + "="):
                 return fn

        return rule

    # 특성 기여도 결과 저장
    explanation_df['original_feature'] = explanation_df['feature_rule'].apply(lambda x: get_original_feature_from_rule(x, feature_names_list_lime))
    list_path = save_dir_path / f"lime_feature_contribution_sample_{instance_idx_str}_exp_{target_class_name}.csv"
    explanation_df.to_csv(list_path, index=False)
    print(f"LIME feature contribution saved to: {list_path}")

    # 시각화
    top_n_features_for_plot = explanation_df.sort_values("weight")

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    bar_colors = ['red' if w < 0 else 'blue' for w in top_n_features_for_plot['weight']]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_n_features_for_plot) * 0.5)))

    sns.barplot(data=top_n_features_for_plot, x="weight", y="feature_rule", hue="feature_rule", palette=bar_colors, dodge=False, ax=ax, legend=False, edgecolor=".6")

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Feature contribution weight")
    ax.set_ylabel("Feature rule")
    ax.set_title(
        f"Top {len(top_n_features_for_plot)} LIME features for sample {instance_idx_str} "
        f"(Actual: {actual_class_name}, Pred: {prob_malicious_title})\n"
        f"Explaining class: {target_class_name}",
        fontsize=10
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    custom_plot_path = (save_dir_path / f"lime_top_features_sample_{instance_idx_str}_exp_{target_class_name}.png")
    fig.savefig(custom_plot_path, dpi=300, bbox_inches="tight")
    print(f"LIME top features chart saved to: {custom_plot_path}")
    plt.close(fig)

    return explanation_df, explanation_list

if __name__ == "__main__":
    setup_directories()
    set_random_seeds(SEED)

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {current_device}")

    preprocess_pipeline, feature_names = load_preprocessing_pipeline(DATA_DIR, PIPELINE_FILENAME)

    if preprocess_pipeline is None or feature_names is None:
        print("Exiting: Failed to load preprocessing pipeline or feature names.")
        exit()

    X_train_preprocessed, y_train = load_numpy_data(DATA_DIR, X_TRAIN_FILENAME, Y_TRAIN_FILENAME, "Training")
    X_test_preprocessed, y_test = load_numpy_data(DATA_DIR, X_TEST_FILENAME, Y_TEST_FILENAME, "Test")

    if X_train_preprocessed is None or y_train is None or X_test_preprocessed is None or y_test is None:
        print("Exiting: Failed to load one or more preprocessed data files.")
        exit()

    idx_train_pos, idx_train_neg = get_class_indices(y_train, "Training data")
    idx_test_pos, idx_test_neg = get_class_indices(y_test, "Test data")

    INPUT_DIM = X_train_preprocessed.shape[1]
    model = load_pytorch_model(MLP, INPUT_DIM, DATA_DIR / MODEL_FILENAME, current_device)

    if model is None:
        print("Exiting: Failed to load model.")
        exit()

    X_background = create_lime_background_data(X_train_preprocessed, idx_train_pos, idx_train_neg, N_BG_PER_CLASS)
    explainer = LimeTabularExplainer(
        training_data         = X_background,
        feature_names         = feature_names,
        class_names           = CLASS_NAMES,
        random_state          = SEED)

    if explainer is None:
        print("Exiting: Failed to initialize LIME explainer.")
        exit()

    idx_pos_one = np.random.choice(idx_test_pos) if len(idx_test_pos) > 0 else None
    idx_neg_one = np.random.choice(idx_test_neg) if len(idx_test_neg) > 0 else None
    x_pos, x_neg = None, None

    if idx_pos_one is not None:
        x_pos = X_test_preprocessed[idx_pos_one]
        print(f"\nSelected individual positive sample from test index {idx_pos_one}.")

    else:
        print("Warning: No positive samples in test data for individual explanation.")

    if idx_neg_one is not None:
        x_neg = X_test_preprocessed[idx_neg_one]
        print(f"Selected individual negative sample from test index {idx_neg_one}.")

    else:
        print("Warning: No negative samples in test data for individual explanation.")

    idx_global = []
    num_pos_global = min(N_GLOBAL_PER_CLASS, len(idx_test_pos))

    if num_pos_global > 0:
        idx_global.extend(np.random.choice(idx_test_pos, size=num_pos_global, replace=False))

    num_neg_global = min(N_GLOBAL_PER_CLASS, len(idx_test_neg))

    if num_neg_global > 0:
        idx_global.extend(np.random.choice(idx_test_neg, size=num_neg_global, replace=False))

    X_global, y_global = None, None

    if idx_global:
        idx_global = np.array(idx_global)
        np.random.shuffle(idx_global)
        X_global = X_test_preprocessed[idx_global]
        y_global = y_test[idx_global]
        print(f"\nGlobal aggregation samples created: X_global shape {X_global.shape}")

    else:
        print("Warning: No samples selected for global aggregation.")

    print("\nGenerating LIME explanations for individual samples")

    # 악성 샘플에 대한 LIME 분석
    if x_pos is not None and idx_pos_one is not None:
        generate_and_save_lime_explanation(
            instance_vector=x_pos, instance_idx_str=str(idx_pos_one),
            instance_actual_label_idx=int(y_test[idx_pos_one]),
            explainer_obj=explainer, model_pred_fn=predict_proba_for_lime,
            model_instance=model, device_instance=current_device,
            num_lime_features=NUM_LIME_FEATURES_LOCAL, num_lime_samples_lime=NUM_LIME_SAMPLES_LOCAL,
            distance_metric_lime_param=DISTANCE_METRIC_LIME, target_class_idx=TARGET_CLASS_IDX_EXPLAIN,
            save_dir_path=RESULTS_LIME_DIR, feature_names_list_lime=feature_names, class_names_list_lime=CLASS_NAMES
        )

    # 정상 샘플에 대한 LIME 분석
    if x_neg is not None and idx_neg_one is not None:
        generate_and_save_lime_explanation(
            instance_vector=x_neg, instance_idx_str=str(idx_neg_one),
            instance_actual_label_idx=int(y_test[idx_neg_one]),
            explainer_obj=explainer, model_pred_fn=predict_proba_for_lime,
            model_instance=model, device_instance=current_device,
            num_lime_features=NUM_LIME_FEATURES_LOCAL, num_lime_samples_lime=NUM_LIME_SAMPLES_LOCAL,
            distance_metric_lime_param=DISTANCE_METRIC_LIME, target_class_idx=TARGET_CLASS_IDX_EXPLAIN,
            save_dir_path=RESULTS_LIME_DIR, feature_names_list_lime=feature_names, class_names_list_lime=CLASS_NAMES
        )

    if X_global is not None and y_global is not None:
        print(f"\nGenerating LIME explanations for {X_global.shape[0]} global samples")
        all_top_feature_rules_global = []     # 각 샘플의 상위 10개 룰 이름만 집계하는 리스트
        all_explanations_df_list_global = []  # 각 샘플 DataFrame을 모두 append

        # 경로 생성
        global_individual_save_dir = RESULTS_LIME_DIR / "global_individual_explanations"
        global_individual_save_dir.mkdir(parents=True, exist_ok=True)

        # 개별 전역 샘플에 대한 지역 LIME 설명을 생성한 뒤, 전역 통계를 계산하기 위해 두 개의 컨테이너 리스트에 결과 수집
        for i in range(X_global.shape[0]):
            instance_vector = X_global[i]
            instance_original_idx = idx_global[i]
            instance_actual_label_idx = int(y_global[i])

            if (i + 1) % 50 == 0 or i == X_global.shape[0] -1:
                 print(f"Explaining global sample {i+1}/{X_global.shape[0]} (Original index: {instance_original_idx})")

            current_explanation_df, current_explanation_list = generate_and_save_lime_explanation(
                instance_vector=instance_vector, instance_idx_str=f"global_{instance_original_idx}",
                instance_actual_label_idx=instance_actual_label_idx,
                explainer_obj=explainer, model_pred_fn=predict_proba_for_lime,
                model_instance=model, device_instance=current_device,
                num_lime_features=NUM_LIME_FEATURES_GLOBAL, num_lime_samples_lime=NUM_LIME_SAMPLES_GLOBAL,
                distance_metric_lime_param=DISTANCE_METRIC_LIME, target_class_idx=TARGET_CLASS_IDX_EXPLAIN,
                save_dir_path=global_individual_save_dir,
                feature_names_list_lime=feature_names, class_names_list_lime=CLASS_NAMES
            )

            if current_explanation_list:
                for feature_rule, weight in current_explanation_list:
                    all_top_feature_rules_global.append(feature_rule)

            if current_explanation_df is not None:
                all_explanations_df_list_global.append(current_explanation_df)

        # 전역 샘플에서 가장 자주 등장한 룰의 빈도 계산
        if all_top_feature_rules_global:
            feature_rule_counts = Counter(all_top_feature_rules_global)
            most_common_feature_rules = feature_rule_counts.most_common(TOP_N_FOR_GLOBAL_VISUALIZATION)
            df_freq = pd.DataFrame(most_common_feature_rules, columns=['feature_rule', 'frequency']).sort_values(by='frequency', ascending=False)

            # 시각화
            sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
            palette = sns.color_palette("crest", len(df_freq))
            fig, ax = plt.subplots(figsize=(12, max(6, len(df_freq) * 0.4)))

            sns.barplot(data=df_freq, x="frequency", y="feature_rule", hue="feature_rule", palette=palette, dodge=False, legend=False, ax=ax, edgecolor=".6")

            ax.set_xlabel("Frequency in top LIME features")
            ax.set_ylabel("LIME feature rule")
            ax.set_title(
                f"Top {len(df_freq)} most frequent LIME feature rules\n"
                f"Explaining class: {CLASS_NAMES[TARGET_CLASS_IDX_EXPLAIN]}"
            )

            sns.despine(left=True, bottom=True)
            plt.tight_layout()

            freq_plot_path = RESULTS_LIME_DIR / f"lime_global_feature_freq_exp_{CLASS_NAMES[TARGET_CLASS_IDX_EXPLAIN]}.png"
            fig.savefig(freq_plot_path, dpi=300, bbox_inches="tight")
            print(f"\nLIME global features frequency plot saved to: {freq_plot_path}")
            plt.close(fig)

        # 모든 샘플에 대한 DataFrame을 합친 후 각 특성의 평균 절대 가중치를 계산
        if all_explanations_df_list_global:
            all_explanations_df_combined = pd.concat(all_explanations_df_list_global, ignore_index=True)

            if not all_explanations_df_combined.empty:
                all_explanations_df_combined["abs_weight"] = all_explanations_df_combined["weight"].abs()
                avg_abs_weights = (all_explanations_df_combined.groupby("feature_rule")["abs_weight"].mean())

                top_avg_abs_weights = (avg_abs_weights.sort_values(ascending=False).head(TOP_N_FOR_GLOBAL_VISUALIZATION))

                # 시각화
                sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
                palette = sns.color_palette("flare", len(top_avg_abs_weights))

                fig, ax = plt.subplots(figsize=(12, max(6, len(top_avg_abs_weights) * 0.4)))

                sns.barplot(x=top_avg_abs_weights.values, y=top_avg_abs_weights.index, hue=top_avg_abs_weights.index, palette=palette, dodge=False, legend=False, ax=ax, edgecolor=".6",)

                ax.set_xlabel("Mean absolute LIME weight")
                ax.set_ylabel("LIME feature rule")
                ax.set_title(
                    f"Top {len(top_avg_abs_weights)} LIME feature rules by mean absolute weight\n"
                    f"Explaining class: {CLASS_NAMES[TARGET_CLASS_IDX_EXPLAIN]}"
                )

                sns.despine(left=True, bottom=True)
                plt.tight_layout()

                avg_weight_plot_path = (RESULTS_LIME_DIR / f"lime_global_avg_abs_weight_exp_{CLASS_NAMES[TARGET_CLASS_IDX_EXPLAIN]}.png")
                fig.savefig(avg_weight_plot_path, dpi=300, bbox_inches="tight")
                print(f"LIME global mean absolute weight plot saved to: {avg_weight_plot_path}")
                plt.close(fig)

                global_summary_path = (RESULTS_LIME_DIR / f"lime_global_feature_summary_exp_{CLASS_NAMES[TARGET_CLASS_IDX_EXPLAIN]}.csv")
                all_explanations_df_combined.to_csv(global_summary_path, index=False)
                print(f"LIME global features summary saved to: {global_summary_path}")

            else:
                print("\nNo valid LIME explanation DataFrames to combine for mean absolute weights.")

        else:
            print("\nNo LIME explanations DataFrames generated to calculate mean absolute weights.")

    else:
        print("\nSkipping global LIME explanations aggregation due to no global samples selected or processed.")

    print("\nLIME analysis finished.")