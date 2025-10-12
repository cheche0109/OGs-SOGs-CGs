import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             roc_curve, auc, confusion_matrix)
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from collections import Counter
import shap
# Load data
data = pd.read_csv("../../All_features_noNA.csv")

# Load gene labels
orphan_genes = set(pd.read_csv("../EOGs_genomes_over1_700_.txt", header=None)[0].tolist())
SOG_genes = set(pd.read_csv("../SOGs_genomes_over1_700_.txt", header=None)[0].tolist())

# Assign labels
data.loc[data.iloc[:, 0].isin(SOG_genes), 'label'] = 0
data.loc[data.iloc[:, 0].isin(orphan_genes), 'label'] = 1

# Remove rows without labels


# Downsample to balance the classes
SOG_sample = resample(data[data['label'] == 0], n_samples=26732, random_state=42, replace=False) #123065
orphan_sample = resample(data[data['label'] == 1], n_samples=26732, random_state=42, replace=False)
print("\nAfter downsampling:")
print(f"Spurious sample size: {len(SOG_sample)}")
print(f"Orphan sample size: {len(orphan_sample)}")

# Combine the balanced dataset
balanced_data = pd.concat([SOG_sample, orphan_sample])

group1 = ["GC", "GC3s", "CpG", "PASTA_aggr_length", "PASTA_aggr_energy", "biosynthetic_cost", "isoelectric_point",
    "upstream_length", "downstream_length", "Tiny", "Small", "Aliphatic", "Aromatic",
    "Non-polar", "Polar", "Charged", "Basic", "Acidic", "signalp", "DNA_Shannon_Entropy",
    "Protein_Shannon_Entropy",
    "Dinuc_AT", "Dinuc_TA", "Dinuc_CG", "Dinuc_AA", "Dinuc_TT", "Dinuc_CC", "Dinuc_GG",
    "Dinuc_AG", "Dinuc_CT", "Dinuc_GT", "Dinuc_AC", "Dinuc_GA", "Dinuc_TC",
    "Trinuc_AAA", "Trinuc_AAT", "Trinuc_AAC", "Trinuc_AAG", "Trinuc_ATA", "Trinuc_ATT", "Trinuc_ATC", "Trinuc_ATG",
    "Trinuc_ACA", "Trinuc_ACT", "Trinuc_ACC", "Trinuc_ACG", "Trinuc_AGA", "Trinuc_AGT", "Trinuc_AGC", "Trinuc_AGG",
    "Trinuc_TAA", "Trinuc_TAT", "Trinuc_TAC", "Trinuc_TAG", "Trinuc_TTA", "Trinuc_TTT", "Trinuc_TTC", "Trinuc_TTG",
    "Trinuc_TCA", "Trinuc_TCT", "Trinuc_TCC", "Trinuc_TCG", "Trinuc_TGA", "Trinuc_TGT", "Trinuc_TGC", "Trinuc_TGG",
    "Trinuc_CAA", "Trinuc_CAT", "Trinuc_CAC", "Trinuc_CAG", "Trinuc_CTA", "Trinuc_CTT", "Trinuc_CTC", "Trinuc_CTG",
    "Trinuc_CCA", "Trinuc_CCT", "Trinuc_CCC", "Trinuc_CCG", "Trinuc_CGA", "Trinuc_CGT", "Trinuc_CGC", "Trinuc_CGG",
    "Trinuc_GAA", "Trinuc_GAT", "Trinuc_GAC", "Trinuc_GAG", "Trinuc_GTA", "Trinuc_GTT", "Trinuc_GTC", "Trinuc_GTG",
    "Trinuc_GCA", "Trinuc_GCT", "Trinuc_GCC", "Trinuc_GCG", "Trinuc_GGA", "Trinuc_GGT", "Trinuc_GGC", "Trinuc_GGG",
    "AAC_A", "AAC_C", "AAC_D", "AAC_E", "AAC_F", "AAC_G", "AAC_H", "AAC_I", "AAC_K", "AAC_L", "AAC_M", "AAC_N",
    "AAC_P", "AAC_Q", "AAC_R", "AAC_S", "AAC_T", "AAC_V", "AAC_W", "AAC_Y",
    "Avg_Hydropathy", "Avg_Residue_Weight", "Avg_Charge"
]

group2 = group1 + ["C_Percentage", "H_Percentage", "E_Percentage", "PHOBIUS_no_tm_domains", "PredProp_diso_pct", "PredProp_tm2_pct", "a","c","d","e","f","g","h","i","k","l","m","n","p","q","r","s","t","v","w","y"]

#group4 = list(data.columns[1:-1])  # All features except ID and label
group3 = group2 + ["Number_genomes_gene_is_present","Number_genomes_in_species","tree_terminal_branch_len_species", "MDS1", "MDS2", "MDS3"]


# Feature groups dictionary
feature_groups = {"Feature Group 1": group1, "Feature Group 2": group2, "Feature Group 3": group3}

# Colors for AUC plot
group_colors = {"Feature Group 1": "black", "Feature Group 2": "green", "Feature Group 3": "red"}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_results = {}

plt.figure(figsize=(8, 6))

for group_name, features in feature_groups.items():
    print(f"Processing {group_name} with {len(features)} features...")
    selected_features = [f for f in features if f in balanced_data.columns]
    X = balanced_data[selected_features]
    y = balanced_data["label"]

    assert not X.isnull().values.any(), "❌ 数据中仍有缺失值，请检查！"

    pipe = Pipeline([
    ("scaler",  StandardScaler()),
    ("clf",     XGBClassifier(objective="binary:logistic", eval_metric="auc",
                              random_state=42))
    ])

    param_grid = {
        "clf__max_depth":        [3, 5, 7],
        "clf__learning_rate":    [0.005, 0.01, 0.1],
        "clf__n_estimators":     [100, 200],
        "clf__subsample":        [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    all_predictions, all_labels = [], []
    feature_importance_values = np.zeros(len(selected_features))

    accuracies, f1_scores, roc_auc_scores = [], [], []
    best_params_list = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
        gs.fit(X_train, y_train)

        best_params_list.append(gs.best_params_)
        best_model = gs.best_estimator_

        #best_model.fit(X_train, y_train)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        all_predictions.extend(y_pred)
        all_labels.extend(y_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        roc_auc_scores.append(roc_auc)

        feature_importance_values += best_model.named_steps['clf'].feature_importances_

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    auc_results[group_name] = mean_auc
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    mean_roc_auc = np.mean(roc_auc_scores)
    most_common_params = dict(Counter(tuple(sorted(d.items())) for d in best_params_list).most_common(1)[0][0])

    performance_report = classification_report(all_labels, all_predictions, target_names=['Spurious', 'Orphans'])
    with open(f"{group_name}_performance_1.2.txt", "w") as f:
        f.write(f"Performance Report for {group_name}:\n")
        f.write(f"\nMean Accuracy: {mean_accuracy:.4f}")
        f.write(f"\nMean F1-Score: {mean_f1:.4f}")
        f.write(f"\nMean ROC-AUC Score: {mean_roc_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"\nBest Parameters (most common across folds): {most_common_params}\n")
        f.write(performance_report)
        
    # Save confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Sourious', 'Orphans'],
                yticklabels=['Spurious', 'Orphans'], annot_kws={"size": 13})
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    plt.savefig(f"{group_name}_confusion_matrix_1.2.png", dpi=800, bbox_inches="tight")
    plt.close()

    # Save feature importance
    mean_feature_importance = feature_importance_values / cv.get_n_splits()
    feature_importance_series = pd.Series(mean_feature_importance, index=selected_features).sort_values(ascending=False)
    
    # Save feature importance as txt and png
    feature_importance_series.to_csv(f"{group_name}_feature_importance_1.2.txt", sep='\t', header=True)


    # Specify which features should be color1 (modify as needed)
    feature1 = ["GC", "GC3s", "CpG", "PASTA_aggr_length", "PASTA_aggr_energy", "biosynthetic_cost", "isoelectric_point",
        "upstream_length", "downstream_length", "Tiny", "Small", "Aliphatic", "Aromatic",
        "Non-polar", "Polar", "Charged", "Basic", "Acidic", "OTHER", "DNA_Shannon_Entropy",
        "Protein_Shannon_Entropy",
        "Dinuc_AT", "Dinuc_TA", "Dinuc_CG", "Dinuc_AA", "Dinuc_TT", "Dinuc_CC", "Dinuc_GG",
        "Dinuc_AG", "Dinuc_CT", "Dinuc_GT", "Dinuc_AC", "Dinuc_GA", "Dinuc_TC",
        "Trinuc_AAA", "Trinuc_AAT", "Trinuc_AAC", "Trinuc_AAG", "Trinuc_ATA", "Trinuc_ATT", "Trinuc_ATC", "Trinuc_ATG",
        "Trinuc_ACA", "Trinuc_ACT", "Trinuc_ACC", "Trinuc_ACG", "Trinuc_AGA", "Trinuc_AGT", "Trinuc_AGC", "Trinuc_AGG",
        "Trinuc_TAA", "Trinuc_TAT", "Trinuc_TAC", "Trinuc_TAG", "Trinuc_TTA", "Trinuc_TTT", "Trinuc_TTC", "Trinuc_TTG",
        "Trinuc_TCA", "Trinuc_TCT", "Trinuc_TCC", "Trinuc_TCG", "Trinuc_TGA", "Trinuc_TGT", "Trinuc_TGC", "Trinuc_TGG",
        "Trinuc_CAA", "Trinuc_CAT", "Trinuc_CAC", "Trinuc_CAG", "Trinuc_CTA", "Trinuc_CTT", "Trinuc_CTC", "Trinuc_CTG",
        "Trinuc_CCA", "Trinuc_CCT", "Trinuc_CCC", "Trinuc_CCG", "Trinuc_CGA", "Trinuc_CGT", "Trinuc_CGC", "Trinuc_CGG",
        "Trinuc_GAA", "Trinuc_GAT", "Trinuc_GAC", "Trinuc_GAG", "Trinuc_GTA", "Trinuc_GTT", "Trinuc_GTC", "Trinuc_GTG",
        "Trinuc_GCA", "Trinuc_GCT", "Trinuc_GCC", "Trinuc_GCG", "Trinuc_GGA", "Trinuc_GGT", "Trinuc_GGC", "Trinuc_GGG",
        "AAC_A", "AAC_C", "AAC_D", "AAC_E", "AAC_F", "AAC_G", "AAC_H", "AAC_I", "AAC_K", "AAC_L", "AAC_M", "AAC_N",
        "AAC_P", "AAC_Q", "AAC_R", "AAC_S", "AAC_T", "AAC_V", "AAC_W", "AAC_Y",
        "Avg_Hydropathy", "Avg_Residue_Weight", "Avg_Charge"]

    feature2 = ["C_Percentage", "H_Percentage", "E_Percentage", "PHOBIUS_no_tm_domains", "PredProp_diso_pct", "PredProp_tm2_pct", "a","c","d","e","f","g","h","i","k","l","m","n","p","q","r","s","t","v","w","y"]

    #group4 = list(data.columns[1:-1])  # All features except ID and label
    feature3 = ["Number_genomes_gene_is_present","Number_genomes_in_species","tree_terminal_branch_len_species", "MDS1", "MDS2", "MDS3", "tax"]

    threshold = feature_importance_series.mean()
    important_features = feature_importance_series[feature_importance_series > threshold]
    

    # Assign colors based on feature names
    bar_colors = []


    color_map = {
        "sequence features": "black",
        "structure features": "green",
        "evolutionary features": "red",
    }


    for feature in important_features.index:
        if feature in feature1:
            bar_colors.append(color_map["sequence features"])
        elif feature in feature2:
            bar_colors.append(color_map["structure features"])
        elif feature in feature3:
            bar_colors.append(color_map["evolutionary features"])

    unique_colors = set(bar_colors)

    # Plot Feature Importanc
    plt.figure(figsize=(15, 6))
    important_features.plot(kind="bar", color=bar_colors)
    plt.axhline(y=threshold, color="black", linestyle="--")
    plt.ylabel("Feature importance", fontsize=14)
    plt.xticks(rotation=90)

    if len(unique_colors) > 0:
        legend_labels = {name: color for name, color in color_map.items() if color in unique_colors}
        legend_patches = [plt.Line2D([0], [0], color=color, lw=6, label=name) for name, color in legend_labels.items()]
        plt.legend(handles=legend_patches, loc="upper right")

    plt.savefig(f"{group_name}_feature_importance_1.2.png", dpi=800, bbox_inches="tight")
    plt.close()

    # Plot ROC curve for each group
    plt.plot(fpr, tpr, label=f"{group_name} (AUC = {mean_auc:.2f})", color=group_colors[group_name])

# Finalize AUC plot
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)  # Random classifier
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("auc_comparison_plot_1.2.png", dpi=800, bbox_inches="tight")
plt.close()

# --- SHAP on a final refit (optional) ---
print("Computing SHAP values...")
final_model = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y).best_estimator_

# Extract fitted classifier and transformed X
clf = final_model.named_steps['clf']
X_trans = final_model.named_steps['scaler'].transform(X)

# SHAP
explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_trans)  # shape: (n_samples, n_features) for binary

# Save mean abs SHAP values
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
pd.Series(mean_abs_shap, index=selected_features).sort_values(ascending=False)\
  .to_csv("shap_values_1.2.txt", sep="\t", header=True)

# --- Relabel features for summary plot ---
renamed_features = []
for feat in selected_features:
    if feat in feature1:
        renamed_features.append("Fseq_" + feat)
    elif feat in feature2:
        renamed_features.append("Fstruct_" + feat)
    elif feat in feature3:
        renamed_features.append("Fevo_" + feat)
    else:
        renamed_features.append(feat)  # leave untouched if not in any group

# SHAP summary plot with new labels
shap.summary_plot(shap_vals, X_trans, feature_names=renamed_features, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot_1.2.png", dpi=800)
plt.close()
