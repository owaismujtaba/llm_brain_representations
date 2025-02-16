import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config as config


def plot_r2(model):

    destination_dir = Path(config.CUR_DIR, 'Results', 'Images')
    root_dir = config.TRAINED_DIR
    subject_r2_scores = {}

    for subject in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject, "Mapping", model)
        print(subject_path)
        if os.path.exists(subject_path):
            r2_scores = []
            
            for fold_file in sorted(os.listdir(subject_path)):
                if fold_file.endswith(".npy"):
                    fold_path = os.path.join(subject_path, fold_file)
                    values = np.load(fold_path) 
                    r2_scores.append(values[2])  
            
            if r2_scores:
                subject_r2_scores[subject] = r2_scores

    subjects = []
    scores = []
    
    for subject, r2_values in subject_r2_scores.items():
        subjects.extend([subject] * len(r2_values))
        scores.extend(r2_values)

    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 6))
    sns.violinplot(x=subjects, y=scores, inner="box", palette="coolwarm")

    plt.xlabel("Subjects", fontsize=12, fontweight="bold")
    plt.ylabel("R²", fontsize=12, fontweight="bold")
    plt.title("R² distribution across 5 folds", fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    os.makedirs(destination_dir, exist_ok=True)

    filename = Path(destination_dir, f'{model}.png')

    plt.savefig(filename, dpi=800)
