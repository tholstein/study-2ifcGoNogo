# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:04:36 2026

@author: Thomas Holstein 
thomas.holstein.pro@proton.me

"""

#%% ________________________________ LIBRAIRIES _______________________________
import os
import numpy as np
import mne
# matplotlib.use('Qt5Agg') #block=True
import matplotlib.pyplot as plt

from mne.decoding import (
    LinearModel,
    Vectorizer,
    get_coef,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
from mne.stats import fdr_correction

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from mne import EvokedArray
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.metrics import roc_auc_score
from PIL import Image



# Utilisation de tous les cœurs du PC
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
#%% ______________________________ PATH MANAGMENT _____________________________
# Start from the current script's location
current_path = Path(__file__).resolve()

# Traverse parent directories until 'STUDY' is found
for parent in current_path.parents:
    if parent.name == "STUDY":
        study_path = parent
        break
else:
    raise FileNotFoundError("Could not find 'OSF_AREEG' folder in the directory tree.")

# Build the full path to the plot directory
plot_dir = study_path / "Plots" / "mvpa"
plot_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Path of the epochs data in bids format
bids_derivatives_dir = study_path / "Data" / "eeg" / "derivatives" / "epochs"

# Path of the inputs for the electrodes selection
inputs_dir = study_path / "Code" / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

outputs_dir = study_path / "Code" / "outputs"


#%% ________________________________ CONSTANTS ________________________________

# -----------------------------------------------------------------------------
# ---------------------------- TIME WINDOWS SELECTION -------------------------
# -----------------------------------------------------------------------------
TIME_WINDOW_MS = 50 # you can test with time windows of 20 and 100 according to the paper parameters of the Supplementary Material
# you can also mark None (or 2ms for 500hz sampling) if you don't want time windows

# -----------------------------------------------------------------------------
# ----------------------------- ELECTRODES SELECTION --------------------------
# -----------------------------------------------------------------------------
REDUCE_NUMBER_OF_ELECTRODES = True 
# False if you don't want to reduce the electrodes (whole scalp: 60 channels)

# If you want to reduce the electrodes number (n features) define the number below
N_CHANNELS_PER_TIME_POINT_PER_CONDITION  = 28 # number of electrodes which are then pooled for each condition (choose 5, 10 or 15) 
# OR take only electrodes from visual cortex 
VISUAL_CORTEX = True 
# if False: Nested Feature Selection -> feature selection inside an inner training loop 
#... while evaluating model performance on an independent outer loop. 
# It stops data leakage and gives a true measure of how well a model works on new data

# if True: a priori ROI based on anatomy (hypothesis-driven)

# -----------------------------------------------------------------------------
# ----------------------------- ALGORITHM SELECTION ---------------------------
# -----------------------------------------------------------------------------
CLASSIFIER = "LDA" # possible to test an other algorithm with "LR", "SVM", "RF" 


#%% ________________________________ FUNCTIONS ________________________________

def reduce_temporal_resolution(data, time_window_ms, sfreq):
    # Calculate the number of samples per window based on time window in milliseconds
    samples_per_window = int(sfreq * time_window_ms / 1000)

    # Calculate the new number of time points after averaging
    num_time_points = data.shape[2] // samples_per_window

    # Create an empty array to store the averaged data
    averaged_data = np.zeros((data.shape[0], data.shape[1], num_time_points))

    # Average the data over each N ms window
    for i in range(num_time_points):
        start_index = i * samples_per_window
        end_index = (i + 1) * samples_per_window
        averaged_data[:, :, i] = np.mean(data[:, :, start_index:end_index], axis=2)

    return averaged_data


def preprocess_epochs(epochs, time_window_ms, sfreq):
    # Pick EEG channels and exclude specified channels
    epochs.pick(picks="eeg", exclude=('bads', 'eog'))
        
    # Get data and events
    x = epochs.get_data()
    if TIME_WINDOW_MS:
        x_averaged = reduce_temporal_resolution(x, time_window_ms, sfreq)
        x = x_averaged
        
        """     /!\    To be changed if the epoch window is different    /!\   """
        time = np.arange(-.2, 0.8, time_window_ms/1000) 
    else:
        time = epochs.times
        
    y = epochs.events[:, 2]

    return x, y, time
    
def update_event_ids(epochs):
    epochs_updated = epochs.copy()
    for i in range(len(epochs_updated.events)):
        if epochs_updated.events[i, 2] % 2 == 0:
            epochs_updated.events[i, 2] = 2
        else:
            epochs_updated.events[i, 2] = 1
    return epochs_updated

def temporal_decoding(X, y, window_size, classifier, REDUCE_NUMBER_OF_ELECTRODES, epochs):    
    
    if classifier == "LDA":
        clf = make_pipeline(StandardScaler(), LinearModel(LinearDiscriminantAnalysis()))
        map_model = make_pipeline(Vectorizer(), StandardScaler(), LinearModel(LinearDiscriminantAnalysis()))
    else: 
        print("Classifier other than LDA is not implemented.")

    cv = StratifiedKFold(n_splits=10, shuffle=False)
    print("\n========================================")
    print(f"Number of trials: {len(y)}")
    print(f"Class distribution: {Counter(y)}")
    
    # Nested feature selection manual cross-validation loop
    scores_folds = []
    coef_patterns_folds = []
    coef_filters_folds = []
    selection_folds = []
    
    n_times = X.shape[2]
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"Fold {fold:02d}")
        print(f"   Train: {Counter(y_train)}")
        print(f"   Test : {Counter(y_test)}")
        
        fold_scores = np.zeros(n_times)
        full_patterns = np.zeros((X.shape[1], n_times))
        full_filters = np.zeros((X.shape[1], n_times))
        fold_selections = np.zeros((X.shape[1], n_times))

        for t in range(n_times):
            X_train_t = X_train[:, :, t]
            X_test_t = X_test[:, :, t]
            
            # NESTED FEATURE SELECTION 
            if REDUCE_NUMBER_OF_ELECTRODES and not VISUAL_CORTEX:
                # Select best electrodes using ONLY training data for this specific time window
                erp_left = np.mean(X_train_t[y_train == 1, :], axis=0)
                erp_right = np.mean(X_train_t[y_train == 2, :], axis=0)
                diff = np.abs(erp_left - erp_right)
                
                # Take top N electrodes STRICTLY for this time window
                selected_features = np.argsort(diff)[::-1][:N_CHANNELS_PER_TIME_POINT_PER_CONDITION]
            else:
                selected_features = list(range(X.shape[1]))
            
            fold_selections[selected_features, t] = 1.0

            # Subset the data with the selected electrodes for "t"
            X_train_t_fold = X_train_t[:, selected_features]
            X_test_t_fold = X_test_t[:, selected_features]

            # Train and score on the specific time point
            clf.fit(X_train_t_fold, y_train)
            
            if hasattr(clf, "decision_function"):
                y_pred = clf.decision_function(X_test_t_fold)
            else:
                y_pred = clf.predict_proba(X_test_t_fold)[:, 1]
                
            fold_scores[t] = roc_auc_score(y_test, y_pred)
            
            # Map Model extraction (Spatial coefficients) for "t"
            map_model.fit(X_train_t_fold, y_train)
            
            fold_patterns_t = get_coef(map_model, 'patterns_', inverse_transform=True)
            fold_filters_t = get_coef(map_model, 'filters_', inverse_transform=True)
            
            full_patterns[selected_features, t] = fold_patterns_t
            full_filters[selected_features, t] = fold_filters_t

        # Save data for this fold
        scores_folds.append(fold_scores)
        coef_patterns_folds.append(full_patterns)
        coef_filters_folds.append(full_filters)
        selection_folds.append(fold_selections)

    # Average scores, spatial coefficients and selection frequencies across cross-validation splits
    scores = np.array(scores_folds)
    scores_mean = np.mean(scores, axis=0)
    
    coef_patterns = np.mean(np.array(coef_patterns_folds), axis=0)
    coef_filters = np.mean(np.array(coef_filters_folds), axis=0)
    
    coef_selections = np.mean(np.array(selection_folds), axis=0)

    if TIME_WINDOW_MS:
        sample_rate_updated = (1000/window_size)   
        epoch_updated = epochs.copy().resample(sample_rate_updated)
    else:
        sample_rate_updated = 500
        epoch_updated = epochs.copy()

    topomap_evoked_patterns = EvokedArray(coef_patterns, epoch_updated.info, tmin=epochs.tmin)  
    topomap_evoked_filters = EvokedArray(coef_filters, epoch_updated.info, tmin=epochs.tmin)  
    
    selection_percent = coef_selections * 100.0
    topomap_evoked_selections = EvokedArray(selection_percent / 1e6, epoch_updated.info, tmin=epochs.tmin)

    return (scores, scores_mean, coef_patterns, topomap_evoked_patterns, 
            coef_filters, topomap_evoked_filters, coef_selections, topomap_evoked_selections)


def statistical_tests(scores, time, alpha=0.05):
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(scores - 0.5, n_permutations=10000)
    mask_permutation = np.zeros_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= alpha:
            mask_permutation[c] = 1
    
    T, pval = stats.ttest_1samp(scores, 0.5)
    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='negcorr')
    significant_times = np.unique(time)[pval_fdr < alpha]
  
    return mask_permutation, significant_times, clusters, cluster_p_values


def statistical_comparaison_tests(scores_conditionA, scores_conditionB, time, alpha=0.05):
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(scores_conditionA - scores_conditionB, n_permutations=10000)
    mask_permutation = np.zeros_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= alpha:
            mask_permutation[c] = 1
            
    T, pval = stats.ttest_1samp(scores_conditionA - scores_conditionB, 0)
    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='negcorr')
    significant_times = np.unique(time)[pval_fdr < alpha]
    
    return mask_permutation, significant_times, clusters, cluster_p_values
    
def plot_temporal_decoding_conditions(scores_mean_list, std_error_list, titles, significance_times_list, colors, mask_condition_list, time, n_features):
    # if TIME_WINDOW_MS == 50:
    #     plt.figure(figsize=(24, 15))
    #     fontsize_suptitle = 28
    #     fontsize_xylabels = 23
    #     fontsize_legend = 24
    #     markersize = 9
    #     ticksize = 21
    #     lineplotsize = 2.5
    # elif TIME_WINDOW_MS == 100:  
    plt.figure(figsize=(24, 15))
    
    fontsize_suptitle = 42
    fontsize_xylabels = 34
    fontsize_legend = 36
    markersize = 14
    ticksize = 32
    lineplotsize = 4  
    # plt.figure(figsize=(max(28, len(time) * 3.5), 14))
    plt.suptitle('Temporal Decoding', fontsize=fontsize_suptitle, 
                 # fontweight="bold", 
                 y=1)

    for i, (scores_mean, std_error, title, significance_times, color, mask_condition) in enumerate(zip(scores_mean_list, std_error_list, titles, significance_times_list, colors, mask_condition_list), start=1):
        plt.subplot(2, 2, i)  
        ax = sns.lineplot(x=time, y=scores_mean, label=title, color=color, linewidth=lineplotsize)
        ax.fill_between(time, scores_mean - std_error, scores_mean + std_error, color=color, alpha=0.3)
        ax.axhline(.5, color='k', linestyle='--', label="chance")
        # ax.set_title('Temporal Decoding', fontsize=30, fontweight="bold")
        ax.set_xlabel('Time (ms)', fontsize=fontsize_xylabels)
        ax.set_ylabel('AUC', fontsize=fontsize_xylabels)
        ax.axvline(0, color='k', linestyle='dotted')
        ax.set_xlim([-.2,.8])
        ax.set_ylim([0.45,.61])
        # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f')) # to choose number of decimal
        ax.set_yticks([0.45, 0.50, 0.55, 0.60])
        
        ax.legend(loc='upper left', fontsize=fontsize_legend, frameon=False)
        ax.set_xticklabels((ax.get_xticks() * 1000).astype(int)) 
        
        mask_flat = np.squeeze(mask_condition)
        y_sig = np.where(mask_flat == 1, 0.49, np.nan)
        plt.plot(time, y_sig, linestyle='-', marker='o', markersize=markersize, color=color, linewidth=markersize)

        ax.tick_params(axis='x', labelsize=ticksize)  
        ax.tick_params(axis='y', labelsize=ticksize)  
    
    plt.tight_layout()  
    plot_file_name = plot_dir / "NFS_per_time_window" / f"{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_electrods" / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_1.jpg"
    plt.savefig(plot_file_name, dpi=600, bbox_inches='tight')
    plt.show()
 

def get_strong_electrods_per_condition_per_significant_time_point(epochs_AR, epochs_PR, epochs_AI, epochs_PI,
                                                                  mean_coef, 
                                                                  mask_AR, mask_PR, mask_AI, mask_PI,
                                                                  VISUAL_CORTEX,
                                                                  n_electrodes=10,
                                                                  time=None):
    if VISUAL_CORTEX == False:
        coef_AR = mean_coef[0]
        coef_PR = mean_coef[1]
        coef_AI = mean_coef[2]
        coef_PI = mean_coef[3]       
        
    else:        
        coef_AR = mean_coef[0,:,:]
        coef_PR = mean_coef[1,:,:]
        coef_AI = mean_coef[2,:,:]
        coef_PI = mean_coef[3,:,:]

    conditions = {
        "AR": {"mask": mask_AR, "coef": coef_AR, "epochs": epochs_AR},
        "PR": {"mask": mask_PR, "coef": coef_PR, "epochs": epochs_PR},
        "AI": {"mask": mask_AI, "coef": coef_AI, "epochs": epochs_AI},
        "PI": {"mask": mask_PI, "coef": coef_PI, "epochs": epochs_PI}
    }

    ch_name_strong_electrods_per_condition_per_significant_time_point = {}

    for condition, data in conditions.items():
        mask = data["mask"]
        coef = data["coef"]
        epochs = data["epochs"]
        
        electrode_names = epochs.ch_names
        ch_name_strong_electrodes_per_time_point = {}

        if time is not None:
            indices_to_consider = np.where(np.round(time, 3) >= 0)[0]
        else:
            indices_to_consider = range(coef.shape[1])

        for index in indices_to_consider:
            coefficients_at_index = coef[:, index]
            index_strong_electrodes = np.argsort(np.abs(coefficients_at_index))[::-1][:n_electrodes]
            ch_name_strong_electrodes = [electrode_names[i] for i in index_strong_electrodes]
            ch_name_strong_electrodes_per_time_point[index] = ch_name_strong_electrodes

        ch_name_strong_electrods_per_condition_per_significant_time_point[condition] = ch_name_strong_electrodes_per_time_point

    return ch_name_strong_electrods_per_condition_per_significant_time_point
 

def plot_topomap_with_mask(grand_average, strong_electrods_per_condition_per_significant_time_point, key_condition, mask_condition, times, coef_name, vlim=(-6, 6)):
    mask_topo = np.zeros(grand_average.data.shape, dtype=bool)
    mask_topo_params = dict(marker='o', markerfacecolor='black', markeredgecolor='white', markersize=8, markeredgewidth=1.5)
    
    for t_val in times:
        idx_time_point = np.argmin(np.abs(grand_average.times - t_val))
        
        if idx_time_point in strong_electrods_per_condition_per_significant_time_point[key_condition]:
            top_n_electrodes = strong_electrods_per_condition_per_significant_time_point[key_condition][idx_time_point]
            for electrode in top_n_electrodes:
                if electrode in grand_average.ch_names:
                    idx_electrode = grand_average.ch_names.index(electrode)
                    mask_topo[idx_electrode, idx_time_point] = True
    
    fig = grand_average.plot_topomap(times=times, vlim=vlim,
                                mask=mask_topo, mask_params=mask_topo_params,
                                show=False, colorbar=True);
    
    fig.suptitle(f"EEG {coef_name} condition {key_condition}")


#Function to calculate the mean of arrays of different sizes
def calculate_mean(condition_data, time_window=TIME_WINDOW_MS):
    max_rows = max(table.shape[0] for table in condition_data)
    
    if time_window:
        n_time_points = int(1000/time_window)
    else:
        n_time_points = int(len(epochs.times))
    
    summed_array = np.zeros((max_rows, n_time_points))
    count_array = np.zeros((max_rows, n_time_points))

    for table in condition_data:
        rows, cols = table.shape
        summed_array[:rows, :cols] += table
        count_array[:rows, :cols] += 1

    mean_array = np.divide(summed_array, count_array, out=np.zeros_like(summed_array), where=count_array!=0)
    return mean_array


def get_mean_coef_patterns_when_electrod_number_differs_per_condition(list_of_coef):
    condition1, condition2, condition3, condition4 = [], [], [], []

    for participant in list_of_coef:
        condition1.append(participant[0])
        condition2.append(participant[1])
        condition3.append(participant[2])
        condition4.append(participant[3])

    mean_condition1 = calculate_mean(condition1)
    mean_condition2 = calculate_mean(condition2)
    mean_condition3 = calculate_mean(condition3)
    mean_condition4 = calculate_mean(condition4)

    return [mean_condition1, mean_condition2, mean_condition3, mean_condition4]


def merge_plots_vertically(image_paths, output_path, spacing=200, dpi_value=600):

    images = [Image.open(x) for x in image_paths] 
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights) + spacing * (len(images) - 1)
    new_im = Image.new('RGB', (max_width, total_height), color='white')

    y_offset = 0
    for im in images:
        x_offset = int((max_width - im.size[0]) / 2)
        new_im.paste(im, (x_offset, y_offset))
        y_offset += im.size[1] + spacing

    new_im.save(output_path, dpi=(dpi_value, dpi_value), quality=100)    
    print(f"Figure sucessfully saved: {output_path}")
#%% _______________________________ MAIN SCRIPT _______________________________
# Define lists to store mean scores
mean_scores_temporal_decoding = []
mean_scores_temporal_generalization = []
list_of_topomap_evoked_AR_patterns = []
list_of_topomap_evoked_PR_patterns = []
list_of_topomap_evoked_AI_patterns = []
list_of_topomap_evoked_PI_patterns = []
list_of_topomap_evoked_AR_filters = []
list_of_topomap_evoked_PR_filters = []
list_of_topomap_evoked_AI_filters = []
list_of_topomap_evoked_PI_filters = []

list_of_topomap_evoked_AR_selections = []
list_of_topomap_evoked_PR_selections = []
list_of_topomap_evoked_AI_selections = []
list_of_topomap_evoked_PI_selections = []

list_of_coef_patterns=[]
list_of_coef_filters=[]
list_of_coef_selections = []

trial_summary = []

# Loop for each participant
for PARTICIPANT_NUM in range(1, 25):  # Assuming participants are numbered from P001 to P024
    
    if PARTICIPANT_NUM < 10:
        SUBJ = f'0{PARTICIPANT_NUM}'
    else:
        SUBJ = f'{PARTICIPANT_NUM}'   
    
    bids_subj = f"sub-{SUBJ}"
    epo_path = bids_derivatives_dir / bids_subj / "eeg" / f"{bids_subj}_task-2ifcGoNogo_proc-epo_eeg.fif"

    # Load epochs
    epochs = mne.read_epochs(epo_path)
    sfreq=500
    epochs.pick(picks="eeg")
    
    if REDUCE_NUMBER_OF_ELECTRODES == True and VISUAL_CORTEX == True:
            electrodes_v1_v2_v3 = ["Oz", "O1", "O2", "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3", "PO4", "PO7", "PO8"]
            posterior_electrodes = ["Oz", "O1", "O2", "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3", "PO4", "PO7", "PO8",
                                    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP9", "TP10"]
            
            centro_posterior_electrodes = ["Oz", "O1", "O2", "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3", "PO4", "PO7", "PO8",
                                    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP9", "TP10",
                                    "C7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "C8"]

            electrodes_of_interest = posterior_electrodes
            epochs.pick_channels(electrodes_of_interest)
            print("\n Take only visual cortex electrodes! ")
            n_features = len(electrodes_of_interest)

    event_ids = epochs.event_id 
    
    # Concatenate epochs for each condition
    AR = mne.concatenate_epochs([epochs["active/relevance/blue"], epochs["active/relevance/red"]]) 
    PR = mne.concatenate_epochs([epochs["passive/relevance/blue"], epochs["passive/relevance/red"]]) 
    AI = mne.concatenate_epochs([epochs["active/irrelevance/blue"], epochs["active/irrelevance/red"]]) 
    PI = mne.concatenate_epochs([epochs["passive/irrelevance/blue"], epochs["passive/irrelevance/red"]]) 

    if REDUCE_NUMBER_OF_ELECTRODES and not VISUAL_CORTEX:
        n_features = f"Nested_Selected_{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}"

    # Update event IDs for each condition
    AR_updated = update_event_ids(AR)
    PR_updated = update_event_ids(PR)
    AI_updated = update_event_ids(AI)
    PI_updated = update_event_ids(PI)
    
    time_window_ms = TIME_WINDOW_MS 

    # Preprocess epochs for each condition
    X_AR, y_AR, time = preprocess_epochs(AR_updated, time_window_ms, sfreq)
    X_PR, y_PR, time = preprocess_epochs(PR_updated, time_window_ms, sfreq)
    X_AI, y_AI, time = preprocess_epochs(AI_updated, time_window_ms, sfreq)
    X_PI, y_PI, time = preprocess_epochs(PI_updated, time_window_ms, sfreq)
    
    for cond_name, labels in zip(["AR", "PR", "AI", "PI"], [y_AR, y_PR, y_AI, y_PI]):
        counts = Counter(labels)
        trial_summary.append({
            "Participant": SUBJ,
            "Condition": cond_name,
            "Total": len(labels),
            "Left": counts[1],
            "Right": counts[2]})
        
    """______________________________Temporal Decoding____________________________"""

    (scores_AR, scores_mean_AR, coef_AR_patterns, topomap_evoked_AR_patterns, coef_AR_filters, topomap_evoked_AR_filters, coef_AR_selections, topomap_evoked_AR_selections) = temporal_decoding(X_AR, y_AR, time_window_ms, CLASSIFIER, REDUCE_NUMBER_OF_ELECTRODES, AR_updated)
    (scores_PR, scores_mean_PR, coef_PR_patterns, topomap_evoked_PR_patterns, coef_PR_filters, topomap_evoked_PR_filters, coef_PR_selections, topomap_evoked_PR_selections) = temporal_decoding(X_PR, y_PR, time_window_ms, CLASSIFIER, REDUCE_NUMBER_OF_ELECTRODES, PR_updated)
    (scores_AI, scores_mean_AI, coef_AI_patterns, topomap_evoked_AI_patterns, coef_AI_filters, topomap_evoked_AI_filters, coef_AI_selections, topomap_evoked_AI_selections) = temporal_decoding(X_AI, y_AI, time_window_ms, CLASSIFIER, REDUCE_NUMBER_OF_ELECTRODES, AI_updated)
    (scores_PI, scores_mean_PI, coef_PI_patterns, topomap_evoked_PI_patterns, coef_PI_filters, topomap_evoked_PI_filters, coef_PI_selections, topomap_evoked_PI_selections) = temporal_decoding(X_PI, y_PI, time_window_ms, CLASSIFIER, REDUCE_NUMBER_OF_ELECTRODES, PI_updated)

    # For doing the grand average topomap we should add in a list
    list_of_topomap_evoked_AR_patterns.append(topomap_evoked_AR_patterns)
    list_of_topomap_evoked_PR_patterns.append(topomap_evoked_PR_patterns)
    list_of_topomap_evoked_AI_patterns.append(topomap_evoked_AI_patterns)
    list_of_topomap_evoked_PI_patterns.append(topomap_evoked_PI_patterns)
    
    list_of_topomap_evoked_AR_filters.append(topomap_evoked_AR_filters)
    list_of_topomap_evoked_PR_filters.append(topomap_evoked_PR_filters)
    list_of_topomap_evoked_AI_filters.append(topomap_evoked_AI_filters)
    list_of_topomap_evoked_PI_filters.append(topomap_evoked_PI_filters)
    
    list_of_topomap_evoked_AR_selections.append(topomap_evoked_AR_selections)
    list_of_topomap_evoked_PR_selections.append(topomap_evoked_PR_selections)
    list_of_topomap_evoked_AI_selections.append(topomap_evoked_AI_selections)
    list_of_topomap_evoked_PI_selections.append(topomap_evoked_PI_selections)

    list_of_coef_filters.append([coef_AR_filters, coef_PR_filters, coef_AI_filters, coef_PI_filters])
    list_of_coef_patterns.append([coef_AR_patterns, coef_PR_patterns, coef_AI_patterns, coef_PI_patterns])
    list_of_coef_selections.append([coef_AR_selections, coef_PR_selections, coef_AI_selections, coef_PI_selections])

    mean_scores_temporal_decoding.append([scores_mean_AR, scores_mean_PR, scores_mean_AI, scores_mean_PI])
    
    # Plot individual results
    plt.figure()
    plt.figure(figsize=(10, 5))

    plt.plot(time, scores_mean_AR, label='Active Relevance', color='red')
    plt.plot(time, scores_mean_PR, label='Passive Relevance', color='orange')
    plt.plot(time, scores_mean_AI, label='Active Irrelevance', color='blue')
    plt.plot(time, scores_mean_PI, label='Passive Irrelevance', color='cyan')
    plt.axhline(0.5, color="k", linestyle="--", label="chance")
    plt.xlabel("Time")
    plt.ylabel("AUC")
    plt.legend(loc='upper left')
    plt.title(f"Temporal Decoding - Participant {SUBJ}")
    plt.ylim(0.3, 0.9)
    plt.show()
    
    # Save the scores as numpy arrays
    np.save(str(outputs_dir) + f"\\sub-{SUBJ}_AR_temporal_decoding_scores.npy", scores_AR)
    np.save(str(outputs_dir) + f"\\sub-{SUBJ}_PR_temporal_decoding_scores.npy", scores_PR)
    np.save(str(outputs_dir) + f"\\sub-{SUBJ}_AI_temporal_decoding_scores.npy", scores_AI)
    np.save(str(outputs_dir) + f"\\sub-{SUBJ}_PI_temporal_decoding_scores.npy", scores_PI)

trial_summary = pd.DataFrame(trial_summary)
print(trial_summary)
print("\nMean number of trials")
print(trial_summary.groupby("Condition")[["Total", "Left", "Right"]].mean())
summary = trial_summary.groupby("Condition")[["Total", "Left", "Right"]].agg(["mean", "std"])
print(summary)

# Calculate mean scores
participant_mean_scores_temporal_decoding = np.mean(mean_scores_temporal_decoding, axis=0)

mean_matrices_per_condition = []
for condition_scores in zip(*mean_scores_temporal_generalization):
    condition_scores_array = np.array(condition_scores)
    mean_matrix = np.mean(condition_scores_array, axis=0)
    mean_matrices_per_condition.append(mean_matrix)

mean_scores_temporal_decoding_array = np.array(mean_scores_temporal_decoding)
std_error_temporal_decoding = (np.std(mean_scores_temporal_decoding_array, axis=0) / np.sqrt(mean_scores_temporal_decoding_array.shape[0]))


#%% ______________________________ Permutation tests __________________________
scores_active_relevant = mean_scores_temporal_decoding_array[:,0,:]
scores_passive_relevant = mean_scores_temporal_decoding_array[:,1,:]
scores_active_irrelevant = mean_scores_temporal_decoding_array[:,2,:]
scores_passive_irrelevant = mean_scores_temporal_decoding_array[:,3,:]

if TIME_WINDOW_MS:
    n = int(200/time_window_ms)
else:
    n = int(200*sfreq/1000)

scores_active_relevant_without_baseline = scores_active_relevant[:, n : scores_active_relevant.shape[1]] 
scores_passive_relevant_without_baseline = scores_passive_relevant[:, n : scores_passive_relevant.shape[1]]
scores_active_irrelevant_without_baseline = scores_active_irrelevant[:, n : scores_active_irrelevant.shape[1]]
scores_passive_irrelevant_without_baseline = scores_passive_irrelevant[:, n : scores_passive_irrelevant.shape[1]]

time_without_baseline = time[n:] 
mask_active_relevant_without_baseline, timesAR, clusters_AR, cluster_p_values_AR = statistical_tests(scores_active_relevant_without_baseline, time_without_baseline)
mask_passive_relevant_without_baseline, timesPR, clusters_PR, cluster_p_values_PR = statistical_tests(scores_passive_relevant_without_baseline, time_without_baseline)
mask_active_irrelevant_without_baseline, timesAI, clusters_AI, cluster_p_values_AI = statistical_tests(scores_active_irrelevant_without_baseline, time_without_baseline)
mask_passive_irrelevant_without_baseline, timesPI, clusters_PI, cluster_p_values_PI = statistical_tests(scores_passive_irrelevant_without_baseline, time_without_baseline)

zeros_boolean_array = np.full((n,1), 0) 
mask_active_relevant_without_baseline = mask_active_relevant_without_baseline.reshape(-1,1)
mask_passive_relevant_without_baseline = mask_passive_relevant_without_baseline.reshape(-1,1)
mask_active_irrelevant_without_baseline = mask_active_irrelevant_without_baseline.reshape(-1,1)
mask_passive_irrelevant_without_baseline = mask_passive_irrelevant_without_baseline.reshape(-1,1)

mask_active_relevant = np.vstack((zeros_boolean_array, mask_active_relevant_without_baseline))
mask_passive_relevant = np.vstack((zeros_boolean_array, mask_passive_relevant_without_baseline))
mask_active_irrelevant = np.vstack((zeros_boolean_array, mask_active_irrelevant_without_baseline))
mask_passive_irrelevant = np.vstack((zeros_boolean_array, mask_passive_irrelevant_without_baseline))

mask_active_vs_passive_relevant, timesAPR, clusters_AR, cluster_p_values_AR = statistical_comparaison_tests(scores_active_relevant, scores_passive_relevant, time)
mask_active_vs_passive_irrelevant, timesAPI, clusters_AR, cluster_p_values_AR = statistical_comparaison_tests(scores_active_irrelevant, scores_passive_irrelevant, time)
mask_relevant_vs_irrelevant_passive, timesRIP, clusters_AR, cluster_p_values_AR = statistical_comparaison_tests(scores_passive_relevant, scores_passive_irrelevant, time)
mask_relevant_vs_irrelevant_active, timesRIA, clusters_AR, cluster_p_values_AR = statistical_comparaison_tests(scores_active_relevant, scores_active_irrelevant, time)
mask_active_relevant_vs_passive_irrelevant, timesARPI, clusters_AR, cluster_p_values_AR = statistical_comparaison_tests(scores_active_relevant, scores_passive_irrelevant, time)


#%% __________________________________ Plots __________________________________
if REDUCE_NUMBER_OF_ELECTRODES == True: n_features = n_features 
else: n_features = 60
    
plot_temporal_decoding_conditions(
    [participant_mean_scores_temporal_decoding[0], participant_mean_scores_temporal_decoding[1], 
     participant_mean_scores_temporal_decoding[2], participant_mean_scores_temporal_decoding[3]],
    [std_error_temporal_decoding[0], std_error_temporal_decoding[1], 
     std_error_temporal_decoding[2], std_error_temporal_decoding[3], 
     ],
    ['Active Relevant', 'Passive Relevant', 'Active Irrelevant', 'Passive Irrelevant'],
    [timesAR, timesPR, timesAI, timesPI],
    ['red', 'darkorange', 'blue', 'darkturquoise'], 
    [mask_active_relevant, mask_passive_relevant, mask_active_irrelevant, mask_passive_irrelevant], time, n_features
)


#%% ____________________________ Condition comparaison ________________________
plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[0], label='Active Relevance', color='red')
plt.plot(time, participant_mean_scores_temporal_decoding[2], label='Active Irrelevance', color='blue')
plt.fill_between(time, participant_mean_scores_temporal_decoding[0] - std_error_temporal_decoding[0],
                  participant_mean_scores_temporal_decoding[0] + std_error_temporal_decoding[0], color='red', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[2] - std_error_temporal_decoding[2],
                  participant_mean_scores_temporal_decoding[2] + std_error_temporal_decoding[2], color='blue', alpha=0.3)
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')
plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.ylabel("AUC")
plt.legend(loc='upper left')
plt.title("Temporal Decoding - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, np.where(np.squeeze(mask_relevant_vs_irrelevant_active) == 1, 0.41, np.nan), linestyle='-', marker='o', markersize=4, color='black', linewidth=5, label='Significativité')
plt.show()

plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[0], label='Active Relevance', color='red')
plt.plot(time, participant_mean_scores_temporal_decoding[1], label='Passive Relevance', color='orange')
plt.fill_between(time, participant_mean_scores_temporal_decoding[0] - std_error_temporal_decoding[0],
                  participant_mean_scores_temporal_decoding[0] + std_error_temporal_decoding[0], color='red', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[1] - std_error_temporal_decoding[1],
                  participant_mean_scores_temporal_decoding[1] + std_error_temporal_decoding[1], color='orange', alpha=0.3)
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')
plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.ylabel("AUC")
plt.legend(loc='upper left')
plt.title("Temporal Decoding - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, np.where(np.squeeze(mask_active_vs_passive_relevant) == 1, 0.41, np.nan), linestyle='-', marker='o', markersize=4, color='black', linewidth=5, label='Significativité')
plt.show()

plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[2], label='Active Irrelevance', color='blue')
plt.plot(time, participant_mean_scores_temporal_decoding[3], label='Passive Irrelevance', color='darkturquoise')
plt.fill_between(time, participant_mean_scores_temporal_decoding[2] - std_error_temporal_decoding[2],
                  participant_mean_scores_temporal_decoding[2] + std_error_temporal_decoding[2], color='blue', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[3] - std_error_temporal_decoding[3],
                  participant_mean_scores_temporal_decoding[3] + std_error_temporal_decoding[3], color='darkturquoise', alpha=0.3)
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')
plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.ylabel("AUC")
plt.legend(loc='upper left')
plt.title("Temporal Decoding - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, np.where(np.squeeze(mask_active_vs_passive_irrelevant) == 1, 0.41, np.nan), linestyle='-', marker='o', markersize=4, color='black', linewidth=5, label='Significativité')
plt.show()

plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[1], label='Passive Relevance', color='orange')
plt.plot(time, participant_mean_scores_temporal_decoding[3], label='Passive Irrelevance', color='darkturquoise')
plt.fill_between(time, participant_mean_scores_temporal_decoding[1] - std_error_temporal_decoding[1],
                  participant_mean_scores_temporal_decoding[1] + std_error_temporal_decoding[1], color='orange', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[3] - std_error_temporal_decoding[3],
                  participant_mean_scores_temporal_decoding[3] + std_error_temporal_decoding[3], color='darkturquoise', alpha=0.3)
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')
plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.ylabel("AUC")
plt.legend(loc='upper left')
plt.title("Temporal Decoding - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, np.where(np.squeeze(mask_relevant_vs_irrelevant_passive) == 1, 0.41, np.nan), linestyle='-', marker='o', markersize=4, color='black', linewidth=5, label='Significativité')
plt.show()


#%% ___________________ Get Electrodes of strong coefficient __________________

top_n_electrodes = 0

if VISUAL_CORTEX == False:
    mean_coef_patterns = get_mean_coef_patterns_when_electrod_number_differs_per_condition(list_of_coef_patterns)
    mean_coef_filters = get_mean_coef_patterns_when_electrod_number_differs_per_condition(list_of_coef_filters)

else:
    mean_coef_patterns = np.mean(list_of_coef_patterns, axis=0) 
    mean_coef_filters = np.mean(list_of_coef_filters, axis=0) 

strong_electrods_per_condition_per_significant_time_point = get_strong_electrods_per_condition_per_significant_time_point(
    AR, PR, AI, PI,
    mean_coef_patterns,
    mask_active_relevant, mask_passive_relevant, mask_active_irrelevant, mask_passive_irrelevant, 
    VISUAL_CORTEX,
    n_electrodes=top_n_electrodes,
    time=time
)

print("\n Coef Patterns:")
for condition, data in strong_electrods_per_condition_per_significant_time_point.items(): 
    print(f"\n \n Pour la condition {condition}:")
    for index, ch_name_strong_electrodes in data.items():
        print(f"\nPour l'indice temporel {index}, les {top_n_electrodes} premières électrodes dans l'ordre des coefficients les plus différents de 0 sont : {ch_name_strong_electrodes}")


#%% _________________________________ Topomaps ________________________________

"""_________________________TOPO MAP WITH MASK STRONG ELECTRODS______________________"""
grand_average_AR_patterns = mne.grand_average(list_of_topomap_evoked_AR_patterns)
grand_average_PR_patterns = mne.grand_average(list_of_topomap_evoked_PR_patterns)
grand_average_AI_patterns = mne.grand_average(list_of_topomap_evoked_AI_patterns)
grand_average_PI_patterns = mne.grand_average(list_of_topomap_evoked_PI_patterns)

conditions = {"AR": mask_active_relevant, 
              "PR": mask_passive_relevant, 
              "AI": mask_active_irrelevant, 
              "PI": mask_passive_irrelevant}

if TIME_WINDOW_MS:
    times = np.arange(0.0, 0.8, TIME_WINDOW_MS / 1000.0)
else:
    times = np.arange(0.0, 0.8, 0.1)

"""________________________Counts significant clusters_________________________ """
n_significant_clusters=0
liste_clusters_conditions = [mask_active_relevant, mask_passive_relevant,
                             mask_active_irrelevant, mask_passive_irrelevant]
for l in liste_clusters_conditions:  
    for i in range(0,len(l)):
        if l[i] == True:
            n_significant_clusters += 1
    

#%% _________________________________ Topomaps ________________________________

# --- COEF PATTERNS GRAND AVERAGE ---
grand_average_AR_patterns = mne.grand_average(list_of_topomap_evoked_AR_patterns)
grand_average_PR_patterns = mne.grand_average(list_of_topomap_evoked_PR_patterns)
grand_average_AI_patterns = mne.grand_average(list_of_topomap_evoked_AI_patterns)
grand_average_PI_patterns = mne.grand_average(list_of_topomap_evoked_PI_patterns)

all_topomap_data = np.concatenate([
    grand_average_AR_patterns.data,
    grand_average_PR_patterns.data,
    grand_average_AI_patterns.data,
    grand_average_PI_patterns.data
])

GLOBAL_VBOUND = np.percentile(np.abs(all_topomap_data), 99)
if GLOBAL_VBOUND == 0:  
    GLOBAL_VBOUND = np.max(np.abs(all_topomap_data)) or 1.0

if TIME_WINDOW_MS == 100: 
    GRID_FONT_SIZE_SUPTITLE   = 40  
    GRID_FONT_SIZE_ROW_LABEL  = 35  
    GRID_FONT_SIZE_CBAR_LABEL = 35  
    GRID_FONT_SIZE_CBAR_TICK  = 30 
    dpi = 600
    suptitle_pad = 0.98
    subplot_height = 14
elif TIME_WINDOW_MS == 50 or 20:
    GRID_FONT_SIZE_SUPTITLE   = 70  
    GRID_FONT_SIZE_ROW_LABEL  = 50  
    GRID_FONT_SIZE_CBAR_LABEL = 50  
    GRID_FONT_SIZE_CBAR_TICK  = 45
    dpi = 300
    suptitle_pad = 1
    subplot_height = 16

GRID_FONT_SIZE_COL_TITLE  = 30  

GRID_PAD_COL_TITLE = 15   
GRID_PAD_ROW_LABEL = 0.40 
GRID_WSPACE        = 0.10 
GRID_HSPACE        = 0.08 

if TIME_WINDOW_MS:
    times = np.arange(0.0, 0.8, TIME_WINDOW_MS / 1000.0)
else:
    times = np.arange(0.0, 0.8, 0.1)


def plot_all_topographies_grid():
    n_cols = len(times)
    n_rows = 4  
    
    GLOBAL_VBOUND = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(28, n_cols * 3.5), subplot_height))

    rows_config = [
        {"key": "AR", "data": grand_average_AR_patterns, "label": "Active\nRelevant"},
        {"key": "PR", "data": grand_average_PR_patterns, "label": "Passive\nRelevant"},
        {"key": "AI", "data": grand_average_AI_patterns, "label": "Active\nIrrelevant"},
        {"key": "PI", "data": grand_average_PI_patterns, "label": "Passive\nIrrelevant"}
    ]
    
    im_ref = None
    mask_topo_params = dict(marker='o', markerfacecolor='black', markeredgecolor='white', markersize=10, markeredgewidth=2)
    
    for row_idx, cfg in enumerate(rows_config):
        cond_key = cfg["key"]
        evoked = cfg["data"]
        
        for col_idx, t_val in enumerate(times):
            ax = axes[row_idx, col_idx] if (n_rows > 1 and n_cols > 1) else axes[max(row_idx, col_idx)]
            
            idx_time_point = np.argmin(np.abs(evoked.times - t_val))
            topo_data = evoked.data[:, idx_time_point] * 1e6
            
            mask_topo = np.zeros(len(evoked.ch_names), dtype=bool)
            if idx_time_point in strong_electrods_per_condition_per_significant_time_point[cond_key]:
                top_n_electrodes = strong_electrods_per_condition_per_significant_time_point[cond_key][idx_time_point]
                for electrode in top_n_electrodes:
                    if electrode in evoked.ch_names:
                        idx_electrode = evoked.ch_names.index(electrode)
                        mask_topo[idx_electrode] = True
            
            im, _ = mne.viz.plot_topomap(
                topo_data, evoked.info, axes=ax, show=False,
                mask=mask_topo, mask_params=mask_topo_params,
                cmap="RdBu_r", vlim=(-GLOBAL_VBOUND, GLOBAL_VBOUND), contours=6
            )
            im_ref = im
            
            if row_idx == 0:
                ax.set_title(
                    f"{int(np.round(t_val * 1000))} ms", 
                    fontsize=GRID_FONT_SIZE_COL_TITLE, 
                    fontweight="bold", 
                    pad=GRID_PAD_COL_TITLE
                )
                
            if col_idx == 0:
                ax.text(
                    -GRID_PAD_ROW_LABEL, 0.5, cfg["label"], 
                    transform=ax.transAxes,
                    fontsize=GRID_FONT_SIZE_ROW_LABEL, 
                    # fontweight="bold",
                    va="center", ha="right"
                )

    plt.suptitle(
        "Reconstructed Spatial Patterns", 
        fontsize=GRID_FONT_SIZE_SUPTITLE, 
        # fontweight="bold",
        y=suptitle_pad
    )
    
    plt.subplots_adjust(
        left=0.20, 
        right=0.88, 
        bottom=0.05, 
        top=0.88, 
        wspace=GRID_WSPACE, 
        hspace=GRID_HSPACE
    )
    
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    clb = fig.colorbar(im_ref, cax=cbar_ax)
    clb.set_label("Spatial Patterns (µV)", fontsize=GRID_FONT_SIZE_CBAR_LABEL, labelpad=15)
    clb.ax.tick_params(labelsize=GRID_FONT_SIZE_CBAR_TICK)
    
    plot_file_name = plot_dir / "NFS_per_time_window" / f"{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_electrods" / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_Topomaps_Grid.jpg"
    plt.savefig(plot_file_name, dpi=dpi, bbox_inches="tight")
    plt.show()

# Exécuter
plot_all_topographies_grid()

#%%
# =============================================================================
# NOUVEAU BLOCK : SELECTION FREQUENCY TOPOMAPS (GRILLE SUBPLOT)
# =============================================================================
grand_average_AR_selections = mne.grand_average(list_of_topomap_evoked_AR_selections)
grand_average_PR_selections = mne.grand_average(list_of_topomap_evoked_PR_selections)
grand_average_AI_selections = mne.grand_average(list_of_topomap_evoked_AI_selections)
grand_average_PI_selections = mne.grand_average(list_of_topomap_evoked_PI_selections)

def plot_selection_frequency_grid():
    n_cols = len(times)
    n_rows = 4  # 4 Conditions (AR, PR, AI, PI)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(28, n_cols * 3.5), subplot_height))

    rows_config = [
        {"key": "AR", "data": grand_average_AR_selections, "label": "Active\nRelevant"},
        {"key": "PR", "data": grand_average_PR_selections, "label": "Passive\nRelevant"},
        {"key": "AI", "data": grand_average_AI_selections, "label": "Active\nIrrelevant"},
        {"key": "PI", "data": grand_average_PI_selections, "label": "Passive\nIrrelevant"}
    ]
    
    im_ref = None
    
    for row_idx, cfg in enumerate(rows_config):
        evoked = cfg["data"]
        
        for col_idx, t_val in enumerate(times):
            ax = axes[row_idx, col_idx] if (n_rows > 1 and n_cols > 1) else axes[max(row_idx, col_idx)]
            
            idx_time_point = np.argmin(np.abs(evoked.times - t_val))
            
            topo_data = evoked.data[:, idx_time_point] * 1e6
            
            im, _ = mne.viz.plot_topomap(
                topo_data, evoked.info, axes=ax, show=False,
                cmap="Reds", vlim=(30.0, 100.0), contours=5
            )
            im_ref = im
            
            if row_idx == 0:
                ax.set_title(
                    f"{int(np.round(t_val * 1000))} ms", 
                    fontsize=GRID_FONT_SIZE_COL_TITLE, 
                    # fontweight="bold", 
                    pad=GRID_PAD_COL_TITLE
                )
                
            if col_idx == 0:
                ax.text(
                    -GRID_PAD_ROW_LABEL, 0.5, cfg["label"], 
                    transform=ax.transAxes,
                    fontsize=GRID_FONT_SIZE_ROW_LABEL, 
                    # fontweight="bold",
                    va="center", ha="right"
                )

    plt.suptitle(
        "Electrode Selection Frequency", 
        fontsize=GRID_FONT_SIZE_SUPTITLE, 
        # fontweight="bold",
        y=suptitle_pad
    )
    
    plt.subplots_adjust(
        left=0.20, 
        right=0.88, 
        bottom=0.05, 
        top=0.88, 
        wspace=GRID_WSPACE, 
        hspace=GRID_HSPACE
    )
    
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    clb = fig.colorbar(im_ref, cax=cbar_ax)
    clb.set_label("Selection Frequency (%)", fontsize=GRID_FONT_SIZE_CBAR_LABEL, labelpad=15)
    clb.ax.tick_params(labelsize=GRID_FONT_SIZE_CBAR_TICK)
    
    plot_file_name = plot_dir / "NFS_per_time_window" / f"{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_electrods" / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_Selection_Frequency_Grid.jpg"
    plt.savefig(plot_file_name, dpi=dpi, bbox_inches="tight")
    plt.show()

plot_selection_frequency_grid()

#%%
# =============================================================================
# THREE GRAPHS in ONE SINGLE FIGURE
# =============================================================================

path_temporal_decoding = plot_dir / "NFS_per_time_window" / f"{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_electrods" / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_1.jpg"
path_spatial_patterns = plot_dir / "NFS_per_time_window" / f"{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_electrods" / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_Topomaps_Grid.jpg"
path_selection_freq = plot_dir / "NFS_per_time_window" / f"{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_electrods" / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_Selection_Frequency_Grid.jpg"

if N_CHANNELS_PER_TIME_POINT_PER_CONDITION != 60: 
    images_to_merge = [
        path_selection_freq,
        path_temporal_decoding,
        path_spatial_patterns,
    ]
else:
    images_to_merge = [
        # path_selection_freq,
        path_temporal_decoding,
        path_spatial_patterns,
    ]

path_combined_figure = plot_dir / "NFS_per_time_window" / f"{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_electrods" / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_COMBINED_DASHBOARD.jpg"

try:
    merge_plots_vertically(images_to_merge, path_combined_figure, spacing=250)
except FileNotFoundError as e:
    print(f"Erreur lors de la fusion : une des images n'a pas été trouvée. Détails : {e}")


#%% ______________ Save the classification performance & parameters ___________
if TIME_WINDOW_MS:
    n = int(200/time_window_ms)
else:
    n = int(200*sfreq/1000)

A_mean_scores_dict = {}
for idx_condition, condition in enumerate(["AR", "PR", "AI", "PI"]):
    
    for idx_time, _ in enumerate(mean_scores_temporal_decoding[0][idx_condition]): 
        if liste_clusters_conditions[idx_condition][idx_time] == 1:
        
            mean_score_condition = np.mean(mean_scores_temporal_decoding_array[:, idx_condition, idx_time])
                
            classifier_used = CLASSIFIER
            time_window_ms_used = TIME_WINDOW_MS
            reduce_number_of_electrodes = REDUCE_NUMBER_OF_ELECTRODES
            
            A_mean_scores_dict[condition] = {
                "mean_score": mean_score_condition,
                "classifier_used": classifier_used,
                "time_window_ms_used": time_window_ms_used,
                "reduce_number_of_electrodes": reduce_number_of_electrodes,
                "n_features": n_features,
                "n_significant_clusters": n_significant_clusters,
            }

clusters_final_mask = [1 if any(mask) else 0 for mask in zip(*liste_clusters_conditions)]

for idx_time, _ in enumerate(mean_scores_temporal_decoding[0][0]): 
    if clusters_final_mask[idx_time] == 1:
        overall_mean_score  = np.mean(mean_scores_temporal_decoding_array[:,:,idx_time])

        classifier_used = CLASSIFIER
        time_window_ms_used = TIME_WINDOW_MS
        reduce_number_of_electrodes = REDUCE_NUMBER_OF_ELECTRODES
        
        A_overall_mean_scores_dict  = {
            "overall_mean_score": overall_mean_score,
            "classifier_used": classifier_used,
            "time_window_ms_used": time_window_ms_used,
            "reduce_number_of_electrodes": reduce_number_of_electrodes,
            "n_features": n_features,
            "n_significant_clusters": n_significant_clusters,
        }

A_overall_mean_scores_df = pd.DataFrame([A_overall_mean_scores_dict])

file_name = "overall_mean_scores.xlsx"
file_path = os.path.join(outputs_dir, file_name)
if not os.path.exists(file_path): 
    print("coucou")
    A_overall_mean_scores_df.to_excel(file_path, header=True, index=False) 
    
else:
    print("The file already exists.")
    df_classif_perf = pd.read_excel(file_path) 
    
    is_already_present = df_classif_perf[df_classif_perf.eq(A_overall_mean_scores_df).all(axis=1)]
    if is_already_present.empty:
        df_classif_perf = df_classif_perf.append(A_overall_mean_scores_df, ignore_index=True) 
    else:   
        print("This classification already exists on the dataframe.")    
    
    df_classif_perf.to_excel(file_path, index=False)
    
#%% _______________________ TEXT EXPORT: CLUSTER TIMES ________________________

def extract_cluster_boundaries(mask, time_array, condition_name):
    mask_flat = np.squeeze(mask)
    
    significant_indices = np.where(mask_flat == 1)[0]
    
    if len(significant_indices) == 0:
        print(f"{condition_name} : Aucun cluster significatif détecté.")
        return
    
    clusters = np.split(significant_indices, np.where(np.diff(significant_indices) != 1)[0] + 1)
    
    print(f"\n--- {condition_name} ---")
    for i, cluster in enumerate(clusters):
        start_time = time_array[cluster[0]] * 1000  # Conversion en ms
        end_time = time_array[cluster[-1]] * 1000   # Conversion en ms
        print(f"Cluster {i+1} : de {start_time:.0f} ms à {end_time:.0f} ms")

print("\n=== BORNES TEMPORELLES DES CLUSTERS (en ms) ===")
extract_cluster_boundaries(mask_active_relevant, time, "Active Relevant (AR)")
extract_cluster_boundaries(mask_passive_relevant, time, "Passive Relevant (PR)")
extract_cluster_boundaries(mask_active_irrelevant, time, "Active Irrelevant (AI)")
extract_cluster_boundaries(mask_passive_irrelevant, time, "Passive Irrelevant (PI)")


