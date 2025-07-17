# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:53:04 2024

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
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
from mne.stats import fdr_correction, permutation_t_test

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from mne import EvokedArray
from skfeature.function.similarity_based import reliefF
import pandas as pd
from pathlib import Path

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
bids_derivatives_dir = study_path / "Data" / "eeg" / "derivatives" / "preproc"

# Path of the inputs for the electrodes selection
inputs_dir = study_path / "Code" / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

outputs_dir = study_path / "Code" / "outputs"


#%% ________________________________ CONSTANTS ________________________________

# -----------------------------------------------------------------------------
# ---------------------------- TIME WINDOWS SELECTION -------------------------
# -----------------------------------------------------------------------------
TIME_WINDOW_MS = 100 # you can test with time windows of 20 and 100 according to the paper parameters of the Supplementary Material
# you can also mark None (or 2ms for 500hz sampling) if you don't want time windows

# -----------------------------------------------------------------------------
# ----------------------------- ELECTRODES SELECTION --------------------------
# -----------------------------------------------------------------------------
REDUCE_NUMBER_OF_ELECTRODES = False # False if you don't want to reduce the electrodes (60 channels)

# If you want to reduce the electrodes number (n features) define the number below
N_CHANNELS_PER_TIME_POINT_PER_CONDITION  = None # number of electrodes which are then pooled for each condition (choose 5, 10 or 15) 
# OR take only electrodes from visual cortex 
VISUAL_CORTEX = False # True if you want that

# -----------------------------------------------------------------------------
# ----------------------------- ALGORITHM SELECTION ---------------------------
# -----------------------------------------------------------------------------
CLASSIFIER = "LDA" # you can test an other algorithm with "LR", "SVM", "RF" 


#%% ________________________________ FUNCTIONS ________________________________

def reduce_temporal_resolution(data, time_window_ms, sfreq):
    # # Calculate the number of samples to average for each 10 ms window
    # samples_per_window = 5
    
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
        time = np.arange(-.2, 0.8, time_window_ms/1000) # SOL 2 : epochs.times
    else:
        time = epochs.times
        
    y = epochs.events[:, 2]

    return x, y, time
    
def update_event_ids(epochs):
    # Create a copy of epochs to avoid modifications to the original data
    epochs_updated = epochs.copy()

    # Iterate through all events in epochs
    for i in range(len(epochs_updated.events)):
        # If the event ID is even, update it to 2
        if epochs_updated.events[i, 2] % 2 == 0:
            epochs_updated.events[i, 2] = 2
        # Otherwise, update it to 1
        else:
            epochs_updated.events[i, 2] = 1

    return epochs_updated


def temporal_decoding(X, y, window_size, classifier, REDUCE_NUMBER_OF_ELECTRODES, epochs):    
    
    # Setup of the classifier pipeline
    # Training on succesive temporal windows
    if classifier == "LDA":
        clf = make_pipeline(StandardScaler(), LinearModel(LinearDiscriminantAnalysis()))
        map_model = make_pipeline(Vectorizer(), StandardScaler(), LinearModel(LinearDiscriminantAnalysis()))
    
    elif classifier == "LR":
        clf = make_pipeline(StandardScaler(),
                            LinearModel(LogisticRegression(solver="liblinear")))
        map_model = make_pipeline(Vectorizer(), StandardScaler(), LinearModel(LogisticRegression(solver="liblinear")))         
    
    elif classifier == "SVM":
        clf = make_pipeline(StandardScaler(),
                            LinearModel(SVC(kernel='linear')))
        map_model = make_pipeline(Vectorizer(), StandardScaler(), LinearModel(SVC(kernel='linear')))

    elif classifier == "RF":
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=9, n_estimators=260))
        map_model = make_pipeline(Vectorizer(), StandardScaler(), LinearModel(LogisticRegression(solver="liblinear")))

    # Sliding estimator for temporal decoding
    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring="roc_auc", verbose=True)
    # Cross-validation for temporal decoding : scores array dimension (n_cv_score, n_temporal_point)
    scores = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=-1) #Utilisation de cv=10 pour plus de précision ; n_jobs=-1, it is set to the number of CPU cores
    # Average scores across cross-validation splits
    scores_mean = np.mean(scores, axis=0)
    
    # TOPO MAP INFO
    map_model.fit(X, y)
    
    coef_patterns = get_coef(map_model, 'patterns_', inverse_transform=True)
    coef_filters = get_coef(map_model, 'filters_', inverse_transform=True)

    if TIME_WINDOW_MS:
        sample_rate_updated = (1000/window_size)   
        epoch_updated = epochs.copy().resample(sample_rate_updated)
    else:
        sample_rate_updated = 500
        epoch_updated = epochs.copy()

    topomap_evoked_patterns = EvokedArray(coef_patterns, epoch_updated.info, tmin=epochs.tmin)  
    topomap_evoked_filters = EvokedArray(coef_filters, epoch_updated.info, tmin=epochs.tmin)  
    
    return scores, scores_mean, coef_patterns, topomap_evoked_patterns, coef_filters, topomap_evoked_filters
    
def statistical_tests(scores, time, alpha=0.05):
    # Cluster permutation test : scores (n_participants, n_time_points)
    # The first dimension should correspond to the difference between paired samples (observations) 
    # in two conditions : here the first condition is the real classification score 
    # and the second condition is the chance classification score (0.5)
    # For the second dimension, the function try to find time points clusters that will have permuted 
    # their score difference (Randomized data are generated with random sign flips)  
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(scores - 0.5, n_permutations=10000)
    mask_permutation = np.zeros_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= alpha:
            mask_permutation[c] = 1
    
    # Univariate t-test - Calculate the T-test for the mean of ONE group of scores
    # expected value in null hypothesis : 0.5
    T, pval = stats.ttest_1samp(scores, 0.5)
    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='negcorr')
    significant_times = np.unique(time)[pval_fdr < alpha]
  
    return mask_permutation, significant_times, clusters, cluster_p_values


def statistical_comparaison_tests(scores_conditionA, scores_conditionB, time, alpha=0.05):
    # Permutation Test
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
    plt.figure(figsize=(16, 10))  # Adjust figure size according to the number of subplots
    for i, (scores_mean, std_error, title, significance_times, color, mask_condition) in enumerate(zip(scores_mean_list, std_error_list, titles,                                                                                                       significance_times_list, colors, mask_condition_list), start=1):
        plt.subplot(2, 2, i)  # Create the corresponding subplot
        ax = sns.lineplot(x=time, y=scores_mean, label=title, color=color)
        ax.fill_between(time, scores_mean - std_error, scores_mean + std_error, color=color, alpha=0.3)
        ax.axhline(.5, color='k', linestyle='--', label="chance")
        ax.set_xlabel('Time (ms)', fontsize=24)
        ax.set_ylabel('AUC', fontsize=24)
        ax.axvline(0, color='k', linestyle='dotted')
        ax.set_xlim([-.2,.8])
        ax.set_ylim([0.45,.61])
        ax.legend(loc='upper left', fontsize=24, frameon=False)
        ax.set_xticklabels((ax.get_xticks() * 1000).astype(int)) # convert to milliseconds
        # plt.plot(significance_times, significance_times*0 + 0.49, 'o', color=color) # t-test
        plt.plot(time, mask_condition*0.48, 'o', color=color, markersize=1) # permutation test
    
        # plt.plot(time, mask_condition*0.46, 'o', color=color) # permutation test
        ax.tick_params(axis='x', labelsize=24)  # Increase x-axis ticks size
        ax.tick_params(axis='y', labelsize=24)  # Increase y-axis ticks size
    
    plt.tight_layout()  # Automatically adjust spacing between subplots
    plot_file_name = plot_dir / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_1.jpg"
    plt.savefig(plot_file_name, dpi=300, bbox_inches='tight')
    plt.show()


def find_significant_electrods_from_coefficient_classification(epochs_AR, epochs_PR, epochs_AI, epochs_PI, 
                                                               mask_AR, mask_PR, mask_AI, mask_PI,
                                                               AR_3d_array, PR_3d_array, AI_3d_array, PI_3d_array):
    alpha = 0.05
    n_permutations = 10000    
    coef_patterns_array_per_time_point = {}
    ch_name_significant_electrods_per_condition_per_significant_time_point = {}
  
    conditions = {
        "AR": {"mask": mask_AR, "coef": AR_3d_array, "epochs": epochs_AR},
        "PR": {"mask": mask_PR, "coef": PR_3d_array, "epochs": epochs_PR},
        "AI": {"mask": mask_AI, "coef": AI_3d_array, "epochs": epochs_AI},
        "PI": {"mask": mask_PI, "coef": PI_3d_array, "epochs": epochs_PI}
    }    
    
    for condition, data in conditions.items():        
        mask = data["mask"]
        coef = data["coef"]
        epochs = data["epochs"]
        
        electrode_names = epochs.ch_names    # Get electrode names from MNE data
    
        # Create a dictionary to store sorted electrode names for each time point
        ch_name_significant_electrodes_per_time_point = {}
        
        # Iterate over each index in the mask
        for time_point_index in range(mask.shape[0]):
            # Check if the value in the mask at this index is 1
            if mask[time_point_index] == 1:
                # Extract the values from AR_array for this index in the third dimension
                indexed_values = coef[:, :, time_point_index]
                # Store these values in a new array of shape (24, 60)
                coef_patterns_array_per_time_point[f'{time_point_index}'] = indexed_values

                # Example with the time_point number 6 and the condition 'AR' so we take "AR_array_index_6"
                X = coef_patterns_array_per_time_point[f'{time_point_index}']
                # -------------- Permutation test ---------------
                _, p_values, _ = permutation_t_test(X, n_permutations=n_permutations, n_jobs=1)
                          
                # Find indices where p-value < 0.05
                significant_indices = np.where(p_values < alpha)[0]
                # Display significant indices
                print(f"\nIndices with p-value < 0.05 for condition {condition}"
                      f"\n and time point {time_point_index} are :", significant_indices)
                
                ch_name_significant_electrodes = [electrode_names[i] for i in significant_indices]
                ch_name_significant_electrodes_per_time_point[time_point_index] = ch_name_significant_electrodes
                # Display corresponding electrodes
                print("Corresponding electrodes :", ch_name_significant_electrodes)
            
        # Store results for this condition
        ch_name_significant_electrods_per_condition_per_significant_time_point[condition] = ch_name_significant_electrodes_per_time_point

    return ch_name_significant_electrods_per_condition_per_significant_time_point
        

def get_strong_electrods_per_condition_per_significant_time_point(epochs_AR, epochs_PR, epochs_AI, epochs_PI,
                                                                  mean_coef, 
                                                                  mask_AR, mask_PR, mask_AI, mask_PI,
                                                                  VISUAL_CORTEX,
                                                                  n_electrodes=10):
    # mean_coef (4, n_electrods, n_time_points)
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

    # Define masks and coefficients for each condition
    conditions = {
        "AR": {"mask": mask_AR, "coef": coef_AR, "epochs": epochs_AR},
        "PR": {"mask": mask_PR, "coef": coef_PR, "epochs": epochs_PR},
        "AI": {"mask": mask_AI, "coef": coef_AI, "epochs": epochs_AI},
        "PI": {"mask": mask_PI, "coef": coef_PI, "epochs": epochs_PI}
    }

    # Create a dictionary to store the important electrodes for each condition
    ch_name_strong_electrods_per_condition_per_significant_time_point = {}

    # For each condition
    for condition, data in conditions.items():
        mask = data["mask"]
        coef = data["coef"]
        epochs = data["epochs"]
        
        # Get electrode names from MNE data
        electrode_names = epochs.ch_names

        # Create a dictionary to store sorted electrode names for each time point
        ch_name_strong_electrodes_per_time_point = {}

        # Find indices where the value in the "mask_" variable equals 1
        indices_to_consider = np.where(mask == 1)[0]

        # For each time index
        for index in indices_to_consider:
            # Retrieve coefficients for this time index and condition
            coefficients_at_index = coef[:, index]
            # Sort electrode indices by the absolute value of the coefficients
            index_strong_electrodes = np.argsort(np.abs(coefficients_at_index))[::-1][:n_electrodes]
            # Get the corresponding electrode names
            ch_name_strong_electrodes = [electrode_names[i] for i in index_strong_electrodes]
            ch_name_strong_electrodes_per_time_point[index] = ch_name_strong_electrodes

        # Store results for this condition
        ch_name_strong_electrods_per_condition_per_significant_time_point[condition] = ch_name_strong_electrodes_per_time_point

    return ch_name_strong_electrods_per_condition_per_significant_time_point
 

def plot_topomap_with_mask(grand_average, strong_electrods_per_condition_per_significant_time_point, key_condition, mask_condition, times, coef_name, vlim=(-6, 6)):
    
    # Get the indices to consider based on the mask of the given condition
    indices_to_consider = np.where(mask_condition == 1)[0]
    
    if indices_to_consider.size == 0:
        mask_topo=None
        mask_topo_params=None
    
    else:
        # Mask initialization
        mask_topo = np.zeros(grand_average.data.shape, dtype=bool)
        mask_topo_params = dict(marker='o', markerfacecolor='black', markeredgecolor='white', markersize=13, markeredgewidth=2)
        
        for idx_time_point in indices_to_consider:
            # Get the electrode names for this time point index
            top_n_electrodes = strong_electrods_per_condition_per_significant_time_point[key_condition][idx_time_point];
            
            # Updating the mask for these electrodes to this time point index
            for electrode in top_n_electrodes:
                idx_electrode = grand_average.ch_names.index(electrode)
                mask_topo[idx_electrode, idx_time_point] = True
    
    # Display of topomap with white circles on selected electrodes
    fig = grand_average.plot_topomap(times=times, vlim=vlim,
                                mask=mask_topo, mask_params=mask_topo_params,
                                show=False, colorbar=True);
    
    fig.suptitle(f"EEG {coef_name} condition {key_condition}")


#Function to calculate the mean of arrays of different sizes
def calculate_mean(condition_data, time_window=TIME_WINDOW_MS):
    max_rows = max(table.shape[0] for table in condition_data)
    
    if time_window:
        print("coucou")
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
    
    # List initialization for each condition
    condition1 = []
    condition2 = []
    condition3 = []
    condition4 = []

    # Accumulate arrays for each condition
    for participant in list_of_coef:
        condition1.append(participant[0])
        condition2.append(participant[1])
        condition3.append(participant[2])
        condition4.append(participant[3])

    # Calculate means for each condition
    mean_condition1 = calculate_mean(condition1)
    mean_condition2 = calculate_mean(condition2)
    mean_condition3 = calculate_mean(condition3)
    mean_condition4 = calculate_mean(condition4)

    # Group results into a list or other structure if necessary
    mean_results = [mean_condition1, mean_condition2, mean_condition3, mean_condition4]
    
    return mean_results


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
list_of_coef_patterns=[]
list_of_coef_filters=[]

# Loop for each participant
for PARTICIPANT_NUM in range(1, 25):  # Assuming participants are numbered from P001 to P024
    
    if PARTICIPANT_NUM < 10:
        SUBJ = f'0{PARTICIPANT_NUM}'
    else:
        SUBJ = f'{PARTICIPANT_NUM}'   
    
    bids_subj = f"sub-{SUBJ}"
    # To open the eeg data file previously saved
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

        
    event_ids = epochs.event_id # vérifier le numéro des conditions (si c'est bien right pair et left impair)
    
    # Concatenate epochs for each condition
    AR = mne.concatenate_epochs([epochs["active/relevance/blue"], epochs["active/relevance/red"]]) # condition active relevance
    PR = mne.concatenate_epochs([epochs["passive/relevance/blue"], epochs["passive/relevance/red"]]) # condition passive relevance
    AI = mne.concatenate_epochs([epochs["active/irrelevance/blue"], epochs["active/irrelevance/red"]]) # condition active irrelevance
    PI = mne.concatenate_epochs([epochs["passive/irrelevance/blue"], epochs["passive/irrelevance/red"]]) # condition passive irrelevance

    if REDUCE_NUMBER_OF_ELECTRODES and VISUAL_CORTEX == False:
        # AR
        electrodes_of_interest_AR = np.load(str(inputs_dir) + f'\\{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_{TIME_WINDOW_MS}ms_channels_features_AR.npy')
        AR.pick_channels(electrodes_of_interest_AR)
        # PR
        electrodes_of_interest_PR = np.load(str(inputs_dir) + f'\\{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_{TIME_WINDOW_MS}ms_channels_features_PR.npy')
        PR.pick_channels(electrodes_of_interest_PR)
        # AI
        electrodes_of_interest_AI = np.load(str(inputs_dir) + f'\\{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_{TIME_WINDOW_MS}ms_channels_features_AI.npy')
        AI.pick_channels(electrodes_of_interest_AI)
        # PI
        electrodes_of_interest_PI = np.load(str(inputs_dir) + f'\\{N_CHANNELS_PER_TIME_POINT_PER_CONDITION}_{TIME_WINDOW_MS}ms_channels_features_PI.npy')
        PI.pick_channels(electrodes_of_interest_PI)
        
        n_features = N_CHANNELS_PER_TIME_POINT_PER_CONDITION


    # Update event IDs for each condition
    AR_updated = update_event_ids(AR)
    PR_updated = update_event_ids(PR)
    AI_updated = update_event_ids(AI)
    PI_updated = update_event_ids(PI)
    
    time_window_ms = TIME_WINDOW_MS # specify the time window in milliseconds

    # Preprocess epochs for each condition
    X_AR, y_AR, time = preprocess_epochs(AR_updated, time_window_ms, sfreq)
    X_PR, y_PR, time = preprocess_epochs(PR_updated, time_window_ms, sfreq)
    X_AI, y_AI, time = preprocess_epochs(AI_updated, time_window_ms, sfreq)
    X_PI, y_PI, time = preprocess_epochs(PI_updated, time_window_ms, sfreq)
       
    """______________________________Temporal Decoding____________________________"""

    # Train models and get feature importance for each condition
    scores_AR, scores_mean_AR, coef_AR_patterns, topomap_evoked_AR_patterns, coef_AR_filters, topomap_evoked_AR_filters = temporal_decoding(X_AR, y_AR, time_window_ms, CLASSIFIER, REDUCE_NUMBER_OF_ELECTRODES, AR_updated)
    scores_PR, scores_mean_PR, coef_PR_patterns, topomap_evoked_PR_patterns, coef_PR_filters, topomap_evoked_PR_filters = temporal_decoding(X_PR, y_PR, time_window_ms, CLASSIFIER, REDUCE_NUMBER_OF_ELECTRODES, PR_updated)
    scores_AI, scores_mean_AI, coef_AI_patterns, topomap_evoked_AI_patterns, coef_AI_filters, topomap_evoked_AI_filters = temporal_decoding(X_AI, y_AI, time_window_ms, CLASSIFIER, REDUCE_NUMBER_OF_ELECTRODES, AI_updated)
    scores_PI, scores_mean_PI, coef_PI_patterns, topomap_evoked_PI_patterns, coef_PI_filters, topomap_evoked_PI_filters = temporal_decoding(X_PI, y_PI, time_window_ms, CLASSIFIER, REDUCE_NUMBER_OF_ELECTRODES, PI_updated)    
    
    # For doing the grand average topomap we should add in a list
    list_of_topomap_evoked_AR_patterns.append(topomap_evoked_AR_patterns)
    list_of_topomap_evoked_PR_patterns.append(topomap_evoked_PR_patterns)
    list_of_topomap_evoked_AI_patterns.append(topomap_evoked_AI_patterns)
    list_of_topomap_evoked_PI_patterns.append(topomap_evoked_PI_patterns)
    
    list_of_topomap_evoked_AR_filters.append(topomap_evoked_AR_filters)
    list_of_topomap_evoked_PR_filters.append(topomap_evoked_PR_filters)
    list_of_topomap_evoked_AI_filters.append(topomap_evoked_AI_filters)
    list_of_topomap_evoked_PI_filters.append(topomap_evoked_PI_filters)
    
    list_of_coef_filters.append([coef_AR_filters, coef_PR_filters, coef_AI_filters, coef_PI_filters])
    list_of_coef_patterns.append([coef_AR_patterns, coef_PR_patterns, coef_AI_patterns, coef_PI_patterns])

    # Store temporal decoding scores for this participant
    mean_scores_temporal_decoding.append([scores_mean_AR, scores_mean_PR, scores_mean_AI, scores_mean_PI])
    
    # Plot individual results of temporal decoding for this participant
    plt.figure()
    plt.figure(figsize=(10, 5))

    plt.plot(time, scores_mean_AR, label='Active Relevance', color='red')
    plt.plot(time, scores_mean_PR, label='Passive Relevance', color='orange')
    plt.plot(time, scores_mean_AI, label='Active Irrelevance', color='blue')
    plt.plot(time, scores_mean_PI, label='Passive Irrelevance', color='cyan')
    plt.axhline(0.5, color="k", linestyle="--", label="chance")
    plt.xlabel("Time")
    # plt.xticks([-200, 0, 100, 200, 300, 400, 500, 600, 700, 800])
    plt.ylabel("AUC")
    plt.legend(loc='upper left')
    plt.title(f"Temporal Decoding - Participant {SUBJ}")
    plt.ylim(0.3, 0.9)
    plt.show()
    
    # ------------------------------- SAVE ------------------------------------
    # Save the scores as numpy arrays
    # Save the scores for each condition as numpy arrays
    np.save(str(outputs_dir) + f"\\sub-{SUBJ}_AR_temporal_decoding_scores.npy", scores_AR)
    np.save(str(outputs_dir) + f"\\sub-{SUBJ}_PR_temporal_decoding_scores.npy", scores_PR)
    np.save(str(outputs_dir) + f"\\sub-{SUBJ}_AI_temporal_decoding_scores.npy", scores_AI)
    np.save(str(outputs_dir) + f"\\sub-{SUBJ}_PI_temporal_decoding_scores.npy", scores_PI)
    

# Calculate mean scores for each condition and each method
participant_mean_scores_temporal_decoding = np.mean(mean_scores_temporal_decoding, axis=0)
# Define a list to store the mean matrices temporal generalization for each condition
mean_matrices_per_condition = []
# For each condition
for condition_scores in zip(*mean_scores_temporal_generalization):
    # Convert the scores of each participant into a numpy array
    condition_scores_array = np.array(condition_scores)
    # Calculate the mean along axis 0 to obtain the mean of the matrices
    mean_matrix = np.mean(condition_scores_array, axis=0)
    # Add the mean matrix to the list
    mean_matrices_per_condition.append(mean_matrix)
    

# Convert mean scores for temporal decoding to numpy array
mean_scores_temporal_decoding_array = np.array(mean_scores_temporal_decoding)

# Calculate standard error between participants for each condition
#L'erreur type est calculée en divisant l'écart type de l'échantillon par la racine carrée de la taille de l'échantillon.
std_error_temporal_decoding = (np.std(mean_scores_temporal_decoding_array, axis=0) / np.sqrt(mean_scores_temporal_decoding_array.shape[0]))


#%% ______________________________ Permutation tests __________________________
scores_active_relevant = mean_scores_temporal_decoding_array[:,0,:]
scores_passive_relevant = mean_scores_temporal_decoding_array[:,1,:]
scores_active_irrelevant = mean_scores_temporal_decoding_array[:,2,:]
scores_passive_irrelevant = mean_scores_temporal_decoding_array[:,3,:]
# scores_ dimension arrays : (n_participants_, n_time_points) 

# -------------- To remove the first time windows of the baseline ---------------
if TIME_WINDOW_MS:
    n = int(200/time_window_ms)#200 means the baseline duration
else:
    n = int(200*sfreq/1000)

scores_active_relevant_without_baseline = scores_active_relevant[:, n : scores_active_relevant.shape[1]] 
scores_passive_relevant_without_baseline = scores_passive_relevant[:, n : scores_passive_relevant.shape[1]]
scores_active_irrelevant_without_baseline = scores_active_irrelevant[:, n : scores_active_irrelevant.shape[1]]
scores_passive_irrelevant_without_baseline = scores_passive_irrelevant[:, n : scores_passive_irrelevant.shape[1]]

# Apply the function for each condition
time_without_baseline = time[n:] # adapt the length of time vector
mask_active_relevant_without_baseline, timesAR, clusters_AR, cluster_p_values_AR = statistical_tests(scores_active_relevant_without_baseline, time_without_baseline)
mask_passive_relevant_without_baseline, timesPR, clusters_PR, cluster_p_values_PR = statistical_tests(scores_passive_relevant_without_baseline, time_without_baseline)
mask_active_irrelevant_without_baseline, timesAI, clusters_AI, cluster_p_values_AI = statistical_tests(scores_active_irrelevant_without_baseline, time_without_baseline)
mask_passive_irrelevant_without_baseline, timesPI, clusters_PI, cluster_p_values_PI = statistical_tests(scores_passive_irrelevant_without_baseline, time_without_baseline)

# to restore 0 values for the first n windows previously removed from statistical analyses (as they correspond to the baseline)
zeros_boolean_array = np.full((n,1), 0) # Création des nouvelles lignes remplies de "False"
mask_active_relevant_without_baseline = mask_active_relevant_without_baseline.reshape(-1,1)
mask_passive_relevant_without_baseline = mask_passive_relevant_without_baseline.reshape(-1,1)
mask_active_irrelevant_without_baseline = mask_active_irrelevant_without_baseline.reshape(-1,1)
mask_passive_irrelevant_without_baseline = mask_passive_irrelevant_without_baseline.reshape(-1,1)

mask_active_relevant = np.vstack((zeros_boolean_array, mask_active_relevant_without_baseline))
mask_passive_relevant = np.vstack((zeros_boolean_array, mask_passive_relevant_without_baseline))
mask_active_irrelevant = np.vstack((zeros_boolean_array, mask_active_irrelevant_without_baseline))
mask_passive_irrelevant = np.vstack((zeros_boolean_array, mask_passive_irrelevant_without_baseline))

# Condition comparaison
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
    ['darkred', 'darkorange', 'blue', 'darkturquoise'], 
    [mask_active_relevant, mask_passive_relevant, mask_active_irrelevant, mask_passive_irrelevant], time, n_features
)


# Plot mean scores for temporal decoding
# time = time*1000
plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[0], label='Active Relevant', color='red')
plt.plot(time, participant_mean_scores_temporal_decoding[1], label='Passive Relevant', color='orange')
plt.plot(time, participant_mean_scores_temporal_decoding[2], label='Active Irrelevant', color='blue')
plt.plot(time, participant_mean_scores_temporal_decoding[3], label='Passive Irrelevant', color='cyan')

# Plot standard error as error bands
plt.fill_between(time, participant_mean_scores_temporal_decoding[0] - std_error_temporal_decoding[0],
                  participant_mean_scores_temporal_decoding[0] + std_error_temporal_decoding[0], color='red', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[1] - std_error_temporal_decoding[1],
                  participant_mean_scores_temporal_decoding[1] + std_error_temporal_decoding[1], color='orange', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[2] - std_error_temporal_decoding[2],
                  participant_mean_scores_temporal_decoding[2] + std_error_temporal_decoding[2], color='blue', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[3] - std_error_temporal_decoding[3],
                  participant_mean_scores_temporal_decoding[3] + std_error_temporal_decoding[3], color='cyan', alpha=0.3)

plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')

plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=10)
# plt.xticks([-200, 0, 100, 200, 300, 400, 500, 600, 700, 800])
plt.ylabel("AUC")
plt.legend(loc='upper left', fontsize=10)
plt.title(f"Temporal Decoding ({CLASSIFIER}) - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, mask_active_relevant*0.48, 'o', color='red') # permutation test
plt.plot(time, mask_passive_relevant*0.46, 'o', color='orange') # permutation test
plt.plot(time, mask_active_irrelevant*0.44, 'o', color='blue') # permutation test
plt.plot(time, mask_passive_irrelevant*0.42, 'o', color='cyan') # permutation test

plot_file_name = plot_dir / f"{CLASSIFIER}_{TIME_WINDOW_MS}_{REDUCE_NUMBER_OF_ELECTRODES}_{n_features}_2.jpg"
plt.savefig(plot_file_name, dpi=300, bbox_inches='tight')

plt.show()

#%% ____________________________ Condition comparaison ________________________
# ----------------------------------------------------------------------------------------------------------------
# Plot mean scores for temporal decoding
# time = time*1000
plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[0], label='Active Relevance', color='red')
plt.plot(time, participant_mean_scores_temporal_decoding[2], label='Active Irrelevance', color='blue')
# Plot standard error as error bands
plt.fill_between(time, participant_mean_scores_temporal_decoding[0] - std_error_temporal_decoding[0],
                  participant_mean_scores_temporal_decoding[0] + std_error_temporal_decoding[0], color='red', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[2] - std_error_temporal_decoding[2],
                  participant_mean_scores_temporal_decoding[2] + std_error_temporal_decoding[2], color='blue', alpha=0.3)
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')
plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# plt.xticks([-200, 0, 100, 200, 300, 400, 500, 600, 700, 800])
plt.ylabel("AUC")
plt.legend(loc='upper left')
plt.title("Temporal Decoding - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, mask_relevant_vs_irrelevant_active*0.48, 'o', color='red') # permutation test
plt.show()

# ----------------------------------------------------------------------------------------------------------------
# Plot mean scores for temporal decoding
# time = time*1000
plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[0], label='Active Relevance', color='red')
plt.plot(time, participant_mean_scores_temporal_decoding[1], label='Passive Relevance', color='orange')
# Plot standard error as error bands
plt.fill_between(time, participant_mean_scores_temporal_decoding[0] - std_error_temporal_decoding[0],
                  participant_mean_scores_temporal_decoding[0] + std_error_temporal_decoding[0], color='red', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[1] - std_error_temporal_decoding[1],
                  participant_mean_scores_temporal_decoding[1] + std_error_temporal_decoding[1], color='orange', alpha=0.3)
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')
plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# plt.xticks([-200, 0, 100, 200, 300, 400, 500, 600, 700, 800])
plt.ylabel("AUC")
plt.legend(loc='upper left')
plt.title("Temporal Decoding - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, mask_active_vs_passive_relevant*0.48, 'o', color='red') # permutation test
plt.show()

# ----------------------------------------------------------------------------------------------------------------
# Plot mean scores for temporal decoding
# time = time*1000
plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[2], label='Active Irrelevance', color='blue')
plt.plot(time, participant_mean_scores_temporal_decoding[3], label='Passive Irrelevance', color='darkturquoise')
# Plot standard error as error bands
plt.fill_between(time, participant_mean_scores_temporal_decoding[2] - std_error_temporal_decoding[2],
                  participant_mean_scores_temporal_decoding[2] + std_error_temporal_decoding[2], color='blue', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[3] - std_error_temporal_decoding[3],
                  participant_mean_scores_temporal_decoding[3] + std_error_temporal_decoding[3], color='darkturquoise', alpha=0.3)
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')
plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# plt.xticks([-200, 0, 100, 200, 300, 400, 500, 600, 700, 800])
plt.ylabel("AUC")
plt.legend(loc='upper left')
plt.title("Temporal Decoding - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, mask_active_vs_passive_irrelevant*0.48, 'o', color='red') # permutation test
plt.show()

# ----------------------------------------------------------------------------------------------------------------
# Plot mean scores for temporal decoding
# time = time*1000
plt.figure()
plt.figure(figsize=(10, 5))
plt.plot(time, participant_mean_scores_temporal_decoding[1], label='Passive Relevance', color='orange')
plt.plot(time, participant_mean_scores_temporal_decoding[3], label='Passive Irrelevance', color='darkturquoise')
# Plot standard error as error bands
plt.fill_between(time, participant_mean_scores_temporal_decoding[1] - std_error_temporal_decoding[1],
                  participant_mean_scores_temporal_decoding[1] + std_error_temporal_decoding[1], color='orange', alpha=0.3)
plt.fill_between(time, participant_mean_scores_temporal_decoding[3] - std_error_temporal_decoding[3],
                  participant_mean_scores_temporal_decoding[3] + std_error_temporal_decoding[3], color='darkturquoise', alpha=0.3)
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.axvline(0, color='k', linestyle='dotted')
plt.xlabel("Time (s)")
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# plt.xticks([-200, 0, 100, 200, 300, 400, 500, 600, 700, 800])
plt.ylabel("AUC")
plt.legend(loc='upper left')
plt.title("Temporal Decoding - Average scores across participants (n=24)")
plt.ylim(0.4, 0.61)
plt.plot(time, mask_relevant_vs_irrelevant_passive*0.48, 'o', color='red') # permutation test
plt.show()


#%% ___________________ Get Electrodes of strong coefficient __________________

top_n_electrodes = 3

if VISUAL_CORTEX == False:
    # ----- CALCUL MOYENNE COEFF QUAND NOMBRE D'ELECTRODES DIFFERENT PAR CONDITION -----
    mean_coef_patterns = get_mean_coef_patterns_when_electrod_number_differs_per_condition(list_of_coef_patterns)
    mean_coef_filters = get_mean_coef_patterns_when_electrod_number_differs_per_condition(list_of_coef_filters)

else:
    mean_coef_patterns = np.mean(list_of_coef_patterns, axis=0) #mean across participants #(4, 32, 10)
    mean_coef_filters = np.mean(list_of_coef_filters, axis=0) #mean across participants

# --- COEF PATTERNS ---
# Utilisation de la fonction pour récupérer les électrodes pour chaque condition
strong_electrods_per_condition_per_significant_time_point = get_strong_electrods_per_condition_per_significant_time_point(
    AR, PR, AI, PI,
    mean_coef_patterns,
    mask_active_relevant, mask_passive_relevant, mask_active_irrelevant, mask_passive_irrelevant, 
    VISUAL_CORTEX,
    n_electrodes=top_n_electrodes,
)

print("\n Coef Patterns:")
# Affichage des résultats
for condition, data in strong_electrods_per_condition_per_significant_time_point.items(): 
    print(f"\n \n Pour la condition {condition}:")
    for index, ch_name_strong_electrodes in data.items():
        print(f"\nPour l'indice temporel {index}, les {top_n_electrodes} premières électrodes dans l'ordre des coefficients les plus différents de 0 sont : {ch_name_strong_electrodes}")



"""_________________________FIND SIGNIFICANT ELECTRODS______________________"""

# Initializing a dictionary to store arrays per condition
coef_patterns_dict = {'AR': [], 'PR': [], 'AI': [], 'PI': []}

# Iterate through each participant
for participant in list_of_coef_patterns:
    # Iterate through each condition for this participant
    for condition_key, condition in zip(coef_patterns_dict.keys(), participant):
        # Add the array to the corresponding list in the dictionary
        coef_patterns_dict[condition_key].append(condition)

# Convert the lists of arrays into numpy arrays
coef_patterns_dict = {condition_key: np.array(arrays) for condition_key, arrays in coef_patterns_dict.items()}

AR_array = coef_patterns_dict['AR']  # Array for condition 'AR'
PR_array = coef_patterns_dict['PR']  # Array for condition 'PR'
AI_array = coef_patterns_dict['AI']  # Array for condition 'AI'
PI_array = coef_patterns_dict['PI']  # Array for condition 'PI'

# Example with the 'AR' condition so we take the mask "mask_active_relevant"
# Suppose 'AR_array' is your array of shape (24, 60, 10)
# and 'mask_active_relevant' is your mask array of shape (10, 1)
# Create an empty dictionary to store the arrays for each index
# ------------------------------- Find for each condition ------------------------
significant_electrods_per_condition_per_significant_time_point = find_significant_electrods_from_coefficient_classification(AR, PR, AI, PI,
                                                           mask_active_relevant, mask_passive_relevant,
                                                           mask_active_irrelevant, mask_passive_irrelevant,
                                                           AR_array, PR_array, AI_array, PI_array)      
        
#%% _________________________________ Topomaps ________________________________
"""_________________________TOPO MAP WITH MASK SIGNIFICANT ELECTRODS______________________"""
# --- COEF PATTERNS ---
grand_average_AR_patterns = mne.grand_average(list_of_topomap_evoked_AR_patterns)
grand_average_PR_patterns = mne.grand_average(list_of_topomap_evoked_PR_patterns)
grand_average_AI_patterns = mne.grand_average(list_of_topomap_evoked_AI_patterns)
grand_average_PI_patterns = mne.grand_average(list_of_topomap_evoked_PI_patterns)

# Utilisation de la fonction pour chaque condition
conditions = {"AR": mask_active_relevant, 
              "PR": mask_passive_relevant, 
              "AI": mask_active_irrelevant, 
              "PI": mask_passive_irrelevant}
times = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
# times = (0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.65, 0.7)

coef_name = 'patterns'
for key_condition, mask_condition in conditions.items():
    print(f"Plotting topomap for condition: {condition}")
    if key_condition == "AR":
        plot_topomap_with_mask(grand_average_AR_patterns, significant_electrods_per_condition_per_significant_time_point, key_condition, mask_condition, times, coef_name)
    elif key_condition == "PR":
        plot_topomap_with_mask(grand_average_PR_patterns, significant_electrods_per_condition_per_significant_time_point, key_condition, mask_condition, times, coef_name)
    elif key_condition == "AI":
        plot_topomap_with_mask(grand_average_AI_patterns, significant_electrods_per_condition_per_significant_time_point, key_condition, mask_condition, times, coef_name)
    elif key_condition == "PI":
        plot_topomap_with_mask(grand_average_PI_patterns, significant_electrods_per_condition_per_significant_time_point, key_condition, mask_condition, times, coef_name)



"""_________________________TOPO MAP WITH MASK STRONG ELECTRODS______________________"""
# --- COEF PATTERNS ---
grand_average_AR_patterns = mne.grand_average(list_of_topomap_evoked_AR_patterns)
grand_average_PR_patterns = mne.grand_average(list_of_topomap_evoked_PR_patterns)
grand_average_AI_patterns = mne.grand_average(list_of_topomap_evoked_AI_patterns)
grand_average_PI_patterns = mne.grand_average(list_of_topomap_evoked_PI_patterns)

# Utilisation de la fonction pour chaque condition
conditions = {"AR": mask_active_relevant, 
              "PR": mask_passive_relevant, 
              "AI": mask_active_irrelevant, 
              "PI": mask_passive_irrelevant}
times = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
# times = (0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.65, 0.7)

coef_name = 'patterns'
for key_condition, data_condition in conditions.items():
    print(f"Plotting topomap for condition: {condition}")
    if key_condition == "AR":
        plot_topomap_with_mask(grand_average_AR_patterns, strong_electrods_per_condition_per_significant_time_point, key_condition, data_condition, times, coef_name)
    elif key_condition == "PR":
        plot_topomap_with_mask(grand_average_PR_patterns, strong_electrods_per_condition_per_significant_time_point, key_condition, data_condition, times, coef_name)
    elif key_condition == "AI":
        plot_topomap_with_mask(grand_average_AI_patterns, strong_electrods_per_condition_per_significant_time_point, key_condition, data_condition, times, coef_name)
    elif key_condition == "PI":
        plot_topomap_with_mask(grand_average_PI_patterns, strong_electrods_per_condition_per_significant_time_point, key_condition, data_condition, times, coef_name)


#%% _______________ Save all electrods which has been significant _____________
if REDUCE_NUMBER_OF_ELECTRODES == False:
    electrodes_which_have_been_significant = []
    # Traverse the dictionaries and concatenate electrode lists
    for dict_conditions in significant_electrods_per_condition_per_significant_time_point.values():
        for electrode_list in dict_conditions.values():
            electrodes_which_have_been_significant.extend(electrode_list)
            
    significant_electrodes = np.array(electrodes_which_have_been_significant)        
    np.save(outputs_dir / f"significant_channels_features_{CLASSIFIER}_{TIME_WINDOW_MS}.npy", significant_electrodes)
            


"""________________________Counts significant clusters_________________________ """
n_significant_clusters=0
liste_clusters_conditions = [mask_active_relevant, mask_passive_relevant,
                             mask_active_irrelevant, mask_passive_irrelevant]
for l in liste_clusters_conditions:  
    for i in range(0,len(l)):
        if l[i] == True:
            n_significant_clusters += 1
    



#%% ______________ Save the classification performance & parameters ___________
if TIME_WINDOW_MS:
    n = int(200/time_window_ms)#200 means the baseline duration
else:
    n = int(200*sfreq/1000)

# Define a dictionary to store mean scores for each condition along with other parameters
A_mean_scores_dict = {}
# Loop through each condition and calculate mean scores
for idx_condition, condition in enumerate(["AR", "PR", "AI", "PI"]):
    
    for idx_time, _ in enumerate(mean_scores_temporal_decoding[0][idx_condition]): # to get only the time points with significant clusters for the current condition
        if liste_clusters_conditions[idx_condition][idx_time] == 1:
        
            # Calculate mean scores for the current condition across all participants
            mean_score_condition = np.mean(mean_scores_temporal_decoding_array[:, idx_condition, idx_time])
                
            # Define other parameters
            classifier_used = CLASSIFIER
            time_window_ms_used = TIME_WINDOW_MS
            reduce_number_of_electrodes = REDUCE_NUMBER_OF_ELECTRODES
            
            # Store everything in the dictionary
            A_mean_scores_dict[condition] = {
                "mean_score": mean_score_condition,
                "classifier_used": classifier_used,
                "time_window_ms_used": time_window_ms_used,
                "reduce_number_of_electrodes": reduce_number_of_electrodes,
                "n_features": n_features,
                "n_significant_clusters": n_significant_clusters,
            }

clusters_final_mask = [1 if any(mask) else 0 for mask in zip(*liste_clusters_conditions)]

# Calculate overall mean score across all conditions
for idx_time, _ in enumerate(mean_scores_temporal_decoding[0][0]): # to get only the time points with significant clusters for the current condition
    if clusters_final_mask[idx_time] == 1:
        # Calculate overall mean score across all conditions   
        overall_mean_score  = np.mean(mean_scores_temporal_decoding_array[:,:,idx_time])

        # Define other parameters
        classifier_used = CLASSIFIER
        time_window_ms_used = TIME_WINDOW_MS
        reduce_number_of_electrodes = REDUCE_NUMBER_OF_ELECTRODES
        # Store everything in the dictionary
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
if not os.path.exists(file_path): # If the classif perf file doesn't exist
    print("coucou")
    A_overall_mean_scores_df.to_excel(file_path, header=True, index=False) # Save to an excel file
    
else:
    print("The file already exists.")
    df_classif_perf = pd.read_excel(file_path) # get the excel file into a pandas dataframe
    
    # Verify if there is already the same classif into the file/dataframe
    is_already_present = df_classif_perf[df_classif_perf.eq(A_overall_mean_scores_df).all(axis=1)]
    if is_already_present.empty:
        df_classif_perf = df_classif_perf.append(A_overall_mean_scores_df, ignore_index=True) # update the dataframe with new classif perf
    else:   
        print("This classification already exists on the dataframe.")    
    
    df_classif_perf.to_excel(file_path, index=False)  # convert and save the updating excel file




