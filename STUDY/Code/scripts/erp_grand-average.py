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
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.collections import PolyCollection
from scipy import stats


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
plot_dir = study_path / "Plots" / "erp" / "grand_average"
plot_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Path of the epochs data in bids format
bids_derivatives_dir = study_path / "Data" / "eeg" / "derivatives" / "epochs"

# Path of the inputs for the electrodes selection
inputs_dir = study_path / "Code" / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

plot_dir_individual = study_path / "Plots" / "erp" / "individual_grand_average"
plot_dir_individual.mkdir(parents=True, exist_ok=True)

outputs_dir = study_path / "Code" / "outputs"

#%% ______________________________ FUNCTIONS _________________________________

# Compute SEM
def custom_sem(data):
    """
    Prend en entrée une matrice (n_participants, n_temps)
    Retourne les bornes inférieures et supérieures du SEM
    """
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    return mean - sem, mean + sem


# Define Cousineau-Morey functions
def cousineau_morey_sem(data):
    """
    Calculates the Cousineau-Morey Standard Error of the Mean (SEM).
    Assumes the input data (n_subjects, n_times) has already been normalized.
    """
    mean = np.mean(data, axis=0)
    sem_norm = stats.sem(data, axis=0)
    M = 4  # Number of conditions
    morey_factor = np.sqrt(M / (M - 1))
    sem_cm = sem_norm * morey_factor
    return mean - sem_cm, mean + sem_cm

def cousineau_morey_ci(data):
    """
    Calculates the Cousineau-Morey 95% Confidence Interval.
    Assumes the input data (n_subjects, n_times) has already been normalized.
    """
    mean = np.mean(data, axis=0)
    sem_norm = stats.sem(data, axis=0)
    M = 4  # Number of conditions
    morey_factor = np.sqrt(M / (M - 1))
    sem_cm = sem_norm * morey_factor
    
    # 95% CI using critical t-value
    n_subjects = data.shape[0]
    t_crit = stats.t.ppf(0.975, df=n_subjects - 1)
    ci_half = t_crit * sem_cm
    return mean - ci_half, mean + ci_half

def get_cousineau_morey_sem_diff(condA, condB):
    """
    Calcule la moyenne et le SEM de Cousineau-Morey pour 2 conditions appariées (M=2).
    condA, condB : Tableaux numpy de dimension (n_participants, n_temps)
    """
    subj_mean = (condA + condB) / 2.0
    grand_mean = np.mean(subj_mean, axis=0, keepdims=True)
    
    norm_A = condA - subj_mean + grand_mean
    norm_B = condB - subj_mean + grand_mean
    
    M = 2
    morey_factor = np.sqrt(M / (M - 1))
    
    mean_A = np.mean(condA, axis=0)
    mean_B = np.mean(condB, axis=0)
    
    sem_A = stats.sem(norm_A, axis=0) * morey_factor
    sem_B = stats.sem(norm_B, axis=0) * morey_factor
    
    return mean_A, sem_A, mean_B, sem_B


#%% _______________________________ MAIN SCRIPT________________________________

state_participant_error_dropping = False
erps_per_participant = []

list_of_individual_evoked_AR = []
list_of_individual_evoked_PR = []
list_of_individual_evoked_AI = []
list_of_individual_evoked_PI = []

nb_active_relevant_epochs=[]
nb_passive_relevant_epochs=[]
nb_active_irrelevant_epochs=[]
nb_passive_irrelevant_epochs=[]

data_dict={}

# Create a list of conditions
conditions = ["active/relevance",
              "passive/relevance",
              "active/irrelevance",
              "passive/irrelevance"]

for condition in conditions:
    data_dict[condition.replace('/','_')]=[]
    
n1a_window = (100, 200)  # N1a (100 to 200ms in Anterior sites)
n170_window = (150, 200) # N170 (150 to 200ms in Posterior sites)
n2_window = (200, 350)  # N2 (200 to 250ms)
p1_window = (80, 200)  # P1 (110 to 190ms)
p2_window = (190, 295)  # P2 (190 to 280ms)
p300_window = (350, 500)  # P3b (350 to 500ms)
p300_window2 = (300, 500)  # P3b (300 to 500ms)


# Loop over participants (from P002 to P006)
for PARTICIPANT_NUM in range(1,25):
    
    if PARTICIPANT_NUM < 10:
        SUBJ = f'0{PARTICIPANT_NUM}'
    else:
        SUBJ = f'{PARTICIPANT_NUM}'   
    
    bids_subj = f"sub-{SUBJ}"
    # To open the eeg data file previously saved
    epo_path = bids_derivatives_dir / bids_subj / "eeg" / f"{bids_subj}_task-2ifcGoNogo_proc-epo_eeg.fif"

    # Load epochs
    epochs = mne.read_epochs(epo_path)
    
    # Epoching settings
    tmin =  -0.200  # start of each epoch (in sec)
    tmax =  0.800  # end of each epoch (in sec)
    baseline = (-0.2, 0) # Marzecova et al. 2017 used 200ms baseline
       
    # -------------------------------------------------------------------------
    # --------------- ADAPTED FOR CLASSICAL CONDITIONS COMPARISON -------------
    # -------------------------------------------------------------------------
    AR = epochs["active/relevance"] # condition active relevance
    PR = epochs["passive/relevance"] # condition passive relevance
    AI = epochs["active/irrelevance"] # condition active irrelevance
    PI = epochs["passive/irrelevance"] # condition passive irrelevance
    
    evoked_AR = AR.average()
    list_of_individual_evoked_AR.append(evoked_AR)    
    evoked_PR = PR.average()
    list_of_individual_evoked_PR.append(evoked_PR) 
    evoked_AI = AI.average()
    list_of_individual_evoked_AI.append(evoked_AI)
    evoked_PI = PI.average()
    list_of_individual_evoked_PI.append(evoked_PI)
    
    # Count the number of epochs rejected per condition for each participant
    nb_active_relevant_epochs.append(140-len(AR))
    nb_passive_relevant_epochs.append(140-len(PR))
    nb_active_irrelevant_epochs.append(140-len(AI))
    nb_passive_irrelevant_epochs.append(140-len(PI))
 
   
# -------------------------------------------------------------------------
# ---------------- NEW - COUSINEAU-MOREY PRE-NORMALIZATION ___________________
# -------------------------------------------------------------------------
n_subjects = len(list_of_individual_evoked_AR)
M = 4  # Number of conditions (AR, PR, AI, PI)

# Deep copy our individual evokeds to preserve the originals intact
list_of_individual_evoked_AR_cm = [e.copy() for e in list_of_individual_evoked_AR]
list_of_individual_evoked_PR_cm = [e.copy() for e in list_of_individual_evoked_PR]
list_of_individual_evoked_AI_cm = [e.copy() for e in list_of_individual_evoked_AI]
list_of_individual_evoked_PI_cm = [e.copy() for e in list_of_individual_evoked_PI]

# 1. Calculate each participant's mean across the 4 conditions
subject_means = []
for i in range(n_subjects):
    subj_mean = (list_of_individual_evoked_AR_cm[i].data +
                 list_of_individual_evoked_PR_cm[i].data +
                 list_of_individual_evoked_AI_cm[i].data +
                 list_of_individual_evoked_PI_cm[i].data) / M
    subject_means.append(subj_mean)

# 2. Calculate the Grand Mean across all subjects and conditions
grand_mean = np.mean(subject_means, axis=0)

# 3. Apply Cousineau Normalization: Y_normalized = Y - SubjectMean + GrandMean
for i in range(n_subjects):
    list_of_individual_evoked_AR_cm[i].data = list_of_individual_evoked_AR_cm[i].data - subject_means[i] + grand_mean
    list_of_individual_evoked_PR_cm[i].data = list_of_individual_evoked_PR_cm[i].data - subject_means[i] + grand_mean
    list_of_individual_evoked_AI_cm[i].data = list_of_individual_evoked_AI_cm[i].data - subject_means[i] + grand_mean
    list_of_individual_evoked_PI_cm[i].data = list_of_individual_evoked_PI_cm[i].data - subject_means[i] + grand_mean

# Pass the Cousineau-Normalized lists to the dictionary
evokeds_dict = {
    "Active Relevant": list_of_individual_evoked_AR_cm,
    "Passive Relevant": list_of_individual_evoked_PR_cm,
    "Active Irrelevant": list_of_individual_evoked_AI_cm,
    "Passive Irrelevant": list_of_individual_evoked_PI_cm,
}
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

 
grand_average_AR = mne.grand_average(list_of_individual_evoked_AR)
grand_average_PR = mne.grand_average(list_of_individual_evoked_PR)
grand_average_AI = mne.grand_average(list_of_individual_evoked_AI)
grand_average_PI = mne.grand_average(list_of_individual_evoked_PI)

grand_average_AR.comment = 'Active Relevant'
grand_average_PR.comment = 'Passive Relevant'
grand_average_AI.comment = 'Active Irrelevant'
grand_average_PI.comment = 'Passive Irrelevant'

list_of_grand_average = [grand_average_AR,
                         grand_average_PR,
                         grand_average_AI,
                         grand_average_PI]

#N1a
fronto_central_cluster = ['FCz','Fz','FC1','FC2', 'F1', 'F2'] # like Emma's thesis
#N170
parieto_occipital_cluster_L = ['P5', 'P7', 'PO3', 'PO7']
parieto_occipital_cluster_R = ['P6', 'P8', 'PO4', 'PO8']

#P1
parieto_occipital_cluster_L_2 = ['O1','PO3', 'PO7']
parieto_occipital_cluster_R_2 = ['O2','PO4', 'PO8']
parieto_occipital_cluster_1 = ['P5', 'P7', 'PO3', 'PO7', 'P6', 'P8', 'PO4', 'PO8']
parieto_occipital_cluster_2 = ['O1','PO3', 'PO7','O2','PO4', 'PO8']
parieto_line_cluster_L = ['P7', 'P5', 'P3', 'P1']
parieto_line_cluster_R = ['P8', 'P6', 'P4', 'P2']
parieto_line_cluster = ['P7', 'P5', 'P3', 'P1', 'P8', 'P6', 'P4', 'P2']


#%% ______________________________ P1 component _______________________________
"""__________________________ Parieto-occipital cluster _______________________"""
limit = 8
# mne.viz.plot_compare_evokeds(list_of_grand_average, picks=parieto_occipital_cluster_1, colors=('red', 'orange', 'blue', 'darkturquoise'), title='O1, PO3, PO7- Grand average (n=24)',
#                              show_sensors=True, combine='mean', ylim=dict(eeg=[-limit, limit]))
# mne.viz.plot_compare_evokeds(list_of_grand_average, picks=parieto_occipital_cluster_2, colors=('red', 'orange', 'blue', 'darkturquoise'), title='O2, PO4, PO8 - Grand average (n=24)',
#                              show_sensors=True, combine='mean', ylim=dict(eeg=[-limit, limit]))
# mne.viz.plot_compare_evokeds(list_of_grand_average, picks=parieto_occipital_cluster_L_2, colors=('red', 'orange', 'blue', 'darkturquoise'), title='O1, PO3, PO7- Grand average (n=24)',
#                              show_sensors=True, combine='mean', ylim=dict(eeg=[-limit, limit]))
# mne.viz.plot_compare_evokeds(list_of_grand_average, picks=parieto_occipital_cluster_R_2, colors=('red', 'orange', 'blue', 'darkturquoise'), title='O2, PO4, PO8 - Grand average (n=24)',
#                              show_sensors=True, combine='mean', ylim=dict(eeg=[-limit, limit]))

"""_______________________________ P6 electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='P6', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False, # /!\ très important /!\
                             time_unit='ms',
                             legend=False,
                             # ci=True, 
                             #we can use cousineau_morey_ci instead, or the classical sem with custom_sem"
                             ci=cousineau_morey_sem,  
                             axes=ax)

ax.axvspan(
    p1_window[0],
    p1_window[1],
    facecolor='none',  
    edgecolor='black',    
    linestyle="-",
    linewidth=2.5
)
ax.set_title('P6 electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  # Add horizontal line y=0
ax.set_xlim(-200, 800)  # Adjust limits of x axis if necessary 
ax.legend(loc='lower left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    
# ASTUCE : Mettre les limites du SEM en pointillés pour les conditions Passives
# Ordre dans le dict : 0=AR, 1=PR, 2=AI, 3=PI. On modifie donc les index 1 et 3.
for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout()  # If the layout is not good
# If you want to add the asterisk on peak component
peak_x = np.mean(p1_window)  # For instance, we take the middle of the window
peak_y = -1.1  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
            # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
            # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P1component_P6.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\P1component_P6.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()

#%% ______________________________ N1a component ______________________________
"""_______________________________ AFz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='AFz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 2]),
                             vlines=[0],
                             combine='mean',
                             show_sensors='lower right',
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             legend=False,
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(n1a_window[0], n1a_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    n1a_window[0],
    n1a_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
# Personnalisation supplémentaire de l'axe si nécessaire
ax.set_title('AFz', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1) # Add horizontal line y=0
ax.set_xlim(-200, 800)  # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    

for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n1a_window)  
peak_y = -2  # Height above peak where to place asterisk
# ax.annotate('ns', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N1acomponent_AFz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\N1acomponent_AFz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()

"""___________________________ Fronto-central cluster _________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks=fronto_central_cluster, colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 2]),
                             vlines=[0],
                             combine='mean',
                             show_sensors='lower right',
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             legend=False,
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(n1a_window[0], n1a_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    n1a_window[0],
    n1a_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
# Personnalisation supplémentaire de l'axe si nécessaire
ax.set_title('Fronto-central cluster', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1) # Add horizontal line y=0
ax.set_xlim(-200, 800)  # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    

for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n1a_window)  
peak_y = -2  # Height above peak where to place asterisk
# ax.annotate('ns', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))

plt.savefig(str(plot_dir) + "\\N1acomponent_Fronto-central-cluster.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\N1acomponent_Fronto-central-cluster.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()


#%% ______________________________ P2 component _______________________________
"""_______________________________ Oz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='Oz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, limit]),
                             vlines=[0],
                             combine='mean',
                             show_sensors=True,
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(p2_window[0], p2_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    p2_window[0],
    p2_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
ax.set_title('Oz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  # Add horizontal line y=0
ax.set_xlim(-200, 800)   # Adjust limits of x axis if necessary 
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    
# ASTUCE : Mettre les limites du SEM en pointillés pour les conditions Passives
# Ordre dans le dict : 0=AR, 1=PR, 2=AI, 3=PI. On modifie donc les index 1 et 3.
for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout()  
# If you want to add the asterisk on peak component
peak_x = np.mean(p2_window)  
peak_y = 7  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P2component_Oz.jpg", dpi=300, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\P2component_Oz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()


"""_______________________________ POz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='POz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, limit]),
                             vlines=[0],
                             combine='mean',
                             show_sensors=True,
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(p2_window[0], p2_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    p2_window[0],
    p2_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
ax.set_title('POz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  # Add horizontal line y=0
ax.set_xlim(-200, 800)   # Adjust limits of x axis if necessary 
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    
# ASTUCE : Mettre les limites du SEM en pointillés pour les conditions Passives
# Ordre dans le dict : 0=AR, 1=PR, 2=AI, 3=PI. On modifie donc les index 1 et 3.
for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout()  
# If you want to add the asterisk on peak component
peak_x = np.mean(p2_window)  
peak_y = 7  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P2component_POz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\P2component_POz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()


#%% ______________________________ N2b component ______________________________
"""_______________________________ Cz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='Cz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             show_sensors='lower right',
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(n2_window[0], n2_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    n2_window[0],
    n2_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
ax.set_title('Cz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  # Add horizontal line y=0
ax.set_xlim(-200, 800) # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    
for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n2_window)  
peak_y = -3  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N2bcomponent_Cz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\N2bcomponent_Cz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()

"""_______________________________ FCz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='FCz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             show_sensors='lower right',
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(n2_window[0], n2_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    n2_window[0],
    n2_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
# ax.axvspan( p300_window[0], p300_window[1], color='lightgray', alpha=0.3)
ax.set_title('FCz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  # Ajouter une ligne horizontale à y=0
ax.set_xlim(-200, 800)  # Ajuster les limites de l'axe x si nécessaire
# ax.set_ylim(-4, 3)  # Ajuster les limites de l'axe x si nécessaire

ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    

for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n2_window)  
peak_y = -3 # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N2bcomponent_FCz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\N2bcomponent_FCz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()

"""_______________________________ Fz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='Fz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             show_sensors=True,
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(n2_window[0], n2_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    n2_window[0],
    n2_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
ax.set_title('Fz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  # Add horizontal line y=0
ax.set_xlim(-200, 800) # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    
for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n2_window)  
peak_y = -3  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N2bcomponent_Fz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\N2bcomponent_Fz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()

"""_______________________________ AFz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='AFz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             show_sensors=True,
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(n2_window[0], n2_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    n2_window[0],
    n2_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
ax.set_title('AFz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  # Add horizontal line y=0
ax.set_xlim(-200, 800) # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    

for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n2_window)  
peak_y = -3  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N2bcomponent_AFz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\N2bcomponent_AFz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()
#%% ______________________________ P3b component ______________________________
"""________________________________Pz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='Pz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-3, 6]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             show_sensors=True,
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(p300_window[0], p300_window[1], color='lightgray', alpha=0.20)
ax.axvspan(
    p300_window[0],
    p300_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
ax.set_title('Pz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=54)
ax.set_ylabel('Amplitude (µV)', fontsize=54)
ax.axhline(0, color='grey', linewidth=1)   # Add horizontal line y=0
ax.set_xlim(-200, 800)  # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=44)
ax.tick_params(axis='y', labelsize=54)
ax.tick_params(axis='x', labelsize=54)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    

for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )
    
# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(p300_window)  # For instance, we take the middle of the window
peak_y = 4.75 # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=18, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P3bcomponent_Pz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\P3bcomponent_Pz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()

"""________________________________CPz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='CPz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             show_sensors='lower right',
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(p300_window[0], p300_window[1], color='lightgray', alpha=0.3)
ax.axvspan(
    p300_window[0],
    p300_window[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
ax.set_title('CPz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  
ax.set_xlim(-200, 800)  
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)
    

for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout() 
# peak_x = np.mean(p300_window)  
# peak_y = 3.5  
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P3bcomponent_CPz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\P3bcomponent_CPz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()

"""_________________________________Cz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(evokeds_dict, picks='Cz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 5},
                                     "Active Irrelevant": {"linewidth": 3, "linestyle":'--'},
                                     "Passive Relevant": {"linewidth": 5},
                                     "Passive Irrelevant": {"linewidth": 3, "linestyle":'--'}},
                             show=False,
                             time_unit='ms',
                             show_sensors='lower right',
                             # ci=True,
                             ci=cousineau_morey_sem,
                             axes=ax)
# Add the shaded gray area
# ax.axvspan(p300_window2[0], p300_window2[1], color='lightgray', alpha=0.3)
ax.axvspan(
    p300_window2[0],
    p300_window2[1],
    facecolor='none',   # pas de remplissage
    edgecolor='black',   # contour 
    linestyle="-", #time-window,
    linewidth=2.5
)
ax.set_title('Cz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=46)
ax.set_ylabel('Amplitude (µV)', fontsize=46)
ax.axhline(0, color='grey', linewidth=1)  
ax.set_xlim(-200, 800)  
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=38)
ax.tick_params(axis='y', labelsize=46)
ax.tick_params(axis='x', labelsize=46)

ci_polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
for poly in ci_polys:
    poly.set_alpha(0.10)

for idx, color in zip([2, 3], ['blue', 'darkturquoise']):
    poly_passive = ax.collections[idx]
    poly_passive.set_edgecolor(color)
    poly_passive.set_linestyle('--')
    poly_passive.set_linewidth(3)
    
if epochs.baseline is not None:
    # Add the shaded gray area
    ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0,
               # label='Baseline'
               )

# plt.tight_layout() 
# peak_x = np.mean(p300_window)  
# peak_y = 3.5  
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P3bcomponent_Cz.jpg", dpi=400, bbox_inches='tight')  # Save in JPG format
plt.savefig(str(plot_dir) + "\\P3bcomponent_Cz.pdf", dpi=600, bbox_inches='tight')  # Save in pdf 
plt.show()

#%%
participants = [f"{i:02d}" for i in range(1, 25)]

for i, subj in enumerate(participants):

    evokeds = [
        list_of_individual_evoked_AR[i],
        list_of_individual_evoked_PR[i],
        list_of_individual_evoked_AI[i],
        list_of_individual_evoked_PI[i]
    ]

    fig, ax = plt.subplots(figsize=(12,8))

    mne.viz.plot_compare_evokeds(
        evokeds,
        picks=fronto_central_cluster,
        combine='mean',
        colors=('red','orange','blue','darkturquoise'),
        ylim=dict(eeg=[-6,6]),
        time_unit='ms',
        show=False,
        axes=ax
    )

    ax.axvspan(*n1a_window, color='lightgray', alpha=0.3)
    ax.axhline(0, color='black')
    ax.set_xlim(-200,800)
    ax.set_title(f'Sub-{subj} - Fronto-central cluster')

    plt.savefig(plot_dir_individual / f"sub-{subj}_N1a_cluster.png")
    plt.close(fig)

#%%
# =============================================================================
# --- INTERACTION P3b : (AR - PR) vs (AI - PI) ---
# =============================================================================

ch_name = 'Pz'

# Création de matrices de taille (n_participants, n_temps)
data_AR = np.array([ev.copy().pick(ch_name).data[0] for ev in list_of_individual_evoked_AR])
data_PR = np.array([ev.copy().pick(ch_name).data[0] for ev in list_of_individual_evoked_PR])
data_AI = np.array([ev.copy().pick(ch_name).data[0] for ev in list_of_individual_evoked_AI])
data_PI = np.array([ev.copy().pick(ch_name).data[0] for ev in list_of_individual_evoked_PI])
times_ms = list_of_individual_evoked_AR[0].times * 1000


diff_rel = (data_AR - data_PR) * 1e6    
diff_irrel = (data_AI - data_PI) * 1e6


# SEM
mean_r, err_r, mean_i, err_i = get_cousineau_morey_sem_diff(diff_rel, diff_irrel)


fig, ax = plt.subplots(figsize=(24, 20))

# Relevant (AR - PR)
ax.plot(times_ms, mean_r, label="(Active Rel. - Passive Rel.)", color="black", linewidth=5)
ax.fill_between(times_ms, mean_r - err_r, mean_r + err_r, 
                facecolor="black", edgecolor="black", linestyle="-", linewidth=1.0, alpha=0.25)

# Irrelevant (AI - PI)
ax.plot(times_ms, mean_i, label="(Active Irrel. - Passive Irrel.)", color="black", linestyle="--", linewidth=3)
ax.fill_between(times_ms, mean_i - err_i, mean_i + err_i,
                facecolor="black", edgecolor="black", linestyle="--", linewidth=1.2, alpha=0.15)

ax.axvspan(p300_window[0], p300_window[1], facecolor='none', edgecolor='black', linestyle="-", linewidth=2.5)
ax.axhline(0, color='grey', linewidth=1)
ax.axvline(0, color="black", linestyle="--", linewidth=1)

ax.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0)

ax.set_xlim(-200, 800)
ax.set_ylim(-2, 2)


ax.set_title('Interaction P3b (Pz)', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=54)
ax.set_ylabel('Amplitude Difference (µV)', fontsize=54)

ax.tick_params(axis='y', labelsize=54)
ax.tick_params(axis='x', labelsize=54)
ax.legend(loc='upper left', fontsize=48)

plt.savefig(str(plot_dir) + "\\P3bcomponent_Interaction_Pz.jpg", dpi=400, bbox_inches='tight')
plt.savefig(str(plot_dir) + "\\P3bcomponent_Interaction_Pz.pdf", dpi=600, bbox_inches='tight')
plt.show()



#%%
# =============================================================================
# --- SUBPLOT INTERACTION P3B : ACTION-BASED PREDICTION & RELEVANCE ---
# =============================================================================


diff_rel   = (data_AR - data_PR)  * 1e6   # Action (Relevant)
diff_irrel = (data_AI - data_PI)  * 1e6   # Action (Irrelevant)
diff_act   = (data_AR - data_AI)  * 1e6   # Relevance (Active)
diff_pas   = (data_PR - data_PI)  * 1e6   # Relevance (Passive)

mean_rel, err_rel, mean_irrel, err_irrel = get_cousineau_morey_sem_diff(diff_rel, diff_irrel)
mean_act, err_act, mean_pas, err_pas     = get_cousineau_morey_sem_diff(diff_act, diff_pas)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(24, 20))

# -----------------------------------------------------------------------------
# GRAPH 1 (UP) : Relevance effect [(AR-AI) vs (PR-PI)]
# -----------------------------------------------------------------------------

ax1.plot(times_ms, mean_act, label="Active Rel. - Active Irrel.", color="black", linewidth=5)
ax1.fill_between(times_ms, mean_act - err_act, mean_act + err_act, 
                 facecolor="black", edgecolor="black", linestyle="-", linewidth=1.0, alpha=0.25)

ax1.plot(times_ms, mean_pas, label="Passive Rel. - Passive Irrel.", color="black", linestyle="--", linewidth=3)
ax1.fill_between(times_ms, mean_pas - err_pas, mean_pas + err_pas,
                 facecolor="black", edgecolor="black", linestyle="--", linewidth=1.2, alpha=0.15)

ax1.axvspan(p300_window[0], p300_window[1], facecolor='none', edgecolor='black', linestyle="-", linewidth=2.5)
ax1.axhline(0, color='grey', linewidth=1)
ax1.axvline(0, color="black", linestyle="--", linewidth=1)
ax1.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0)

ax1.set_xlim(-200, 800)
ax1.set_ylim(-2, 2)

ax1.set_title('', fontsize=0)
ax1.set_ylabel('Amplitude Diff. (µV)', fontsize=54)
ax1.tick_params(axis='y', labelsize=54)

ax1.tick_params(axis='x', labelbottom=False) 
ax1.legend(loc='upper left', fontsize=46)

# -----------------------------------------------------------------------------
# GRAPH2 (DOWN) : Action-based prediction effect [(AR-PR) vs (AI-PI)]
# -----------------------------------------------------------------------------

ax2.plot(times_ms, mean_rel, label="Active Rel. - Passive Rel.", color="black", linewidth=5)
ax2.fill_between(times_ms, mean_rel - err_rel, mean_rel + err_rel, 
                 facecolor="black", edgecolor="black", linestyle="-", linewidth=1.0, alpha=0.25)

ax2.plot(times_ms, mean_irrel, label="Active Irrel. - Passive Irrel.", color="black", linestyle="--", linewidth=3)
ax2.fill_between(times_ms, mean_irrel - err_irrel, mean_irrel + err_irrel,
                 facecolor="black", edgecolor="black", linestyle="--", linewidth=1.2, alpha=0.15)

ax2.axvspan(p300_window[0], p300_window[1], facecolor='none', edgecolor='black', linestyle="-", linewidth=2.5)
ax2.axhline(0, color='grey', linewidth=1)
ax2.axvline(0, color="black", linestyle="--", linewidth=1)
ax2.axvspan(baseline[0]*1000, baseline[1]*1000, color='gray', alpha=0.08, zorder=0)

ax2.set_xlim(-200, 800)
ax2.set_ylim(-2, 2)

ax2.set_title('', fontsize=0) # Titre masqué
ax2.set_ylabel('Amplitude Diff. (µV)', fontsize=54)
ax2.tick_params(axis='y', labelsize=54)

ax2.tick_params(axis='x', labelsize=54)
ax2.legend(loc='upper left', fontsize=46)
ax2.set_xlabel('Time (ms)', fontsize=54)

plt.tight_layout()

plt.savefig(str(plot_dir) + "\\P3bcomponent_Interactions_Subplots_Pz.jpg", dpi=400, bbox_inches='tight')
plt.savefig(str(plot_dir) + "\\P3bcomponent_Interactions_Subplots_Pz.pdf", dpi=600, bbox_inches='tight')
plt.show()

