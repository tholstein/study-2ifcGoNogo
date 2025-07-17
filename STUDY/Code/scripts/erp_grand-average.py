# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:03:22 2024

@author: Thomas Holstein
thomas.holstein.pro@proton.me

"""
#%% ________________________________ LIBRAIRIES _______________________________
import os
import numpy as np
import mne
# matplotlib.use('Qt5Agg') #block=True
import matplotlib.pyplot as plt

import autoreject
from autoreject import get_rejection_threshold
#from autoreject import RejectLog
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
plot_dir = study_path / "Plots" / "erp" / "grand_average"
plot_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Path of the epochs data in bids format
bids_derivatives_dir = study_path / "Data" / "eeg" / "derivatives" / "epochs"

# Path of the inputs for the electrodes selection
inputs_dir = study_path / "Code" / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

outputs_dir = study_path / "Code" / "outputs"

#%% _______________________________ MAIN SCRIPT________________________________

state_participant_error_dropping = False
erps_per_participant = []

list_of_individual_evoked_AR = []
list_of_individual_evoked_PR = []
list_of_individual_evoked_AI = []
list_of_individual_evoked_PI = []

list_of_individual_evoked_AR_LEFT = []
list_of_individual_evoked_PR_LEFT  = []
list_of_individual_evoked_AI_LEFT  = []
list_of_individual_evoked_PI_LEFT  = []

list_of_individual_evoked_AR_RIGHT = []
list_of_individual_evoked_PR_RIGHT  = []
list_of_individual_evoked_AI_RIGHT  = []
list_of_individual_evoked_PI_RIGHT  = []

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
mne.viz.plot_compare_evokeds(list_of_grand_average, picks=parieto_occipital_cluster_1, colors=('red', 'orange', 'blue', 'darkturquoise'), title='O1, PO3, PO7- Grand average (n=24)',
                             show_sensors=True, combine='mean', ylim=dict(eeg=[-limit, limit]))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks=parieto_occipital_cluster_2, colors=('red', 'orange', 'blue', 'darkturquoise'), title='O2, PO4, PO8 - Grand average (n=24)',
                             show_sensors=True, combine='mean', ylim=dict(eeg=[-limit, limit]))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks=parieto_occipital_cluster_L_2, colors=('red', 'orange', 'blue', 'darkturquoise'), title='O1, PO3, PO7- Grand average (n=24)',
                             show_sensors=True, combine='mean', ylim=dict(eeg=[-limit, limit]))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks=parieto_occipital_cluster_R_2, colors=('red', 'orange', 'blue', 'darkturquoise'), title='O2, PO4, PO8 - Grand average (n=24)',
                             show_sensors=True, combine='mean', ylim=dict(eeg=[-limit, limit]))

"""_______________________________ P6 electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='P6', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False, # /!\ très important /!\
                             time_unit='ms',
                             legend=False,
                             axes=ax)
# Add the shaded gray area
ax.axvspan(p1_window[0], p1_window[1], color='lightgray', alpha=0.3)
ax.set_title('P6 electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  # Add horizontal line y=0
ax.set_xlim(-200, 800)  # Adjust limits of x axis if necessary 
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout()  # If the layout is not good
# If you want to add the asterisk on peak component
peak_x = np.mean(p1_window)  # For instance, we take the middle of the window
peak_y = -1.1  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
            # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
            # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P1component_P6.jpg")  # Save in JPG format
plt.show()

#%% ______________________________ N1a component ______________________________
"""_______________________________ AFz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='AFz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 2]),
                             vlines=[0],
                             combine='mean',
                             show_sensors='lower right',
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             legend=False,
                             axes=ax)
# Add the shaded gray area
ax.axvspan(n1a_window[0], n1a_window[1], color='lightgray', alpha=0.3)
# Personnalisation supplémentaire de l'axe si nécessaire
ax.set_title('AFz', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3) # Add horizontal line y=0
ax.set_xlim(-200, 800)  # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n1a_window)  
peak_y = -2  # Height above peak where to place asterisk
# ax.annotate('ns', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N1acomponent_AFz.jpg") # Save in JPG format
plt.show()

"""___________________________ Fronto-central cluster _________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks=fronto_central_cluster, colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 2]),
                             vlines=[0],
                             combine='mean',
                             show_sensors='lower right',
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             legend=False,
                             axes=ax)
# Add the shaded gray area
ax.axvspan(n1a_window[0], n1a_window[1], color='lightgray', alpha=0.3)
# Personnalisation supplémentaire de l'axe si nécessaire
ax.set_title('Fronto-central cluster', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3) # Add horizontal line y=0
ax.set_xlim(-200, 800)  # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n1a_window)  
peak_y = -2  # Height above peak where to place asterisk
# ax.annotate('ns', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))

plt.savefig(str(plot_dir) + "\\N1acomponent_Fronto-central-cluster.jpg") # Save in JPG format
plt.show()


#%% ______________________________ P2 component _______________________________
"""_______________________________ Oz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='Oz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, limit]),
                             vlines=[0],
                             combine='mean',
                             show_sensors=True,
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             axes=ax)
# Add the shaded gray area
ax.axvspan(p2_window[0], p2_window[1], color='lightgray', alpha=0.3)
ax.set_title('Oz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  # Add horizontal line y=0
ax.set_xlim(-200, 800)   # Adjust limits of x axis if necessary 
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout()  
# If you want to add the asterisk on peak component
peak_x = np.mean(p2_window)  
peak_y = 7  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P2component_Oz.jpg")  # Save in JPG format
plt.show()

"""_______________________________ POz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='POz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, limit]),
                             vlines=[0],
                             combine='mean',
                             show_sensors=True,
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             axes=ax)
# Add the shaded gray area
ax.axvspan(p2_window[0], p2_window[1], color='lightgray', alpha=0.3)
ax.set_title('POz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  # Add horizontal line y=0
ax.set_xlim(-200, 800)   # Adjust limits of x axis if necessary 
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout()  
# If you want to add the asterisk on peak component
peak_x = np.mean(p2_window)  
peak_y = 7  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P2component_POz.jpg")  # Save in JPG format
plt.show()


#%% ______________________________ N2b component ______________________________
"""_______________________________ Cz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='Cz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             show_sensors=True,
                             axes=ax)
# Add the shaded gray area
ax.axvspan(n2_window[0], n2_window[1], color='lightgray', alpha=0.3)
ax.set_title('Cz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  # Add horizontal line y=0
ax.set_xlim(-200, 800) # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n2_window)  
peak_y = -3  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N2bcomponent_Cz.jpg")  # Save in JPG format
plt.show()

"""_______________________________ FCz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='FCz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             show_sensors=True,
                             axes=ax)
# Add the shaded gray area
ax.axvspan(n2_window[0], n2_window[1], color='lightgray', alpha=0.3)
# ax.axvspan( p300_window[0], p300_window[1], color='lightgray', alpha=0.3)
ax.set_title('FCz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  # Ajouter une ligne horizontale à y=0
ax.set_xlim(-200, 800)  # Ajuster les limites de l'axe x si nécessaire
# ax.set_ylim(-4, 3)  # Ajuster les limites de l'axe x si nécessaire

ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n2_window)  
peak_y = -3 # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N2bcomponent_FCz.jpg")  # Save in JPG format
plt.show()

"""_______________________________ Fz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='Fz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             show_sensors=True,
                             axes=ax)
# Add the shaded gray area
ax.axvspan(n2_window[0], n2_window[1], color='lightgray', alpha=0.3)
ax.set_title('Fz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  # Add horizontal line y=0
ax.set_xlim(-200, 800) # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n2_window)  
peak_y = -3  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N2bcomponent_Fz.jpg")  # Save in JPG format
plt.show()

"""_______________________________ AFz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='AFz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             show_sensors=True,
                             axes=ax)
# Add the shaded gray area
ax.axvspan(n2_window[0], n2_window[1], color='lightgray', alpha=0.3)
ax.set_title('AFz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  # Add horizontal line y=0
ax.set_xlim(-200, 800) # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(n2_window)  
peak_y = -3  # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\N2bcomponent_AFz.jpg")  # Save in JPG format
plt.show()

#%% ______________________________ P3b component ______________________________
"""________________________________Pz electrod ________________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='Pz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-3, 6]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             show_sensors=True,
                             axes=ax)
# Add the shaded gray area
ax.axvspan(p300_window[0], p300_window[1], color='lightgray', alpha=0.3)
ax.set_title('Pz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)   # Add horizontal line y=0
ax.set_xlim(-200, 800)  # Adjust limits of x axis if necessary 
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# If you want to add the asterisk on peak component
peak_x = np.mean(p300_window)  # For instance, we take the middle of the window
peak_y = 4.75 # Height above peak where to place asterisk
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=18, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P3bcomponent_Pz.jpg")  # Save in JPG format
plt.show()

"""________________________________CPz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='CPz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             show_sensors='lower right',
                             axes=ax)
# Add the shaded gray area
ax.axvspan(p300_window[0], p300_window[1], color='lightgray', alpha=0.3)
ax.set_title('CPz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  
ax.set_xlim(-200, 800)  
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# peak_x = np.mean(p300_window)  
# peak_y = 3.5  
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P3bcomponent_CPz.jpg")  # Save in JPG format
plt.show()

"""_________________________________Cz electrod _______________________________"""
fig, ax = plt.subplots(figsize=(24, 20))
mne.viz.plot_compare_evokeds(list_of_grand_average, picks='Cz', colors=('red', 'orange', 'blue', 'darkturquoise'),
                             ylim=dict(eeg=[-4, 4]),
                             vlines=[0],
                             styles={"Active Relevant": {"linewidth": 10},
                                     "Active Irrelevant": {"linewidth": 10},
                                     "Passive Relevant": {"linewidth": 10},
                                     "Passive Irrelevant": {"linewidth": 10}},
                             show=False,
                             time_unit='ms',
                             show_sensors='lower right',
                             axes=ax)
# Add the shaded gray area
ax.axvspan(p300_window2[0], p300_window2[1], color='lightgray', alpha=0.3)
ax.set_title('Cz electrode', fontsize=0)
ax.set_xlabel('Time (ms)', fontsize=32)
ax.set_ylabel('Amplitude (µV)', fontsize=32)
ax.axhline(0, color='black', linewidth=3)  
ax.set_xlim(-200, 800)  
# ax.set_ylim(-4, 3)  
ax.legend(loc='upper left', fontsize=34)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
# plt.tight_layout() 
# peak_x = np.mean(p300_window)  
# peak_y = 3.5  
# ax.annotate('*', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.1),
#             fontsize=16, ha='center', va='center')
#             # bbox=dict(boxstyle='circle,pad=0.3', fc=None, ec=None, lw=1),
#             # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', lw=1))
plt.savefig(str(plot_dir) + "\\P3bcomponent_Cz.jpg")  # Save in JPG format
plt.show()




