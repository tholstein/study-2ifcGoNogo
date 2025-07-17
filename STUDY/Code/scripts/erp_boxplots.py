# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:03:22 2024

@author: Thomas Holstein
thomas.holstein.pro@proton.me

"""
#%% ________________________________ LIBRAIRIES _______________________________

import numpy as np
import pandas as pd
import mne
import matplotlib
# matplotlib.use('Qt5Agg') #block=True
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
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
plot_dir = study_path / "Plots" / "erp" / "boxplots"
plot_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Path of the epochs data in bids format
bids_derivatives_dir = study_path / "Data" / "eeg" / "derivatives" / "epochs"

# Path of the inputs for the electrodes selection
inputs_dir = study_path / "Code" / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

outputs_dir = study_path / "Code" / "outputs"


#%% ________________________________ FUNCTIONS ________________________________
def print_peak_measures(ch, tmin, tmax, lat, amp, condition=None):
    if ch == None:
        print("Composant absent")
    else:
        print(f'Condition: {condition}')
        print(f'Channel: {ch}')
        print(f'Time Window: {tmin:.3f} - {tmax:.3f} ms')
        print(f'Peak Latency: {lat:.3f} ms')
        print(f'Peak Amplitude: {amp:.3f} µV \n')


def annotate_significant_pairs(ax, data, x, y, order, component, electrode, pval_df):
    """
    Annotate significant condition pairs based on corrected p-values.
    
    Parameters:
        ax: matplotlib Axes object
        data: DataFrame with the data to plot
        x (str): column name for x-axis (usually 'Condition')
        y (str): column name for y-axis (usually 'Amplitude')
        order (list): order of categories on x-axis
        component (str): name of ERP component
        electrode (str): name of electrode
        pval_df (pd.DataFrame): DataFrame with columns ['Component', 'Electrode', 'Comparison', 'Corrected_p']
    """

    # Comparaisons to be considered
    pairs = [("Active Relevant", "Passive Relevant"),
             ("Active Relevant", "Active Irrelevant"), 
             ("Passive Relevant", "Passive Irrelevant"),
             ("Active Irrelevant", "Passive Irrelevant")]

    custom_pairs = []
    custom_annotations = []

    for cond1, cond2 in pairs:
        label = f"{cond1} vs {cond2}"

        row = pval_df[
            (pval_df["Component"] == component) &
            (pval_df["Electrode"] == electrode) &
            (pval_df["Comparison"] == label)
        ]

        if not row.empty:
            pval = row["Corrected_p"].values[0]
            if pd.isna(pval) or pval > 0.05:
                continue  # Ignoe non-significant comparisons 

            # Choose annotations
            if pval <= 0.0001:
                annotation = "****"
            elif pval <= 0.001:
                annotation = "***"
            elif pval <= 0.01:
                annotation = "**"
            else:
                annotation = "*"

            custom_pairs.append((cond1, cond2))
            custom_annotations.append(annotation)

    if custom_pairs:
        annotator = Annotator(ax, pairs=custom_pairs, data=data, x=x, y=y, order=order)
        annotator.set_custom_annotations(custom_annotations)
        annotator.configure(text_format='star', loc='inside', fontsize=32)
        annotator.annotate()



#%% ________________________________ CONSTANTS ________________________________
# Create a list of conditions
conditions = ["active/relevance",
              "passive/relevance",
              "active/irrelevance",
              "passive/irrelevance"]

# Time-windows
n1a_window = (0.100, 0.200)  # N1a (100 to 200ms in Anterior sites)
n170_window = (0.150, 0.200) # N170 (150 to 200ms in Posterior sites)
n2_window = (0.200, 0.350)  # N2 (200 to 250ms)
p1_window = (0.080, 0.200)  # P1 (80 to 200ms)
p2_window = (0.190, 0.295)  # P2 (190 to 280ms)
# p3_window = (0.350, 0.500)  # P3b (350 to 500ms)
# # If you want to analyse P3 on Cz and CPz component uncomment the line below
# # (because the temporal window changes for these electrodes compared to Pz)
p3_window = (0.300, 0.500)  # P3 (300 to 500ms)

time_windows = [n1a_window,
                n170_window,
                n2_window,
                p1_window,
                p2_window,
                p3_window]

n1a_electrod = ['FCz_cluster']
n170_electrod = ['Oz']
n2_electrod = ['Cz']
p1_electrod = ['P6']
p2_electrod = ['Oz']
p3_electrod = ['Cz']
components_sites = [n1a_electrod,
                    n170_electrod,
                    n2_electrod,
                    p1_electrod,
                    p2_electrod,
                    p3_electrod]

list_get_peak_all_components = [['N1a'],
                                ['N170'],
                                ['N2'],
                                ['P1'],
                                ['P2'],
                                ['P3']]

#%% _______________________________ MAIN SCRIPT________________________________

state_participant_error_dropping = False
erps_per_participant = []

list_of_individual_evoked_AR = []
list_of_individual_evoked_PR = []
list_of_individual_evoked_AI = []
list_of_individual_evoked_PI = []


# Loop over participants (from P002 to P006)
for PARTICIPANT_NUM in range(1, 25):
    
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
    baseline = (-0.2, 0) 
       
    AR = epochs["active/relevance"] # condition active relevance
    PR = epochs["passive/relevance"] # condition passive relevance
    AI = epochs["active/irrelevance"] # condition active irrelevance
    PI = epochs["passive/irrelevance"] # condition passive irrelevance
    
       
    evoked_all = epochs.average()
       
    evoked_AR = AR.average()
    list_of_individual_evoked_AR.append(evoked_AR)    
    evoked_PR = PR.average()
    list_of_individual_evoked_PR.append(evoked_PR) 
    evoked_AI = AI.average()
    list_of_individual_evoked_AI.append(evoked_AI)
    evoked_PI = PI.average()
    list_of_individual_evoked_PI.append(evoked_PI)   
    
    evoked_AR.comment = 'Active Relevant'
    evoked_PR.comment = 'Passive Relevant'
    evoked_AI.comment = 'Active Irrelevant'
    evoked_PI.comment = 'Passive Irrelevant'
    
    list_of_evoked_per_condition = [evoked_AR,
                                    evoked_PR,
                                    evoked_AI,
                                    evoked_PI]

    electrod_frontal_cluster = ['Fz', 'FCz', 'Cz'] 
    data_AR_frontal_cluster = [evoked_AR.copy().pick_channels(electrod_frontal_cluster).get_data()]
    data_PR_frontal_cluster = [evoked_PR.copy().pick_channels(electrod_frontal_cluster).get_data()]
    data_AI_frontal_cluster = [evoked_AI.copy().pick_channels(electrod_frontal_cluster).get_data()]
    data_PI_frontal_cluster = [evoked_PI.copy().pick_channels(electrod_frontal_cluster).get_data()]
    
    mean_AR_frontal_cluster = sum(data_AR_frontal_cluster[0]) / len(data_AR_frontal_cluster[0])
    mean_PR_frontal_cluster = sum(data_PR_frontal_cluster[0]) / len(data_PR_frontal_cluster[0])
    mean_AI_frontal_cluster = sum(data_AI_frontal_cluster[0]) / len(data_AI_frontal_cluster[0])
    mean_PI_frontal_cluster = sum(data_PI_frontal_cluster[0]) / len(data_PI_frontal_cluster[0])
    
    list_of_mean_per_condition = [mean_AR_frontal_cluster,
                                    mean_PR_frontal_cluster,
                                    mean_AI_frontal_cluster,
                                    mean_PI_frontal_cluster]
    
    new_channel_name = 'fronto_central_electrods'  # Nom de la nouvelle électrode
    for idx, condition, in enumerate(list_of_evoked_per_condition):
        # Create new electrod in evoked info
        ch_info = {'cal': 1.0, 'logno': len(condition.info['chs']) + 5,
                   'scanno': len(condition.info['chs']) + 1, 'range': 1.0, 
                   'unit_mul': condition.info['chs'][0]['unit_mul'],
                   'ch_name': new_channel_name, 
                   'unit': condition.info['chs'][0]['unit'],
                   'kind': condition.info['chs'][0]['kind'],
                   'coil_type': condition.info['chs'][0]['coil_type'],
                   'coord_frame': condition.info['chs'][0]['coord_frame'], 'loc': np.zeros(12)}
        condition.info['chs'].append(ch_info)
        condition.info._update_redundant()
        # Add data of the new electrode created (fronto-cetral cluster) 
        condition.data = np.vstack((condition.data, list_of_mean_per_condition[idx]))
        
    
    electrod_frontal_cluster = ['FCz','Fz','FC1','FC2','F1','F2'] 
    new_channel_name = 'FCz_cluster'
    data_AR_frontal_cluster = [evoked_AR.copy().pick_channels(electrod_frontal_cluster).get_data()]
    data_PR_frontal_cluster = [evoked_PR.copy().pick_channels(electrod_frontal_cluster).get_data()]
    data_AI_frontal_cluster = [evoked_AI.copy().pick_channels(electrod_frontal_cluster).get_data()]
    data_PI_frontal_cluster = [evoked_PI.copy().pick_channels(electrod_frontal_cluster).get_data()] 
    mean_AR_frontal_cluster = sum(data_AR_frontal_cluster[0]) / len(data_AR_frontal_cluster[0])
    mean_PR_frontal_cluster = sum(data_PR_frontal_cluster[0]) / len(data_PR_frontal_cluster[0])
    mean_AI_frontal_cluster = sum(data_AI_frontal_cluster[0]) / len(data_AI_frontal_cluster[0])
    mean_PI_frontal_cluster = sum(data_PI_frontal_cluster[0]) / len(data_PI_frontal_cluster[0])    
    list_of_mean_per_condition = [mean_AR_frontal_cluster,
                                    mean_PR_frontal_cluster,
                                    mean_AI_frontal_cluster,
                                    mean_PI_frontal_cluster] 
    for idx, condition, in enumerate(list_of_evoked_per_condition):
        # Create new electrod in evoked info
        ch_info = {'cal': 1.0, 'logno': len(condition.info['chs']) + 6,
                   'scanno': len(condition.info['chs']) + 1, 'range': 1.0, 
                   'unit_mul': condition.info['chs'][0]['unit_mul'],
                   'ch_name': new_channel_name, 
                   'unit': condition.info['chs'][0]['unit'],
                   'kind': condition.info['chs'][0]['kind'],
                   'coil_type': condition.info['chs'][0]['coil_type'],
                   'coord_frame': condition.info['chs'][0]['coord_frame'], 'loc': np.zeros(12)}
        condition.info['chs'].append(ch_info)
        condition.info._update_redundant()
        # Add data of the new electrode created (fronto-cetral cluster) 
        condition.data = np.vstack((condition.data, list_of_mean_per_condition[idx]))


    for i_component, t_window in enumerate(time_windows):
        if t_window == n1a_window or t_window == n170_window or t_window == n2_window:
            mode = 'neg'
        else:
            mode = 'pos'
        for idx, condition, in enumerate(list_of_evoked_per_condition):    
            if list_get_peak_all_components[i_component][0] == 'N1a':
                site = condition.copy().pick(n1a_electrod)
            elif list_get_peak_all_components[i_component][0] == 'N170':
                site = condition.copy().pick(n170_electrod)
            elif list_get_peak_all_components[i_component][0] == 'N2':
                site = condition.copy().pick(n2_electrod)
            elif list_get_peak_all_components[i_component][0] == 'P1':
                site = condition.copy().pick(p1_electrod)
            elif list_get_peak_all_components[i_component][0] == 'P2':
                site = condition.copy().pick(p2_electrod)
            elif list_get_peak_all_components[i_component][0] == 'P3':
                site = condition.copy().pick(p3_electrod) #P3
         
            try: 
                ch, lat, amp = site.get_peak(ch_type='eeg', tmin=t_window[0], tmax=t_window[1],
                                            mode=mode, return_amplitude=True)  
                lat = lat *1e3
                amp = round(amp * 1e6,2)
            
            except ValueError:
                print(f"\n /!\ No values encountered for the component : {list_get_peak_all_components[i_component][0]} /!\ \n")
                
                # Get epoch data for the window of interest
                epoch_data = site.get_data()[:, int(t_window[0]*epochs.info['sfreq']):int(t_window[1]*epochs.info['sfreq'])]
                ch = site.ch_names[0]
                
                if np.any(epoch_data > 0):  # Check if there are any positive values in the epoch data
                    # Calculate the minimum positive value and corresponding latency
                    amp = np.min(epoch_data[epoch_data > 0]) * 1e6  
                    lat = (np.argmin(epoch_data) / epochs.info['sfreq'] + t_window[0]) * 1e3
                elif np.any(epoch_data < 0):  # Check if there are any negative values in the epoch data
                    # Calculate the maximum negative value and corresponding latency
                    amp = np.max(epoch_data[epoch_data < 0]) * 1e6  
                    lat = (np.argmax(epoch_data) / epochs.info['sfreq'] + t_window[0]) * 1e3  
                else:
                    # If there are no positive or negative values, set latency and amplitude to None
                    lat = None
                    amp = None
                pass
            list_get_peak_all_components[i_component].append([t_window, PARTICIPANT_NUM, condition.comment, ch, lat, amp])
            print_peak_measures(ch, t_window[0], t_window[1], lat, amp, condition.comment) 
    
    # Restructure data ERP component     
    data_ERP_components = []
    for row in list_get_peak_all_components:
        component = row[0]
        for item in row[1:]:
            data_ERP_components.append({'Component': component, 
                                       'Time-window': (int(item[0][0]*1e3),int(item[0][1]*1e3)),
                                       'Participant': item[1],
                                       'Condition': item[2], 
                                       'Electrode': item[3], 
                                       'Latence (ms)': item[4], 
                                       'Amplitude': item[5]}) # in microvolts


data_ERP_components = pd.DataFrame(data_ERP_components)

col=list(data_ERP_components.columns)
data_ERP_components[['Agency', 'Task-relevance']] = data_ERP_components['Condition'].str.split(' ', 1, expand=True)

N1a = data_ERP_components.loc[data_ERP_components['Component'] == 'N1a']
N170 = data_ERP_components.loc[data_ERP_components['Component'] == 'N170']
N2 = data_ERP_components.loc[data_ERP_components['Component'] == 'N2']
P1 = data_ERP_components.loc[data_ERP_components['Component'] == 'P1']
P2 = data_ERP_components.loc[data_ERP_components['Component'] == 'P2']
P3 = data_ERP_components.loc[data_ERP_components['Component'] == 'P3']
Data = [N1a, N170, N2, P1, P2, P3]


# -------------------- Descriptive statistics initialization ------------------

descriptive_statistics_AR = [['N1a'],['N170'],
                             ['N2'],['P1'],
                             ['P2'],['P3']]
descriptive_statistics_PR = [['N1a'],['N170'],
                             ['N2'],['P1'],
                             ['P2'],['P3']]
descriptive_statistics_AI = [['N1a'],['N170'],
                             ['N2'],['P1'],
                             ['P2'],['P3']]
descriptive_statistics_PI = [['N1a'],['N170'],
                             ['N2'],['P1'],
                             ['P2'],['P3']]


descriptive_statistics_A = [['N1a'],['N170'],
                             ['N2'],['P1'],
                             ['P2'],['P3']]

descriptive_statistics_P = [['N1a'],['N170'],
                             ['N2'],['P1'],
                             ['P2'],['P3']]

descriptive_statistics_R = [['N1a'],['N170'],
                             ['N2'],['P1'],
                             ['P2'],['P3']]

descriptive_statistics_I = [['N1a'],['N170'],
                             ['N2'],['P1'],
                             ['P2'],['P3']]

values_of_components = []

for values in Data: 
    value = (
        values['Amplitude'].loc[values['Condition'] == 'Active Relevant'], #0
        values['Amplitude'].loc[values['Condition'] == 'Passive Relevant'], #1
        values['Amplitude'].loc[values['Condition'] == 'Active Irrelevant'], #2
        values['Amplitude'].loc[values['Condition'] == 'Passive Irrelevant'] #3
    )
    values_of_components.append(value)
    
values_of_components_ActivePassive_and_RelevantIrrelevant = []

for values in Data: 
    value = (
        values['Amplitude'].loc[values['Agency'] == 'Active'], #0
        values['Amplitude'].loc[values['Agency'] == 'Passive'], #1
        values['Amplitude'].loc[values['Task-relevance'] == 'Relevant'], #2
        values['Amplitude'].loc[values['Task-relevance'] == 'Irrelevant'], #3
    )
    values_of_components_ActivePassive_and_RelevantIrrelevant.append(value)
    
      
#%% ________________________________ Box-plots ________________________________

# Load csv file for corrected p-values from R script
pval_df = pd.read_csv(inputs_dir / "posthoc_corrected_pvalues.csv")  # modifie le chemin
pval_df["Comparison"] = pval_df["Comparison"].str.strip()  


for i, data in enumerate(Data):
    # ------------------------- Main effect comparisions ---------------------- 
    # ---------------------------- Agency main effect ------------------------- 
    # Collapsing Task-relevance => moyenne de chaque participant pour chaque Agency
    agency_means = (
        data.groupby(['Participant', 'Agency'])['Amplitude']
        .mean()
        .reset_index()
    )
    
    plt.figure(figsize=(16, 12))
    order = ['Active', 'Passive']
    x = 'Agency'
    y = 'Amplitude'
    
    ax = sns.boxplot(x=x, y=y, data=agency_means, order=order,
                     palette=('pink', 'skyblue'), showfliers=False)
    
    sns.swarmplot(x=x, y=y, data=agency_means, order=order,
                  color='black', size=10, alpha=0.5)
    
    ax.set_title(f"{list_get_peak_all_components[i][0]} on {list_get_peak_all_components[i][1][3]} electrode", fontsize=40)
    ax.set_xlabel("Agency", fontsize=0)
    ax.set_ylabel("Amplitude (µV)", fontsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.tick_params(axis='x', labelsize=32)
    # plt.savefig(str(plot_dir) + "\\agency_plot_{list_get_peak_all_components[i][0]}_{list_get_peak_all_components[i][1][3]}.jpg")  # Sauvegarde au format PNG

    # -------------------------- Relevance main effect ------------------------ 
    # Collapsing Agency => moyenne de chaque participant pour chaque Task-relevance
    relevance_means = (
        data.groupby(['Participant', 'Task-relevance'])['Amplitude']
        .mean()
        .reset_index()
    )
    
    plt.figure(figsize=(16, 12))
    order = ['Relevant', 'Irrelevant']
    x = 'Task-relevance'
    y = 'Amplitude'
    
    ax = sns.boxplot(x=x, y=y, data=relevance_means, order=order,
                     palette=('pink', 'skyblue'), showfliers=False)
    
    sns.swarmplot(x=x, y=y, data=relevance_means, order=order,
                  color='black', size=10, alpha=0.5)
    
    ax.set_title(f"{list_get_peak_all_components[i][0]} on {list_get_peak_all_components[i][1][3]} electrode", fontsize=40)
    ax.set_xlabel("Relevance", fontsize=0)
    ax.set_ylabel("Amplitude (µV)", fontsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.tick_params(axis='x', labelsize=32)
    # plt.savefig(str(plot_dir) + "\\relevance_plot_{list_get_peak_all_components[i][0]}_{list_get_peak_all_components[i][1][3]}.jpg")  # Sauvegarde au format PNG
    
    # -------------------------- Post-hoc comparisons ------------------------- 
    order = ['Active Relevant', 'Passive Relevant', 'Active Irrelevant', 'Passive Irrelevant']
    x='Condition'
    y='Amplitude' 
    plt.figure(figsize=(24, 12))
    order = ['Active Relevant', 'Passive Relevant', 'Active Irrelevant', 'Passive Irrelevant']
    ax = sns.boxplot(x=x, y=y, data=data, 
                palette=('red', 'orange', '#263f8a', 'skyblue'),
                order=['Active Relevant', 'Passive Relevant', 'Active Irrelevant', 'Passive Irrelevant'],
                showfliers=False, #to place outliers within the boxplot and not separately,             
                )      
    
    sns.swarmplot(x=x, y=y, data=data, order=order, edgecolor='white', linewidth=2,
              color='black', size=12, dodge=False, alpha=0.8)
    
    ax.set_title(f"{list_get_peak_all_components[i][0]} on {list_get_peak_all_components[i][1][3]} electrode", fontsize=40)  # Ajuster la taille de la police du titre
    ax.set_xlabel("Condition", fontsize=0)       # Ajuster la taille de la police du label de l'axe x
    ax.set_ylabel("Amplitude (µV)", fontsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.tick_params(axis='x', labelsize=32)
    
    pairs=[("Active Relevant", "Passive Relevant"), ("Active Relevant", "Active Irrelevant"), 
           ("Passive Relevant", "Passive Irrelevant"), ("Active Irrelevant", "Passive Irrelevant")]
    
    # Determine the name of the current component and electrode
    component = list_get_peak_all_components[i][0]
    electrode = list_get_peak_all_components[i][1][3]
    
    # Annotations (asterisks if significant)
    annotate_significant_pairs(ax, data, x, y, order, component, electrode, pval_df)
    # Save 
    plt.savefig(f"{plot_dir}/all_conditions_plot_{list_get_peak_all_components[i][0]}_{list_get_peak_all_components[i][1][3]}.jpg")
    
    # ------------------------- Descriptive statistics ------------------------
    descriptive_statistics_AR[i].append(data.loc[data['Condition'] == 'Active Relevant', 'Amplitude'].describe())
    descriptive_statistics_PR[i].append(data.loc[data['Condition'] == 'Passive Relevant', 'Amplitude'].describe())
    descriptive_statistics_AI[i].append(data.loc[data['Condition'] == 'Active Irrelevant', 'Amplitude'].describe())
    descriptive_statistics_PI[i].append(data.loc[data['Condition'] == 'Passive Irrelevant', 'Amplitude'].describe())
    
    descriptive_statistics_A[i].append(data.loc[data['Agency'] == 'Active', 'Amplitude'].describe())
    descriptive_statistics_P[i].append(data.loc[data['Agency'] == 'Passive', 'Amplitude'].describe())
    descriptive_statistics_R[i].append(data.loc[data['Task-relevance'] == 'Relevant', 'Amplitude'].describe())
    descriptive_statistics_I[i].append(data.loc[data['Task-relevance'] == 'Irrelevant', 'Amplitude'].describe())


#%% ______________________________ Violin plots _______________________________

for i, data in enumerate(Data):
    # Color palette
    palette_conditions = {'Active Relevant': 'red', 'Passive Relevant': 'orange',
                          'Active Irrelevant': '#263f8a', 'Passive Irrelevant': 'skyblue'}
    palette_agency = {'Active': 'pink', 'Passive': 'skyblue'}
    palette_relevance = {'Relevant': 'pink', 'Irrelevant': 'skyblue'}
    
    # ------------------------- Main effect comparisions ---------------------- 
    # ---------------------------- Agency main effect -------------------------
    # Compute mean amplitude per participant per Agency level
    agency_means = (
        data.groupby(['Participant', 'Agency'])['Amplitude']
        .mean()
        .reset_index()
    )
    
    plt.figure(figsize=(16, 12))
    order_agency = ['Active', 'Passive']
    x = 'Agency'
    y = 'Amplitude'
    ax = sns.violinplot(x=x, y=y, data=agency_means, order=order_agency, palette=palette_agency,
                        inner=None, linewidth=4)
    
    # Add a line for the group median
    for j, group in enumerate(order_agency):
        median_val = agency_means[agency_means['Agency'] == group][y].median()
        ax.plot([j - 0.4, j + 0.4], [median_val, median_val], color='black', linewidth=4)
    
    # Add individual participant data points
    sns.swarmplot(x=x, y=y, data=agency_means, order=order_agency,
                  color='black', size=12, dodge=False, alpha=0.6, ax=ax)
    
    ax.set_title(f"{list_get_peak_all_components[i][0]} on {list_get_peak_all_components[i][1][3]} electrode", fontsize=40)
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("Amplitude (µV)", fontsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.tick_params(axis='x', labelsize=32)
    # plt.savefig(str(plot_dir) + "\\agency_plot_violin{list_get_peak_all_components[i][0]}_{list_get_peak_all_components[i][1][3]}.png")
    
    # -------------------------- Relevance main effect ------------------------ 
    # Compute mean amplitude per participant per Task-Relevance level
    relevance_means = (
        data.groupby(['Participant', 'Task-relevance'])['Amplitude']
        .mean()
        .reset_index()
    )
    
    plt.figure(figsize=(16, 12))
    order_relevance = ['Relevant', 'Irrelevant']
    x = 'Task-relevance'
    y = 'Amplitude'
    ax = sns.violinplot(x=x, y=y, data=relevance_means, order=order_relevance, palette=palette_relevance,
                        inner=None, linewidth=2)
    
    # Add a line for the group median
    for j, group in enumerate(order_relevance):
        median_val = relevance_means[relevance_means['Task-relevance'] == group][y].median()
        ax.plot([j - 0.4, j + 0.4], [median_val, median_val], color='black', linewidth=4)
    
    # Add individual participant data points
    sns.swarmplot(x=x, y=y, data=relevance_means, order=order_relevance,
                  color='black', size=12, dodge=False, alpha=0.6, ax=ax)
    
    ax.set_title(f"{list_get_peak_all_components[i][0]} on {list_get_peak_all_components[i][1][3]} electrode", fontsize=40)
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("Amplitude (µV)", fontsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.tick_params(axis='x', labelsize=32)
    # plt.savefig(str(plot_dir) + "\\relevance_plot_violin{list_get_peak_all_components[i][0]}_{list_get_peak_all_components[i][1][3]}.png")
    
    
    # -------------------------- Post-hoc comparisons ------------------------- 
    plt.figure(figsize=(24, 12))
    order = ['Active Relevant', 'Passive Relevant', 'Active Irrelevant', 'Passive Irrelevant']
    x = 'Condition'
    y = 'Amplitude'
    
    ax = sns.violinplot(x=x, y=y, data=data, order=order, palette=palette_conditions,
                        inner=None, linewidth=2)
    
    # Adds a line for the median
    for j, group in enumerate(order):
        median_val = data[data[x] == group][y].median()
        ax.plot([j - 0.4, j + 0.4], [median_val, median_val], color='black', linewidth=4)
    
    # Dots
    sns.swarmplot(x=x, y=y, data=data, order=order, color='white', size=12, dodge=False, alpha=0.9, ax=ax)

    # Display setup
    ax.set_title(f"{list_get_peak_all_components[i][0]} on {list_get_peak_all_components[i][1][3]} electrode", fontsize=40)
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("Amplitude (µV)", fontsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.tick_params(axis='x', labelsize=32)

    # Determine the name of the current component and electrode
    component = list_get_peak_all_components[i][0]
    electrode = list_get_peak_all_components[i][1][3]
    
    # Annotations (asterisks if significant)
    annotate_significant_pairs(ax, data, x, y, order, component, electrode, pval_df)
    # plt.savefig(str(plot_dir) + "\\all_conditions_plot_violin{list_get_peak_all_components[i][0]}_{list_get_peak_all_components[i][1][3]}.png")

    
