# -*- coding: utf-8 -*-
"""
@author: Thomas Holstein 
thomas.holstein.pro@proton.me
"""

import arviz as az
import numpy as np
import pandas as pd
from pathlib import Path
import hssm
print(hssm.__version__)

#%% ______________________________ PATH MANAGEMENT _____________________________
current_path = Path(__file__).resolve()

for parent in current_path.parents:
    if parent.name == "STUDY":
        study_path = parent
        break
else:
    # Fix: Consistent error message with the 'STUDY' folder search
    raise FileNotFoundError("Could not find 'STUDY' folder in the directory tree.")

# Setup output and data directories
plot_dir = study_path / "Plots" / "mvpa"
plot_dir.mkdir(parents=True, exist_ok=True) 

bids_derivatives_dir = study_path / "Data" / "eeg" / "derivatives" / "epochs"
inputs_dir = study_path / "Code" / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True) 

beh_dir = study_path / "Data" / "beh"
beh_dir.mkdir(parents=True, exist_ok=True) 

outputs_dir = study_path / "Code" / "outputs"

#%% ______________________________ DATA LOADING & DIAGNOSTICS __________________
# Load the sampled posterior from the NetCDF file created in the previous script
nc_path = outputs_dir / "ddm_results.nc" 
model_all = az.from_netcdf(nc_path)   # Returns an InferenceData object
posterior = model_all.posterior        # Extract the posterior xarray Dataset

# Global summary of all parameters (Mean, HDI, etc.)
summary = az.summary(posterior, hdi_prob=0.95)

# Bayesian Probability: Calculate the proportion of the posterior distribution > 0
# This represents the probability that the effect of Agency_Condition is positive
v_bayesian_proba = np.mean(model_all.posterior['v_Agency_Condition'].values > 0)
a_bayesian_proba = np.mean(model_all.posterior['a_Agency_Condition'].values > 0)
t_bayesian_proba = np.mean(model_all.posterior['t_Agency_Condition'].values > 0)
z_bayesian_proba = np.mean(model_all.posterior['z_Agency_Condition'].values > 0)

print(f"Prob(v > 0): {v_bayesian_proba}")
print(f"Prob(a > 0): {a_bayesian_proba}")
print(f"Prob(t > 0): {t_bayesian_proba}")
print(f"Prob(z > 0): {z_bayesian_proba}")

# Convergence validation on the loaded results
rhat = az.rhat(model_all)
print("Max R-hat:", float(rhat.to_array().max()))

ess_bulk = az.ess(model_all, method="bulk")
print("Min bulk ESS:", float(ess_bulk.to_array().min()))

ess_tail = az.ess(model_all, method="tail")
print("Min tail ESS:", float(ess_tail.to_array().min()))

#%% _________________________ SUBJECT-LEVEL EXTRACTION _______________________
# Goal: Extract the posterior mean of each parameter per participant and save to CSV

# Filter summary table for participants' random effects using regex
# Pattern matches parameters like 'v_Agency_Condition|ID[0, 24]'
df_v = summary[summary.index.str.match(r"^v_Agency_Condition\|ID\[\d+, \d+\]$")].copy()
df_a = summary[summary.index.str.match(r"^a_Agency_Condition\|ID\[\d+, \d+\]$")].copy()
df_t = summary[summary.index.str.match(r"^t_Agency_Condition\|ID\[\d+, \d+\]$")].copy()
df_z = summary[summary.index.str.match(r"^z_Agency_Condition\|ID\[\d+, \d+\]$")].copy()

# Clean the index to keep only the participant ID (the second number in the ID[index, id] pair)
extract_last_number = lambda s: s.str.extract(r'ID\[\d+,\s*(\d+)\]')[0].astype(int)
df_v.index = extract_last_number(df_v.index)
df_a.index = extract_last_number(df_a.index)
df_t.index = extract_last_number(df_t.index)
df_z.index = extract_last_number(df_z.index)

# Export individual parameter files
df_v.to_csv(outputs_dir / "ddm_v.csv")
df_a.to_csv(outputs_dir / "ddm_a.csv")
df_t.to_csv(outputs_dir / "ddm_t.csv")
df_z.to_csv(outputs_dir / "ddm_z.csv")

# Merge all parameters into a single master CSV file for behavioral analysis
df_v = df_v.add_suffix('_v')
df_a = df_a.add_suffix('_a')
df_t = df_t.add_suffix('_t')
df_z = df_z.add_suffix('_z')

data_ddm = pd.concat([df_v, df_a, df_t, df_z], axis=1)
data_ddm.to_csv(outputs_dir / "ddm_data.csv")

#%% __________________________ DETAILED POSTERIOR ANALYSIS _____________________
# Method 2: Manual stacking of chains and draws for high-precision analysis
# This part ensures we are looking at the full distribution across all samples

# --- Drift rate (v) ---
# .stack(sample=("chain","draw")) collapses the MCMC dimensions into a single vector of samples
samps_v = posterior['v_C(Agency_Condition)'].stack(sample=("chain","draw")).values.ravel()
prob_pos_v = np.mean(samps_v > 0.0)
mean_post_v = samps_v.mean()
hdi95_v = az.hdi(samps_v, hdi_prob=0.95)
print(f"v -> Prob > 0: {prob_pos_v}, Mean: {mean_post_v}, 95% HDI: {hdi95_v}")

# --- Boundary separation (a) ---
samps_a = posterior['a_C(Agency_Condition)'].stack(sample=("chain","draw")).values.ravel()
prob_pos_a = np.mean(samps_a > 0.0)
mean_post_a = samps_a.mean()
hdi95_a = az.hdi(samps_a, hdi_prob=0.95)
print(f"a -> Prob > 0: {prob_pos_a}, Mean: {mean_post_a}, 95% HDI: {hdi95_a}")

# --- Non-decision time (t) ---
samps_t = posterior['t_C(Agency_Condition)'].stack(sample=("chain","draw")).values.ravel()
prob_pos_t = np.mean(samps_t > 0.0)
mean_post_t = samps_t.mean()
hdi95_t = az.hdi(samps_t, hdi_prob=0.95)
print(f"t -> Prob > 0: {prob_pos_t}, Mean: {mean_post_t}, 95% HDI: {hdi95_t}")

# --- Starting point (z) ---
samps_z = posterior['z_C(Agency_Condition)'].stack(sample=("chain","draw")).values.ravel()
prob_pos_z = np.mean(samps_z > 0.0)
mean_post_z = samps_z.mean()
hdi95_z = az.hdi(samps_z, hdi_prob=0.95)
print(f"z -> Prob > 0: {prob_pos_z}, Mean: {mean_post_z}, 95% HDI: {hdi95_z}")