# -*- coding: utf-8 -*-
"""
@author: Thomas Holstein 
thomas.holstein.pro@proton.me
"""

import scipy
print(scipy.__version__)
import pandas as pd
from pathlib import Path
import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt
import arviz as az
import hssm
print(hssm.__version__)
print("HSSM works!")

#%% ______________________________ PATH MANAGEMENT _____________________________
# Dynamic path resolution to ensure portability across different machines
current_path = Path(__file__).resolve()

# Locate the project root by searching for the 'STUDY' directory in the hierarchy
for parent in current_path.parents:
    if parent.name == "STUDY":
        study_path = parent
        break
else:
    # Note: Ensure the error message matches the searched folder name ('STUDY')
    raise FileNotFoundError("Could not find 'STUDY' folder in the directory tree.")

# Initialize project directory structure
plot_dir = study_path / "Plots" / "mvpa"
plot_dir.mkdir(parents=True, exist_ok=True) 

bids_derivatives_dir = study_path / "Data" / "eeg" / "derivatives" / "epochs"
inputs_dir = study_path / "Code" / "inputs"
inputs_dir.mkdir(parents=True, exist_ok=True)

beh_dir = study_path / "Data" / "beh"
beh_dir.mkdir(parents=True, exist_ok=True)

outputs_dir = study_path / "Code" / "outputs"

#%% ______________________________ DATA PREPROCESSING ___________________________
beh_path = beh_dir / "beh_data.csv"
my_data = pd.read_csv(beh_path)

# Clean dataset: remove missing values in critical columns and standardize naming
my_data = my_data.dropna(subset=['Reaction_Time', 'Discrimination'])
my_data = my_data.rename(columns={
    "Reaction_Time": "rt", 
    "Discrimination": "response", 
    "Participant": "ID"
})

# Recode conditions: Agency_Condition (A=1, P=0) and Response (0 to -1 for DDM compatibility)
my_data['Agency_Condition'] = my_data['Agency_Condition'].replace({'A': 1, 'P': 0})
my_data['response'] = my_data['response'].replace(0, -1)

# Define Agency_Condition as an ordered categorical variable for the model formula
my_data['Agency_Condition'] = pd.Categorical(my_data['Agency_Condition'], categories=[0,1], ordered=True)


#%% ______________________________ DDM PROCESSING ___________________________

# HSSM Model Definition: Drift Diffusion Model (DDM)
# Global formula specifies a random intercept and slope per participant (ID) for Agency_Condition
model = hssm.HSSM(
    data=my_data,
    model="ddm",  
    prior_settings="safe", # Use weakly informative priors to avoid convergence issues
    noncentered=True,       # Non-centered parametrization to reduce divergent transitions
    global_formula='y ~ Agency_Condition + (1 + Agency_Condition | ID)',
)

# Bayesian Sampling execution
if __name__ == "__main__":
    # Sample from the posterior distribution using NUTS
    model_all = model.sample(draws=2000, tune=2000, chains=4, target_accept=0.95, max_treedepth=12)
    # Export results to NetCDF format for long-term storage and ArviZ compatibility
    model_all.to_netcdf(outputs_dir / "ddm_results.nc")

#%% ______________________________ POSTERIOR ANALYSIS ___________________________
# Extract InferenceData object from the model results
idata = model_all 

# Generate summary statistics: Mean, HDI (Highest Density Interval), R-hat, and ESS
summary = az.summary(idata, round_to=3)
print(summary)

# Convergence Diagnostics
rhat = az.rhat(idata)
print("Max R-hat:", float(rhat.to_array().max())) # Values > 1.05 indicate convergence issues

# Effective Sample Size (ESS)
# Bulk ESS: Stability of the mean/median
ess_bulk = az.ess(idata, method="bulk")
print("Min bulk ESS:", float(ess_bulk.to_array().min()))

# Tail ESS: Reliability of the distribution tails (crucial for credible intervals)
ess_tail = az.ess(idata, method="tail")
print("Min tail ESS:", float(ess_tail.to_array().min()))

# Visualization: MCMC Traceplots to verify chain mixing and stationarity
az.plot_trace(idata)
plt.tight_layout()
plt.savefig(outputs_dir / "traceplots_ddm.png", dpi=600)

# Visualization: Pair plots to inspect correlations between key DDM parameters (v, a, t, z)
az.plot_pair(idata, var_names=["~v","~a","~t","~z"], divergences=True, kind="kde")
plt.tight_layout()
plt.savefig(outputs_dir / "pairs_ddm.png", dpi=600)

# Posterior Predictive Checks (PPC): Compare simulated data with observed data
if hasattr(idata, "posterior_predictive") and idata.posterior_predictive is not None:
    az.plot_ppc(idata, group="posterior_predictive")
    plt.tight_layout()
    plt.savefig(outputs_dir / "ppc_ddm.png", dpi=200)
else:
    print("No posterior_predictive found in idata.")

# Filter summary to display only parameters related to the drift rate (v) and the experimental condition
print(summary.loc[[name for name in summary.index if "v" in name or "Agency_Condition" in name]])