#====================== LIBRAIRIES =========================
rm(list=ls())
library(kableExtra)
library(car)
library(rstatix)
library(BayesFactor)
library(ggpubr)
library(psych)
library(tidyverse)
library(RColorBrewer)
library(here)
library(readr)
library(readxl)
library(dplyr)

#====================== PATH MANAGMENT =======================
# Find the absolute path of the current script
get_script_path <- function() {
  if (!is.null(sys.frame(1)$ofile)) {
    return(normalizePath(dirname(sys.frame(1)$ofile)))
  } else if (requireNamespace("rstudioapi", quietly = TRUE)) {
    return(normalizePath(dirname(rstudioapi::getActiveDocumentContext()$path)))
  } else {
    stop("Cannot determine script path. Please run this from an R script or RStudio.")
  }
}
# Get the script path
script_path <- get_script_path()
# Normalize and split
script_path <- normalizePath(script_path, winslash = "/")  # Make sure it's in Unix style
# Split the path and find the "STUDY" directory
path_parts <- strsplit(script_path, "/")[[1]]
# Try to find folder
osf_index <- which(path_parts == "STUDY")

if (length(osf_index) == 0) {
  stop("'STUDY' folder not found in the path.")
}
# Reconstruct the path to STUDY
osf_root <- do.call(file.path, as.list(path_parts[1:osf_index]))
# Build the path to the 'inputs' folder
inputs_dir <- file.path(osf_root, "Data", "beh")
outputs_dir <- file.path(osf_root, "Code", "outputs")
# Build the path to the 'Plots' folder
plots_dir <- file.path(osf_root, "Plots", "beh")

# Read CSV file
 df <- read_csv(file.path(inputs_dir, "beh_data.csv"))
# # If you prefer Excel file you can read it by uncomment the line below
#df <- read_excel(file.path(inputs_dir,"beh_data.xlsx"))

#====================== MAIN SCRIPT =======================

# Ensure Reaction_Time is numeric
df$Reaction_Time <- as.numeric(df$Reaction_Time)
# Display a summary of the dataset
summary(df)
# Filter only relevant (go) trials (otherwise there are missing values (no-go trials))
df_relevant <- df %>%
  filter(Relevance == "relevant")
# Remove rows with missing Reaction_Time values
df_clean <- df_relevant %>%
  filter(!is.na(Reaction_Time))

#=============== Discrimination performance =================
# Compute discrimination performance by participant and agency condition
performance_by_participant <- df_clean %>%
  group_by(Participant, Agency_Condition) %>%
  summarise(
    perf = sum(Discrimination, na.rm = TRUE) / n()
  )
# Separate performance by condition
perf_active <- performance_by_participant %>%
  filter(Agency_Condition == "A")
perf_passive <- performance_by_participant %>%
  filter(Agency_Condition == "P")

# Normality tests (Shapiro-Wilk)
shapiro_perf_A <- shapiro.test(perf_active$perf)
shapiro_perf_P <- shapiro.test(perf_passive$perf)
cat("Shapiro-Wilk test for performance (perf):\n")
print(shapiro_perf_A)
print(shapiro_perf_P)

# Levene's test for homogeneity of variances
levene_result <- leveneTest(perf ~ factor(Agency_Condition), data = performance_by_participant)
cat("\nLevene's test for equality of variances (perf):\n")
print(levene_result)

# Paired t-test for performance
t_test_perf <- t.test(perf_active$perf, perf_passive$perf, var.equal = TRUE, paired = TRUE)
cat("\nPaired t-test for performance (perf):\n")
print(t_test_perf)

# Descriptive stats
mean(perf_active$perf)
sd(perf_active$perf)

mean(perf_passive$perf)
sd(perf_passive$perf)

#=============== Reaction Time =================

# Compute mean RT per participant and condition
mean_by_participant_rt <- df_clean %>%
  group_by(Participant, Agency_Condition) %>%
  summarise(mean_RT = mean(Reaction_Time), .groups = "drop")

# Separate RTs by condition
mean_active <- mean_by_participant_rt %>%
  filter(Agency_Condition == "A")
mean_passive <- mean_by_participant_rt %>%
  filter(Agency_Condition == "P")

# Normality tests (Shapiro-Wilk)
shapiro_RT_A <- shapiro.test(mean_active$mean_RT)
shapiro_RT_P <- shapiro.test(mean_passive$mean_RT)
cat("\nShapiro-Wilk test for Reaction Time:\n")
print(shapiro_RT_A)
print(shapiro_RT_P)

# Levene's test for RT
levene_result <- leveneTest(mean_RT ~ Agency_Condition, data = mean_by_participant_rt)
cat("\nLevene's test for equality of variances (Reaction Time):\n")
print(levene_result)

# Paired t-test for Reaction Time
t_test_rt <- t.test(mean_active$mean_RT, mean_passive$mean_RT, var.equal = TRUE, paired = TRUE)
cat("\nPaired t-test for Reaction Time:\n")
print(t_test_rt)

# Cohen's d for RT (paired)
diff_rt <- mean_active$mean_RT - mean_passive$mean_RT
cohen_d_rt <- mean(diff_rt) / sd(diff_rt)
cat("\nCohen's d (paired) for Reaction Time:\n")
print(cohen_d_rt)

# Descriptive stats (converted to ms)
mean(mean_active$mean_RT * 1000)
sd(mean_active$mean_RT * 1000)

mean(mean_passive$mean_RT * 1000)
sd(mean_passive$mean_RT * 1000)

