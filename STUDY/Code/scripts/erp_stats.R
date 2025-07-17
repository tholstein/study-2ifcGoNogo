#====================== LIBRAIRIES =========================
rm(list=ls())
library(dplyr)
library(tidyr)
library(afex)       # aov_ez
library(car)        # leveneTest
library(rstatix)    # wilcox tests + p.adjust
library(emmeans)
library(here)
library(readr)
library(readxl)
library(stringr)
library(ggpubr)
library(bestNormalize)
library(ggplot2)


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
inputs_dir <- file.path(osf_root, "Code", "inputs")
outputs_dir <- file.path(osf_root, "Code", "outputs")


# Build the path to the 'Plots' folder
plots_dir <- file.path(osf_root, "Plots", "erp")

# Read CSV file
recap <- read_csv(file.path(inputs_dir, "ERPs_stats.csv"))
# # If you prefer Excel file you can read it by uncomment the line below
# recap <- read_excel(file.path(inputs_dir, "ERPs_stats.csv"))

#====================== FUNCTIONS =========================
run_paired_ttests <- function(data, subject_col, condition_col, value_col, comp, elec, method = "bonferroni") {
  conditions <- unique(data[[condition_col]])
  combi <- combn(conditions, 2, simplify = FALSE)
  
  pvals <- c()
  t_results <- list()
  comparisons <- c()
  
  for (pair in combi) {
    cond1 <- pair[1]
    cond2 <- pair[2]
    
    d1 <- data %>% filter(!!sym(condition_col) == cond1) %>% arrange(!!sym(subject_col))
    d2 <- data %>% filter(!!sym(condition_col) == cond2) %>% arrange(!!sym(subject_col))
    
    common_ids <- intersect(d1[[subject_col]], d2[[subject_col]])
    d1 <- d1 %>% filter((!!sym(subject_col)) %in% common_ids)
    d2 <- d2 %>% filter((!!sym(subject_col)) %in% common_ids)
    
    x1 <- d1[[value_col]]
    x2 <- d2[[value_col]]
    
    ttest <- t.test(x1, x2, paired = TRUE)
    diff <- x1 - x2
    cohen_d <- mean(diff) / sd(diff)
    
    pvals <- c(pvals, ttest$p.value)
    comparisons <- c(comparisons, paste(cond1, "vs", cond2))
    
    t_results[[paste(cond1, "vs", cond2)]] <- list(
      t = ttest$statistic,
      df = ttest$parameter,
      p_uncorrected = ttest$p.value,
      cohen_d = cohen_d
    )
  }
  
  # Correction
  pvals_corrected <- p.adjust(pvals, method = method)
  
  # Console display
  cat("🔍 Résultats des t-tests pour", comp, "-", elec, "\n")
  for (i in seq_along(comparisons)) {
    res <- t_results[[comparisons[i]]]
    cat(sprintf(
      "  ▶ %s: t(%d) = %.2f, p = %.4f (corrigé = %.4f), d = %.3f\n",
      comparisons[i], res$df, res$t, res$p_uncorrected, pvals_corrected[i], res$cohen_d
    ))
  }
  
  # Return only corrected p-values for CSV
  df_csv <- data.frame(
    Component = comp,
    Electrode = elec,
    Comparison = comparisons,
    Corrected_p = pvals_corrected
  )
  
  return(df_csv)
}


#====================== MAIN SCRIPT =======================
# Data preprataion 
recap$Condition <- as.factor(recap$Condition)
recap$Amplitude <- as.numeric(recap$Amplitude)
recap$Participant <- as.factor(recap$Participant)

# Separating agency and task-relevance conditions 
recap[, c("Agency", "Relevance")] <- stringr::str_split_fixed(recap$Condition, " ", 2)

# Initilization
anova_results <- list()
wilcoxon_results <- list()
posthoc_results_list <- list()

# ERP components and electrodes 
combinations <- recap %>%
  distinct(Component, Electrode)

#====================== Principal for loop ===========================
for (i in 1:nrow(combinations)) {
  comp <- combinations$Component[i]
  elec <- combinations$Electrode[i]
  
  cat("\n===== Component:", comp, "| Electrode:", elec, "=====\n")
  
  data_subset <- recap %>%
    filter(Component == comp, Electrode == elec)
  
  # # Optional: ================= QQ PLOTS ==============
  # custom_colors <- c(
  #   "Active Relevant" = "red",
  #   "Passive Relevant" = "darkorange",
  #   "Active Irrelevant" = "darkblue",
  #   "Passive Irrelevant" = "lightblue"
  # )
  # 
  # # QQ plot with colors for dots and shadows
  # p <- ggqqplot(
  #   data_subset,
  #   x = "Amplitude",
  #   facet.by = "Condition",
  #   color = "Condition",  # points
  #   fill = "Condition",   # ombre (ribbon)
  #   ggtheme = theme_bw()
  # ) +
  #   scale_color_manual(values = custom_colors) +
  #   scale_fill_manual(values = custom_colors) +
  #   labs(
  #     title = paste("QQ Plot -", comp, "-", elec),
  #     subtitle = "Par condition"
  #   )
  # 
  # print(p)
  
  #=================== Normality test by condition ==================
  normality_pass <- TRUE
  for (cond in unique(data_subset$Condition)) {
    amp <- data_subset %>%
      filter(Condition == cond) %>%
      pull(Amplitude)
    
    if (length(amp) >= 3) {
      pval <- shapiro.test(amp)$p.value
      cat("🔹 Normalité -", cond, ": p =", round(pval, 4), "\n")
      if (pval <= 0.05) {
        normality_pass <- FALSE
      }
    } else {
      normality_pass <- FALSE
    }
  }
  
  #=================== Homogeneity of variances ==================
  homogeneity_pass <- FALSE
  if (normality_pass) {
    lev_test <- car::leveneTest(Amplitude ~ Condition, data = data_subset)
    lev_p <- lev_test$`Pr(>F)`[1]
    cat("🔹 Levene test p =", round(lev_p, 4), "\n")
    homogeneity_pass <- lev_p > 0.05
  }
  
  # === Run ANOVA if normality + homogeneity are OK ===
  if (normality_pass && homogeneity_pass) {
    cat("✅  Normal data and homogeneous variances → ANOVA with aov_ez()\n")
    
    anova_data <- data_subset %>%
      select(Participant, Amplitude, Condition, Agency, Relevance)
    
    result <- aov_ez(
      id = "Participant",
      dv = "Amplitude",
      within = c("Agency", "Relevance"),
      data = anova_data
    )
    
    print(result)
    anova_results[[paste(comp, elec, sep = "_")]] <- result
    
    #=================== Residual normality test (Shapiro-Wilk) ================
    # Combine all residuals into a single numeric vector
    residuals_all <- as.numeric(as.matrix(residuals(result$lm)))
    resid_df <- data.frame(Residual = residuals_all)
    shapiro_resid <- shapiro.test(residuals_all)
    cat("📊 Shapiro-Wilk test on all residuals: p =", round(shapiro_resid$p.value, 4), "\n")
    
    #=================== QQ Plot of residuals ================
    qqplot_resid <- ggqqplot(
      resid_df,
      x = "Residual",
      color = I("steelblue"),
      fill = I("lightblue"),
      ggtheme = theme_bw()
    ) +
      labs(
        title = paste("QQ Plot of Residuals -", comp, "-", elec)
      )
    
    print(qqplot_resid)
    
    plot_filename <- paste0("qqplot_", comp, "_", elec, ".png")
    ggsave(
      filename = file.path(plots_dir, plot_filename),
      plot = qqplot_resid,
      width = 6,
      height = 5,
      dpi = 300
    )    
    dev.off()
    
    
    #=================== Post-hoc paired t-tests with Bonferroni correction ================
    df_posthoc <- run_paired_ttests(data_subset, "Participant", "Condition", "Amplitude", comp, elec)
    posthoc_results_list[[paste(comp, elec, sep = "_")]] <- df_posthoc
    
    
  } else {
    
    cat("↪ Non-normal data → Yeo-Johnson transformation\n")
    
    
    # Yeo-Johnson transformation
    transformed_subset <- data_subset # we keep the original data to check
    transformed_subset$Amplitude <- NA 
    trans <- bestNormalize::yeojohnson(data_subset$Amplitude)
    transformed_subset$Amplitude <- predict(trans)
    
    
    # Recheck normality after transformation
    transformed_normality <- TRUE
    for (cond in unique(transformed_subset$Condition)) {
      amps <- transformed_subset %>% filter(Condition == cond) %>% pull(Amplitude)
      if (length(amps) >= 3) {
        pval <- shapiro.test(amps)$p.value
        cat("🔁 After Yeo-Johnson -", cond, ": p =", round(pval, 4), "\n")
        if (pval <= 0.05) {
          transformed_normality <- FALSE
        }
      } else {
        transformed_normality <- FALSE
      }
    }
    
    # Levene test on transformed data
    homogeneity_pass <- FALSE
    if (transformed_normality) {
      lev_test <- car::leveneTest(Amplitude ~ Condition, data = transformed_subset)
      lev_p <- lev_test$`Pr(>F)`[1]
      cat("🔹 Levene test after transformation : p =", round(lev_p, 4), "\n")
      homogeneity_pass <- lev_p > 0.05
    }
    
    if (transformed_normality && homogeneity_pass) {
      cat("✅ Transformed data are normal & homogeneous variances → ANOVA on transformed data\n")
      
      anova_data <- transformed_subset %>%
        select(Participant, Amplitude, Condition, Agency, Relevance)
      
      result <- aov_ez(
        id = "Participant",
        dv = "Amplitude",
        within = c("Agency", "Relevance"),
        data = anova_data
      )
      
      print(result)
      anova_results[[paste(comp, elec, "transformed", sep = "_")]] <- result
      
      #=================== Residual normality test (Shapiro-Wilk) ================
      # Combine all residuals into a single numeric vector
      residuals_all <- as.numeric(as.matrix(residuals(result$lm)))
      resid_df <- data.frame(Residual = residuals_all)
      shapiro_resid <- shapiro.test(residuals_all)
      cat("📊 Shapiro-Wilk test on all residuals: p =", round(shapiro_resid$p.value, 4), "\n")
      
      #=================== QQ Plot of residuals ================
      qqplot_resid <- ggqqplot(
        resid_df,
        x = "Residual",
        color = I("steelblue"),
        fill = I("lightblue"),
        ggtheme = theme_bw()
      ) +
        labs(
          title = paste("QQ Plot of Residuals -", comp, "-", elec)
        )
      
      print(qqplot_resid)
      
      plot_filename <- paste0("qqplot_", comp, "_", elec, ".png")
      ggsave(
        filename = file.path(plots_dir, plot_filename),
        plot = qqplot_resid,
        width = 6,
        height = 5,
        dpi = 300
      )    
      dev.off()
      
      # #=================== Post-hoc paired t-tests with Bonferroni correction ================
      df_posthoc <- run_paired_ttests(data_subset, "Participant", "Condition", "Amplitude", comp, elec)
      posthoc_results_list[[paste(comp, elec, "transformed", sep = "_")]] <- df_posthoc
      
      
    } else {
      
      cat("❌ Even after transformation → Wilcoxon\n")
      
      wide_data <- data_subset %>%
        select(Participant, Condition, Amplitude) %>%
        pivot_wider(names_from = Condition, values_from = Amplitude)
      
      conditions <- colnames(wide_data)[-1]
      combinaisons <- combn(conditions, 2, simplify = FALSE)
      
      pvals <- c()
      noms <- c()
      
      for (pair in combinaisons) {
        d1 <- wide_data[[pair[1]]]
        d2 <- wide_data[[pair[2]]]
        
        if (all(!is.na(d1)) && all(!is.na(d2))) {
          test <- wilcox.test(d1, d2, paired = TRUE, exact = FALSE)
          raw_p <- test$p.value
          nom_test <- paste(pair[1], "vs", pair[2])
          
          cat("🟠", nom_test, "| p =", round(raw_p, 4), "\n")
          
          wilcoxon_results[[paste(comp, elec, nom_test, sep = "_")]] <- test
          pvals <- c(pvals, raw_p)
          noms <- c(noms, nom_test)
        }
      }
      
      # Bonferroni Correction
      corrected <- p.adjust(pvals, method = "bonferroni")
      cat("\n🔹 Bonferroni correction applied :\n")
      for (j in seq_along(noms)) {
        cat(noms[j], "→ p corrected =", round(corrected[j], 4), "\n")
      }
      
      wilcox_df <- data.frame(
        Component = comp,
        Electrode = elec,
        Comparison = noms,
        Raw_p = pvals,
        Corrected_p = corrected
      )
      posthoc_results_list[[paste(comp, elec, "wilcoxon", sep = "_")]] <- wilcox_df
      
    }
  }
}

# Combiner toutes les data.frames
final_posthoc_results <- bind_rows(posthoc_results_list)

if (!dir.exists(outputs_dir)) {
  dir.create(outputs_dir, recursive = TRUE)
}

write.csv(
  final_posthoc_results,
  file = file.path(outputs_dir, "posthoc_corrected_pvalues.csv"),
  row.names = FALSE
)

