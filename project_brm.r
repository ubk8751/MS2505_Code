# =====================================================
# Bayesian Analysis for Email Spam Classification
# =====================================================

# Load Required Libraries and Check Installations
required_packages <- c("bayesplot", "rstanarm", "ggplot2", "brms", "dplyr")
installed_packages <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed_packages) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# Set a global random seed for reproducibility
set.seed(123)

# =====================================================
# Setup Directories
# =====================================================

# Create Necessary Directories
dirs <- c("logs", "figures")
sapply(
  dirs,
  function(dir) if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
)

# Define log file paths
r_output_log_file <- "logs/R_output.log"
results_log_file <- "logs/results.log"

# Redirect general R output to R_output.log
sink(r_output_log_file)

# =====================================================
# Load Data
# =====================================================

data_path <- "data/mail_data_bin.csv"
if (!file.exists(data_path)) stop("Data file not found!")
mail_data <- read.csv(data_path)

# Ensure the response variable is binary and properly coded
if (!all(mail_data$Category %in% c(0, 1))) {
  stop("The response variable 'Category' must be binary (0 or 1).")
}

# Metadata
spam_count <- sum(mail_data$Category == 1)
ham_count <- sum(mail_data$Category == 0)
total_emails <- nrow(mail_data)

# Print basic information to R_output.log
cat("Data loaded successfully.\n")

# =====================================================
# Fit Bayesian Model with Informative Prior
# =====================================================

# Calculate equivalent normal prior on log-odds scale
prior_alpha <- 1
prior_beta <- 1
posterior_alpha <- prior_alpha + spam_count # alpha + y
posterior_beta <- prior_beta + total_emails - spam_count # beta + n - y

# Train the model
fit <- brm(
  Category ~ 1,
  data = mail_data,
  family = bernoulli(),
  prior = prior_string(
    paste0("beta(", posterior_alpha, ", ", posterior_beta, ")"),
    class = "Intercept"
  ) # Correctly specify normal prior using prior_string
)

# Close R_output.log sink after training completes
sink()
pp_check(fit, type = "dens_overlay")

# Redirect output to results log
sink(results_log_file)

# =====================================================
# Print Posterior Diagnostics and Convergence Checks
# =====================================================

# Print model summary and convergence diagnostics
cat("\n--- Convergence Diagnostics ---\n")
summary_results <- summary(fit)

# R-hat values and Effective Sample Sizes
cat("R-hat values:\n")
cat(paste(capture.output(print(summary_results$fixed, digits = 2)), collapse = "\n"), "\n")
cat("Effective sample sizes (n_eff):\n")
cat(paste(capture.output(print(summary_results$random, digits = 2)), collapse = "\n"), "\n")

# =====================================================
# Monte Carlo Standard Error
# =====================================================

cat("\n--- Monte Carlo Standard Error (MCSE) ---\n")
posterior_draws <- posterior_samples(fit, pars = "b_Intercept")
mcse_results <- summarise(posterior_draws, MCSE = sd(b_Intercept) / sqrt(n()))
cat("MCSE for Intercept:", mcse_results$MCSE, "\n")

# =====================================================
# Generate Trace Plots
# =====================================================

trace_plot_path <- "figures/trace_plots.pdf"
pdf(trace_plot_path)
mcmc_trace(as.array(fit), pars = c("b_Intercept")) + 
  ggtitle("Trace Plots: Intercept")
dev.off()
cat("Trace plots saved to:", trace_plot_path, "\n")

# =====================================================
# Generate Density Plots
# =====================================================

# Save density plots with posterior mean and credible interval
density_plot_path <- "figures/density_plots.pdf"

# Compute summary statistics
posterior_summary <- posterior_samples(fit, pars = "b_Intercept") %>%
  summarise(
    mean = mean(b_Intercept),
    lower = quantile(b_Intercept, 0.025),
    upper = quantile(b_Intercept, 0.975)
  )

pdf(density_plot_path)

# Extract posterior draws for plotting
posterior_draws <- posterior_samples(fit, pars = "b_Intercept")

# Create density plot with posterior mean and 95% CI
ggplot(data = posterior_draws, aes(x = b_Intercept)) +
  geom_density(fill = "lightblue", alpha = 0.7) +  # Density curve
  geom_vline(xintercept = posterior_summary$mean, 
             color = "red", 
             linetype = "dashed", 
             size = 1) +  # Posterior mean
  annotate("text", x = posterior_summary$mean, y = 0, 
           label = paste0("Mean: ", round(posterior_summary$mean, 2)),
           vjust = -1, color = "red") +  # Annotate the mean
  geom_vline(xintercept = posterior_summary$lower, 
             color = "blue", 
             linetype = "dotted") +  # Lower CI
  geom_vline(xintercept = posterior_summary$upper, 
             color = "blue", 
             linetype = "dotted") +  # Upper CI
  annotate("text", x = posterior_summary$lower, y = 0, 
           label = paste0("Lower: ", round(posterior_summary$lower, 2)),
           vjust = -1, color = "blue") +  # Annotate the lower CI
  annotate("text", x = posterior_summary$upper, y = 0, 
           label = paste0("Upper: ", round(posterior_summary$upper, 2)),
           vjust = -1, color = "blue") +  # Annotate the upper CI
  labs(
    title = "Density Plot: Posterior for Intercept",
    x = "Intercept (log-odds)",
    y = "Density"
  ) +
  theme_minimal()

dev.off()
cat("Density plot saved to:", density_plot_path, "\n")

# =====================================================
# Autocorrelation Analysis
# =====================================================

autocorr_plot_path <- "figures/autocorrelation_plots.pdf"
pdf(autocorr_plot_path)
mcmc_acf(as.array(fit), pars = c("b_Intercept")) +
  ggtitle("Autocorrelation Plot: Intercept")
dev.off()
cat("Autocorrelation plots saved to:", autocorr_plot_path, "\n")

# =====================================================
# Posterior Predictive Checks
# =====================================================

cat("\n--- Posterior Predictive Check ---\n")

# Generate posterior predictive samples (binary outcomes)
posterior_samples <- posterior_predict(fit)

# Calculate the proportion of spam in each predictive sample
proportion_spam_binary <- apply(posterior_samples, 1, mean)
cat(
  "Mean Proportion of Spam (Binary Outcomes):",
  mean(proportion_spam_binary), "\n"
)

# Generate expected probabilities for each observation
posterior_probs <- posterior_epred(fit)

# Calculate the average probability across observations
expected_proportion <- rowMeans(posterior_probs)
cat(
  "Mean Proportion of Spam (Expected Probabilities):",
  mean(expected_proportion), "\n"
)

# =====================================================
# Save Figures
# =====================================================

# Save binary outcome proportions plot
binary_outcomes_plot <- "figures/posterior_proportion_spam_binary_outcomes.pdf"
pdf(binary_outcomes_plot)
hist(
  proportion_spam_binary,
  main = "Posterior Predictive Check: Proportion of Spam Emails",
  xlab = "Proportion of Spam Emails",
  col = "lightblue",
  border = "black"
)
dev.off()

# Save expected probabilities plot
expected_probs_plot <- "figures/posterior_expected_proportion_spam.pdf"
pdf(expected_probs_plot)
hist(
  expected_proportion,
  main = "Posterior Predictive Check: Expected Proportion of Spam Emails",
  xlab = "Proportion of Spam Emails",
  col = "lightgreen",
  border = "black"
)
dev.off()

# Save session information
session_info_path <- "logs/session_info.log"
writeLines(capture.output(sessionInfo()), session_info_path)

# Close results.log sink
sink()

cat("Analysis complete. Check 'logs' and 'figures' directories for results.\n")
