# Load required libraries
library(bayesplot)
library(rstanarm)
library(ggplot2)
library(brms)

# Set a global random seed for reproducibility
set.seed(123)

# Create the directory if it doesn't exist
if (!dir.exists("Project/logs")) {
    dir.create("Project/logs", recursive = TRUE)
}

# Create the directory for figures if it doesn't exist
if (!dir.exists("figures")) {
    dir.create("figures", recursive = TRUE)
}

# Specify the log file path
log_file <- "Project/logs/R_output.log"

# Open the sink to redirect output
sink(log_file)

# Read the data
mail_data <- read.csv("Project/data/mail_data_bin.csv")

# Set metadata
alpha_prior <- 1
beta_prior <- 1
total_emails <- nrow(mail_data)
spam_count <- sum(mail_data$Category == 1)
ham_count <- sum(mail_data$Category == 0)

# Fit the model with a vague prior
fit_prior <- brm(
    Category ~ 1,
    data = mail_data,
    family = bernoulli(),
    prior = prior(beta(1, 1),
        class = "Intercept"
    )
)

# Prior predictive check
pdf("figures/prior_predictive_check.pdf")
pp_check(fit_prior, type = "hist") +
    ggtitle("Prior Predictive Check for Email Spam Model")
dev.off()

# Fit a robust model with a more informative prior
fit_robust <- brm(
    Category ~ 1,
    data = mail_data,
    family = bernoulli(),
    prior = prior(beta(2, 2), class = "Intercept")
)

# Posterior predictive check
pdf("figures/posterior_predictive_check.pdf")
pp_check(fit_robust, type = "dens_overlay") +
    ggtitle("Posterior Predictive Check for Robust Email Spam Model")
dev.off()

# Compute posterior parameters for P(spam)
alpha_post <- alpha_prior + spam_count
beta_post <- beta_prior + ham_count

# Posterior probability of an email being spam
posterior_spam <- alpha_post / (alpha_post + beta_post)

# Print results
cat("Prior: Beta(", alpha_prior, ",", beta_prior, ")\n")
cat("Spam Count:", spam_count, "\n")
cat("Ham Count:", ham_count, "\n")
cat("Posterior: Beta(", alpha_post, ",", beta_post, ")\n")
cat("P(spam):", posterior_spam, "\n")

# Posterior Predictive Checking
cat("\n--- Posterior Predictive Checking ---\n")

# Simulate posterior predictive samples
num_samples <- 1000
posterior_samples <- rbeta(num_samples, alpha_post, beta_post)

observed_counts <- spam_count / total_emails

# Generate density overlay
y <- rep(observed_counts, num_samples)
yrep <- matrix(posterior_samples, nrow = num_samples)

# Save density overlay plot
# pdf("../figures/ppc_density_overlay.pdf")
# ppc_dens_overlay(y = y, yrep = yrep) +
#    ggtitle("Posterior Predictive Check: Density Overlay")
# dev.off()

# Generate and save histogram of posterior samples
pdf("figures/ppc_histogram.pdf")
hist(posterior_samples,
    breaks = 30, col = "blue", border = "white",
    main = "Posterior Predictive Distribution", xlab = "P(spam)"
)
dev.off()

# Sensitivity Analysis
cat("\n--- Sensitivity Analysis ---\n")
sensitivity_results <- data.frame()
alpha_values <- seq(0.5, 2, by = 0.5)
beta_values <- seq(0.5, 2, by = 0.5)

for (alpha in alpha_values) {
    for (beta in beta_values) {
        alpha_post_temp <- alpha + spam_count
        beta_post_temp <- beta + ham_count
        posterior_temp <- alpha_post_temp / (alpha_post_temp + beta_post_temp)
        sensitivity_results <- rbind(
            sensitivity_results,
            data.frame(alpha, beta, posterior_temp)
        )
    }
}

# Display the sensitivity analysis results
cat("Sensitivity Analysis Results:\n")
print(sensitivity_results)

# Save sensitivity results to CSV
write.csv(sensitivity_results,
    "Project/logs/sensitivity_analysis.csv",
    row.names = FALSE
)

# Flush the output and close the sink
flush.console()
sink()
