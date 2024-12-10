# Create the directory if it doesn't exist
if (!dir.exists("Project/logs")) {
    dir.create("Project/logs", recursive = TRUE)
}

# Specify the log file path
log_file <- "Project/logs/R_output.log"

# Open the sink to redirect output
sink(log_file)

# Load the data
mail_data <- read.csv("Project/data/mail_data_bin.csv")

# Set prior parameters for Beta(1,1)
alpha_prior <- 1
beta_prior <- 1

# Total emails
total_emails <- nrow(mail_data)

# Count spam emails
spam_count <- sum(mail_data$Category == 1)

# Count ham emails
ham_count <- sum(mail_data$Category == 0)

# Compute posterior parameters
alpha_post <- alpha_prior + spam_count
beta_post <- beta_prior + ham_count

# Sample from the posterior distribution
set.seed(42) # For reproducibility
num_samples <- 10000 # Number of samples
samples <- rbeta(num_samples, alpha_post, beta_post)

# Estimate theta as the mean of the samples
theta_estimate <- mean(samples)

# Print results
cat("Posterior Samples (First 10):", head(samples, 10), "\n")
cat("Estimated Theta (Mean):", theta_estimate, "\n")

# Visualize the posterior distribution
hist(samples, breaks = 50, main = "Posterior Distribution of Theta (P(Y = spam))",
     xlab = "Theta (P(Y = spam))", col = "lightblue", border = "black")
abline(v = theta_estimate, col = "red", lwd = 2, lty = 2)

# Flush the output and close the sink
flush.console()
sink()