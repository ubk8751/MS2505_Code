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

# Extract the spam count and total count
k <- sum(mail_data$Category == 1) # Number of spam emails
n <- nrow(mail_data) # Total number of emails

# Beta(1,1) prior parameters
alpha_prior <- 1
beta_prior <- 1

# Posterior parameters
alpha_post <- alpha_prior + k
beta_post <- beta_prior + (n - k)

# Define the posterior distribution (up to a constant factor)
posterior <- function(theta) {
  if (theta < 0 || theta > 1) {
    return(0) # Return 0 if theta is outside the valid range
  }
  return(dbeta(theta, alpha_post, beta_post))
}

# MCMC using Metropolis-Hastings
mcmc <- function(start, iter, proposal_sd) {
  theta <- start
  samples <- numeric(iter)

  for (i in 1:iter) {
    # Propose a new value from a normal distribution
    theta_new <- rnorm(1, mean = theta, sd = proposal_sd)

    # Calculate acceptance probability only if theta_new is valid
    if (posterior(theta_new) > 0) {
      acceptance_ratio <- posterior(theta_new) / posterior(theta)
    } else {
      acceptance_ratio <- 0
    }

    # Accept or reject the new value
    if (runif(1) < acceptance_ratio) {
      theta <- theta_new
    }

    # Store the sample
    samples[i] <- theta
  }

  return(samples)
}

# Run the MCMC
set.seed(123) # For reproducibility
samples <- mcmc(start = 0.5, iter = 10000, proposal_sd = 0.1)

# Remove samples outside the [0, 1] range (not strictly necessary anymore)
samples <- samples[samples >= 0 & samples <= 1]

# Summarize results
cat("Mean of posterior samples (estimate of theta):", mean(samples), "\n")
cat("95% credible interval:", quantile(samples, c(0.025, 0.975)), "\n")

# Plot the sampled posterior
hist(samples,
  probability = TRUE, breaks = 50,
  main = "Posterior Distribution of Theta (P(spam))",
  xlab = "Theta (P(spam))"
)
curve(dbeta(x, alpha_post, beta_post),
  col = "red",
  add = TRUE,
  lwd = 2
)
legend("topright",
  legend = c("True Posterior"),
  col = c("red"),
  lty = 1,
  lwd = 2
)

# Flush the output and close the sink
flush.console()
sink()