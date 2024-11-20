# This code simulates binomial data for n = 10 trials with theta = 0.5.
n <- 10 # Number of trials
theta <- 0.5 # Probability of success
y <- rbinom(1000, size = n, prob = theta)
hist(y, main = "Simulated Binomial Data", xlab = "Number of Successes")

# This code calculates and plots the likelihood of observing y = 6 successes
# out of n = 10 trials.
theta_seq <- seq(0, 1, by = 0.01)
likelihood <- dbinom(6, size = 10, prob = theta_seq)
plot(theta_seq, likelihood, type = "l",
     main = "Likelihood for y=6, n=10",
     xlab = "theta", ylab = "Likelihood")

# This code calculates the posterior distribution for a binomial
# example with uniform prior.
bayesian_posterior <- function(theta, y, n) { 
dbinom(y, size = n, prob = theta) * dunif(theta, 0, 1) 
}
theta_seq <- seq(0, 1, by = 0.01)
posterior <- sapply(theta_seq, bayesian_posterior, y = 6, n = 10)
plot(theta_seq, posterior, type = "l", main = "Posterior Distribution",
     xlab = "theta", ylab = "Density")