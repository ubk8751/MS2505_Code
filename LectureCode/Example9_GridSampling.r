# Single parameter sampling of theta

# Define grid for theta
theta_grid <- seq(0, 1, length.out = 100) # Defines a sequence of values for θ.
# Prior (e.g., uniform)
prior <- rep(1, length(theta_grid))

# Likelihood: example for binomial likelihood
y <- 5 # observed successes
n <- 10 # total trials
likelihood <- dbinom(y, size=n, prob=theta_grid) # Computes likelihood values on the grid.

# Posterior Calculated and normalized across the grid.
# Unnormalized posterior
unnormalized_posterior <- likelihood * prior

# Normalize to get posterior
posterior <- unnormalized_posterior / sum(unnormalized_posterior)

# Plot posterior distribution
plot(theta_grid, posterior, type="h",
     main="Posterior of theta",
     xlab="theta", ylab="Posterior Probability")