# Grid setup for posterior computations
alpha_grid <- seq(0, 10, length.out = 100)
beta_grid <- seq(0, 10, length.out = 100)

# Simulated likelihood values
# (replace with actual computations)
likelihood <- outer(
  alpha_grid, beta_grid,
  function(alpha, beta) {
    dnorm(alpha, mean = 5, sd = 1) *
    dnorm(beta, mean = 5, sd = 2)
  }
)

# Compute the joint posterior on the grid
prior_alpha <- dnorm(alpha_grid, mean = 5, sd = 2)
prior_beta <- dnorm(beta_grid, mean = 5, sd = 2)
prior <- outer(prior_alpha, prior_beta, "*")

posterior <- likelihood * prior
posterior <- posterior / sum(posterior)
# Normalize to make it a valid probability

# Compute marginal posterior for alpha
posterior_alpha <- apply(posterior, 1, sum)

# Draw samples from the marginal
# posterior of alpha
set.seed(123)
sample_alpha_indices <- sample(
  seq_along(alpha_grid),
  size = 1000,
  prob = posterior_alpha, replace = TRUE
)
sample_alpha <- alpha_grid[sample_alpha_indices]

# Draw conditional samples of beta given alpha
sample_beta <- numeric(1000)
for (s in seq_len(1000)) {
  alpha_index <- sample_alpha_indices[s]
  posterior_beta_given_alpha <- posterior[alpha_index, ]
  posterior_beta_given_alpha <- posterior_beta_given_alpha / sum(posterior_beta_given_alpha) # Normalize
  beta_index <- sample(seq_along(beta_grid), size = 1, prob = posterior_beta_given_alpha)
  sample_beta[s] <- beta_grid[beta_index]
}
# Add uniform random jitter
grid_spacing_alpha <- diff(alpha_grid[1:2])
grid_spacing_beta <- diff(beta_grid[1:2])
sample_alpha <- sample_alpha + runif( 1000, -grid_spacing_alpha / 2, grid_spacing_alpha / 2 )
sample_beta <- sample_beta + runif( 1000, -grid_spacing_beta / 2, grid_spacing_beta / 2 )

# Visualization
par(mfrow = c(1, 2))
hist(sample_alpha, main = "Posterior Samples of Alpha", xlab = "Alpha")
hist(sample_beta, main = "Posterior Samples of Beta", xlab = "Beta")
# Scatter plot to show the joint samples
plot(sample_alpha, sample_beta,
  main = "Joint Posterior Samples",
  xlab = "Alpha", ylab = "Beta", pch = 19,
  col = rgb(0.1, 0.2, 0.8, 0.3)
)

# Simulate Monte Carlo samples
set.seed(123) # For reproducibility
posterior_means <- replicate( 1000, mean(rnorm(100, mean = 0, sd = 1)) )
# Plot histogram of posterior means
hist(posterior_means,
  main = "Posterior Mean Estimation",
  xlab = "Estimated Mean"
)
