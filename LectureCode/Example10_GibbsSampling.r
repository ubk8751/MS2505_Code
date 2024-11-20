# Simulated observed data
y <- rnorm(50, mean = 10, sd = 3)
# 50 observations, true mean=10, sd=3
# Initialize values
n <- length(y)
mu <- mean(y)
sigma_sq <- var(y)
iterations <- 1000

# Collect Gibbs samples, approximating the joint posterior.
mu_samples <- numeric(iterations)
sigma_samples <- numeric(iterations)

# Gibbs sampling loop
for (i in 1:iterations) {
  # Step 1: Sample mu given sigma_sq
  mu <- rnorm(1, mean = mean(y), sd = sqrt(sigma_sq / n))
  mu_samples[i] <- mu
  # Step 2: Sample sigma_sq given mu
  sigma_sq <- 1 / rgamma(1, shape = (n - 1) / 2, rate = sum((y - mu)^2) / 2)
  sigma_samples[i] <- sqrt(sigma_sq)
}

# Plot results to show the distributions for likely values of myu and sigma.
par(mfrow = c(1, 2))
hist(mu_samples,
  main = "Posterior of mu",
  xlab = "mu"
)
hist(sigma_samples,
  main = "Posterior of sigma",
  xlab = "sigma"
)
