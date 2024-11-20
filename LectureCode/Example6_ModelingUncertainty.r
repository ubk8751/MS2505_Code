mu <- 5 # Example posterior mean
sigma <- 2 # Example posterior std
# Draw samples from posterior
samples <- rnorm(1000, mean = mu, sd = sigma)
hist(samples,
  main = "Histogram of Posterior Samples",
  xlab = "Parameter Value"
)
mu <- mean(samples)
sigma <- sd(samples)
samples_new <- rnorm(1000, mean = mu, sd = sigma)
# Step 4: Plot the histogram of the samples
hist(samples_new,
  main = "Histogram of Posterior Samples",
  xlab = "Parameter Value"
)
