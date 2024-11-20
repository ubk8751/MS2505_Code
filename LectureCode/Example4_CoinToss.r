# Define parameters
n <- 10 # Number of trials
y <- 6 # Number of observed successes
# Define a sequence of theta values
theta_seq <- seq(0, 1, length = 100)
# Calculate the posterior distribution
#with Beta(y+1, n-y+1)
posterior <- dbeta(theta_seq, y + 1, n - y + 1)

# Plot the posterior distribution
plot(theta_seq, posterior, type = "l",
     main = "Posterior Distribution for Beta(7,5)",
     xlab = "theta", ylab = "Density")
     
# Calculate the mean of the posterior distribution
posterior_mean <- (y + 1) / (y + 1 + (n - y + 1))
print(posterior_mean)