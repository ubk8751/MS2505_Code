# Define prior distribution
prior <- rbeta(1000, 1, 1)
# Update with observed data
posterior <- dbeta(seq(0, 1, length = 100), 6, 6)
# Plot results
plot(seq(0, 1, length = 100),
     posterior, type = "l")
    