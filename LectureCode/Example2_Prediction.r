set.seed(123)
n <- 100
theta <- rnorm(n, mean = 70, sd = 10)
# Simulate true weights
y_observed <- rnorm(n, mean = mean(theta), sd = 5)
# Predicting a new value based on observed data
y_new <- rnorm(1, mean = mean(y_observed), sd = 5)
# Show predicted value
print(y_new)