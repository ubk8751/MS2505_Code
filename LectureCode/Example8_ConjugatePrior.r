library(ggplot2)

# Simulated data
set.seed(123)

n <- 100 # Sample linewidth
true_mu <- 5
true_sigma <- 2
data <- rnorm(n, mean = true_mu, sd = true_sigma)

# Prior parameters
mu_0 <- 0
# Prior mean for mu
tau_0 <- 5
# Prior standard deviation for mu
alpha_0 <- 2
# Shape parameter for Inv-Gamma (prior for sigma^2)
beta_0 <- 1
# Scale parameter for Inv-Gamma (prior for sigma^2)

# Sufficient statistics
y_bar <- mean(data)
s_sq <- var(data)

# Posterior parameters
tau_n <- 1 / sqrt(1 / tau_0^2 + n / s_sq)
# Posterior std for mu
mu_n <- (mu_0 / tau_0^2 + n * y_bar / s_sq) * tau_n^2
# Posterior mean for mu
alpha_n <- alpha_0 + n / 2
# Posterior shape for sigma^2
beta_n <- beta_0 + 0.5 * sum((data - y_bar)^2)
# Posterior scale for sigma^2

# Monte Carlo sampling
N_mc <- 5000
sigma_sq_samples <- 1 / rgamma(N_mc, alpha_n, rate = beta_n)
sigma_samples <- sqrt(sigma_sq_samples)
mu_samples <- rnorm(N_mc, mean = mu_n, sd = sqrt(sigma_sq_samples / n))
# Plot histograms and posterior averages
posterior_mu_mean <- mean(mu_samples)
posterior_sigma_mean <- mean(sigma_samples)

# Histogram for mu
ggplot(data.frame(mu_samples), aes(x = mu_samples)) +
  geom_histogram(bins = 30, color = "black", fill = "blue", alpha = 0.7) +
  geom_vline(xintercept = posterior_mu_mean, color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = expression(paste("Posterior Distribution of myu")), x = "myu", y = "Frequency") +
  theme_minimal()

# Histogram for sigma
ggplot(data.frame(sigma_samples), aes(x = sigma_samples)) +
  geom_histogram(bins = 30, color = "black",  fill = "green", alpha = 0.7) +
  geom_vline(xintercept = posterior_sigma_mean, color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = expression(paste("Posterior Distribution of sigma")), x = "sigma", y = "Frequency") +
  theme_minimal()

print(paste("Posterior myu mean:", posterior_mu_mean))
print(paste("Posterior sigma mean:", posterior_sigma_mean))