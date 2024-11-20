# Probability of a girl birth given placenta previa

# As a specific example of a factor that may influence the sex ratio,
# we consider the maternal condition placenta previa, an unusual condition
# of pregnancy in which the placenta is implanted low in the
# uterus, obstructing the fetus from a normal vaginal delivery. An
# early study concerning the sex of placenta previa births in Germany
# found that of a total of 980 births, 437 were female. How much evidence
# does this provide for the claim that the proportion of female
# births in the population of placenta previa births is less than 0.485,
# the proportion of female births in the general population?

# Assumptions from the text:

# Observation: 437 female births, 543 male births.
# Prior: Uniform, Posterior: Beta(438, 544).

# Placenta previa posterior
girls <- 437
boys <- 543
theta_seq <- seq(0, 1, by = 0.01)
posterior <- dbeta(theta_seq, girls + 1, boys + 1)
plot(theta_seq, posterior,
  type = "l",
  main = "Posterior for Placenta Previa",
  xlab = "theta", ylab = "Density"
)

predictive_prob <- sum(theta_seq * posterior) / sum(posterior)
print(paste("Predictive probability:", predictive_prob))

# MLE Prediction version:by
# Observed data
total <- girls + boys
# Bayesian Estimation (Posterior with Beta prior)
posterior_mean <- (girls + 1) / (total + 2)
# Posterior mean for Beta(girls+1, boys+1)
# Plot Bayesian posterior
plot(theta_seq, posterior,
  type = "l",
  main = "Posterior Distribution for Theta",
  xlab = "theta", ylab = "Density"
)

# Traditional Estimation (MLE)
mle_estimate <- girls / total
# MLE is the sample proportion
abline(v = mle_estimate, col = "blue", lty = 2, lwd = 2)
# Add MLE line
abline(v = posterior_mean, col = "red", lty = 2, lwd = 2)
# Add Bayesian posterior mean line
legend("topright",
  legend = c("MLE Estimate", "Bayesian Posterior Mean"),
  col = c("blue", "red"), lty = 2, lwd = 2
)