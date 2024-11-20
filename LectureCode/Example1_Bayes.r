# Defining probabilities
p_cancer <- 0.05
p_positive_given_cancer <- 0.78
p_positive <- 0.096
# Applying Bayes’ Rule
p_cancer_given_positive <- (p_positive_given_cancer * p_cancer) / p_positive
# Result
print(p_cancer_given_positive)