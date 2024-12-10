# Create the directory if it doesn't exist
if (!dir.exists("Project/logs")) {
    dir.create("Project/logs", recursive = TRUE)
}

# Specify the log file path
log_file <- "Project/logs/R_output.log"

# Open the sink to redirect output
sink(log_file)

# Read the data
mail_data <- read.csv("Project/data/mail_data_bin.csv")

# Set metadata
alpha_prior <- 1
beta_prior <- 1
total_emails <- nrow(mail_data)
spam_count <- sum(mail_data$Category == 1)
ham_count <- sum(mail_data$Category == 0)

# Compute posterior parameters for P(spam)
alpha_post <- alpha_prior + spam_count
beta_post <- beta_prior + ham_count

# Posterior probability of an email being spam
posterior_spam <- alpha_post / (alpha_post + beta_post)

# Print results
cat("Prior: Beta(", alpha_prior, ",", beta_prior, ")\n")
cat("Spam Count:", spam_count, "\n")
cat("Ham Count:", ham_count, "\n")
cat("Posterior: Beta(", alpha_post, ",", beta_post, ")\n")
cat("P(spam):", posterior_spam, "\n")

# Flush the output and close the sink
flush.console()
sink()
