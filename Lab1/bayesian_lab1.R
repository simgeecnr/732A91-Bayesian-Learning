# Bayesian Learning Lab 1
# nazbi056 & simci637

# QUESTION 1

alpha <- 8 
beta <- 8  
n <- 70 
s <- 22
f <- n-s
nDraws <- 10000 

# part a)-------------------------------------------------------------------------------------
alpha_posterior <- alpha + s
beta_posterior <- beta + f

true_mean <- alpha_posterior / (alpha_posterior + beta_posterior)
true_variance <- (alpha_posterior * beta_posterior) / ((alpha_posterior+beta_posterior+1)*
                                                         (alpha_posterior+beta_posterior)^2)

cat("True mean:", true_mean,"\n")
cat("True SD:", sqrt(true_variance),"\n")

set.seed(123)
# Draws from posterior distribution
random_values_posterior <- rbeta(nDraws, alpha_posterior, beta_posterior)

sample_means <- sapply(2:nDraws, function(i) mean(random_values_posterior[1:i]))
sample_stds <- sapply(2:nDraws, function(i) sd(random_values_posterior[1:i]))

mean_sd_data <- data.frame(Draws = 2:nDraws, Mean = sample_means, Std = sample_stds)

plot(mean_sd_data$Draws, mean_sd_data$Mean, type = 'l', xlab = "Draws", ylab = "Mean", 
     main = "Convergence to True Mean")
abline(h = true_mean, col = "red", lwd = 2, lty = 2)
grid()

plot(mean_sd_data$Draws, mean_sd_data$Std, type = 'l', xlab = "Draws", ylab = "SD", 
     main = "Convergence to True SD")
abline(h = sqrt(true_variance), col = "red", lwd = 2, lty = 2)
grid()


# part b) -------------------------------------------------------------------------------------
posterior_probability <- mean(random_values_posterior > 0.3)
exact_posterior_probability <- 1 - pbeta(0.3, alpha_posterior, beta_posterior)

cat("Posterior probability:", posterior_probability, "\n", 
    "Exact Value:", exact_posterior_probability)


# part c) -------------------------------------------------------------------------------------

# Calculate the odds phi = theta / (1 - theta)
phi_samples <- random_values_posterior / (1 - random_values_posterior)

# Plot the posterior distribution of phi
hist(phi_samples, breaks = 30, freq = FALSE, 
     main = "Posterior Distribution of Phi", xlab = "Phi")
lines(density(phi_samples), col = "red", lwd = 2)

#QUESTION 2

# part a) -------------------------------------------------------------------------------------
mu <- 3.6
obs <- c(33, 24, 48, 32, 55, 74, 23, 17)
n <- length(obs)
log_obs <- log(obs)
mean_data <- mean(log_obs) # mean of the observations
tau2 <- sum((log(obs) - mu)^2) / n # variance of the observations

nDraws <- 10000
theta <- matrix(0, nDraws, 2)
theta[,2] <- (n*tau2)/ rchisq(nDraws, n)
theta[,1] <- rnorm(nDraws, mean=mean_data, sd=sqrt(theta[,2]/n))

# Plotting the histogram of sigma^2-draws
hist(theta[,2], breaks = 50, main = 'Histogram of sigma^2 draws') 

# part b) -------------------------------------------------------------------------------------
G <- 2 * pnorm(sqrt(theta[,2])/ sqrt(2)) - 1
hist(G, breaks = 100, main = "Histogram of Gini Index")


# part c) -------------------------------------------------------------------------------------
# Compute the lower and upper bounds of the credible interval
lower_bound <- quantile(G, 0.025)
upper_bound <- quantile(G, 0.975)

cat("95% Equal Tail Credible Interval for G:", round(lower_bound, 4), "-", 
    round(upper_bound, 4), "\n")
cat("Difference between upper and lower bound:", upper_bound-lower_bound)

# part d) -------------------------------------------------------------------------------------
density_estimate <- density(G)
sorted_density <- sort(density_estimate$y, decreasing = TRUE) # Sort density values
cumulative_density <- cumsum(sorted_density)

cdf <- cumulative_density[(cumulative_density < sum(sorted_density) * 0.95)]
idx <- density_estimate$x[order(density_estimate$y,decreasing=TRUE)][1:length(cdf)]
lower_bound_hpdi <- min(idx)
upper_bound_hpdi <- max(idx)

cat("HPDI for G:", round(lower_bound_hpdi, 4), "-", round(upper_bound_hpdi, 4), "\n")
cat("Difference between upper and lower bound:", upper_bound_hpdi-lower_bound_hpdi)

hist(G, breaks = 100, main = "Credible Interval and HPDI")
abline(v = lower_bound, col = "blue", lwd = 2, lty = 2)
abline(v = upper_bound, col = "blue", lwd = 2, lty = 2)
rect(lower_bound, 0, upper_bound, 
     max(hist(G, breaks = 100, plot = FALSE)$density), col = "blue", border = NA)

abline(v = lower_bound_hpdi, col = "red", lwd = 2, lty = 2)
abline(v = upper_bound_hpdi, col = "red", lwd = 2, lty = 2)
rect(lower_bound_hpdi, 0, upper_bound_hpdi, 
     max(hist(G, breaks = 100, plot = FALSE)$density), col = "red", border = NA)

# QUESTION 3
# part a) -------------------------------------------------------------------------------------

posterior_fnc <- function(k, y, mu, lambda) {
  n <- length(y)
  sum_cos <- sum(cos(y - mu))
  numerator <- exp(k * sum_cos - k * lambda)
  denominator <- (2*pi*besselI(k, 0))^n 
  result<-numerator / denominator
  return(result)
}

y <- c(-2.79, 2.33, 1.83, -2.44, 2.23, 2.33, 2.07, 2.02, 2.14, 2.54)
mu <- 2.4
lambda <- 0.5
k <- seq(0.1, 10, length.out = 1000)
posterior_values <- posterior_fnc(k, y, mu, lambda)

# normalize the posterior values to integrate to one
posterior_values_normalized <- posterior_values / sum(posterior_values * diff(k[1:2]))

plot(k, posterior_values_normalized, type = 'l', lwd = 2, col = 'red', 
     main = "Posterior Distribution", xlab = 'k', ylab = 'Likelihood')

# part b) -------------------------------------------------------------------------------------
max_posterior_index <- which.max(posterior_values_normalized)
cat("k value that maximizes the likelihood:", k[max_posterior_index])

plot(k, posterior_values_normalized, type = 'l', lwd = 2, col = 'red', 
     main = "Posterior Distribution", xlab = 'k', ylab = 'Likelihood')
abline(v = 2.587387, col = "blue", lwd = 2, lty = 2)



