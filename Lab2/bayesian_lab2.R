# nazbi056 & simci637

library(readxl)
library(mvtnorm)

# Question 1: Linear and polynomial regression

df <- read_xlsx("Linkoping2022.xlsx")
df$time <- 1:nrow(df) / 365 

n <- nrow(df)
x1 <- rep(1, times = n)
x2 <- df$time
x3 <- (df$time)^2

X <- cbind(x1, x2, x3)
y <- matrix(df$temp, ncol = 1)

# Part a)

set.seed(123)
# Given prior hyperparameters

# mean and standard deviation for guassian distribution
mu0 <- matrix(c(0,100,-100), ncol = 1)
omega0 <- 0.01 * diag(3)

# number of chi-squared degrees of freedom and the scaling parameter for inverse chi-square
nu0 <- 1
sigma0_sqr <- 1

# Getting joint prior for beta and sigma^2
nDraws <- 10
sigma_new <- matrix(ncol=1, nrow=nDraws)
beta_new <- matrix(ncol=3, nrow=nDraws)

for (i in 1:nDraws){
  sigma_new[i,] <- (nu0*sigma0_sqr)/ rchisq(1, nu0)
  beta_new[i,] <- rmvnorm(1, mu0, sigma_new[i,]*solve(omega0))   
}

y_prior <- matrix(nrow = n, ncol = nDraws)
for (l in 1:nDraws){
  err <- rnorm(1, mean = 0, sd = 1)
  y_prior[,l] <- X[,1] * beta_new[l,1] + X[,2] * beta_new[l,2] + X[,3] * beta_new[l,3] + err
}

plot(df$time, df$temp, ylim = c(-40, 100), ylab = 'Temperature', xlab = 'Time')
colors <- c("red", "blue", "green", "cyan", "magenta", "yellow", "orange", "purple", 
            "brown", "darkgreen")
for (i in 1:nDraws){
  lines(df$time, y_prior[,i], col = colors[i], type = 'l', lwd = 2)
}

cat("Minimum variance is", min(sigma_new), "\n",
    "Maximum variance is", max(sigma_new), "\n")

set.seed(123)
# Selected prior hyperparameters

# mean and standard deviation for guassian distribution
mu0 <- matrix(c(-10,100,-100), ncol = 1)
omega0 <- 0.01 * diag(3)

# number of chi-squared degrees of freedom and the scaling parameter for inverse chi-square
nu0 <- 100
sigma0_sqr <- 0.2

# Getting joint prior for beta and sigma^2
nDraws <- 10
sigma_new <- matrix(ncol=1, nrow=nDraws)
beta_new <- matrix(ncol=3, nrow=nDraws)

for (i in 1:nDraws){
  sigma_new[i,] <- (nu0*sigma0_sqr)/ rchisq(1, nu0)
  beta_new[i,] <- rmvnorm(1, mu0, sigma_new[i,]*solve(omega0))   
}

y_prior <- matrix(nrow = n, ncol = nDraws)
for (l in 1:nDraws){
  err <- rnorm(1, mean = 0, sd = 1)
  y_prior[,l] <- X[,1] * beta_new[l,1] + X[,2] * beta_new[l,2] + X[,3] * beta_new[l,3] + err
}

plot(df$time, df$temp, ylim = c(-40, 40), ylab = 'Temperature', xlab = 'Time')
colors <- c("red", "blue", "green", "cyan", "magenta", "yellow", "orange", "purple", 
            "brown", "darkgreen")
for (i in 1:nDraws){
  lines(df$time, y_prior[,i], col = colors[i], type = 'l', lwd = 2)
}

cat("Minimum variance is", min(sigma_new), "\n",
    "Maximum variance is", max(sigma_new), "\n")


# Part b)

beta_hat <- solve(t(X) %*% X) %*% t(X) %*% y

mu_n <- solve(t(X) %*% X + omega0) %*% ((t(X) %*% X) %*% beta_hat) + 
  (omega0 %*% mu0)
omega_n <- t(X) %*% X + omega0 
nu_n <- nu0 + n 
sigma_n_sqr <- (nu0 %*% sigma0_sqr + (t(y) %*% y) + 
                  (t(mu0) %*% omega0 %*% mu0) - (t(mu_n) %*% omega_n %*% mu_n)) / nu_n 


get_posterior_parameters <- function(nDraws){
  sigma_beta_posterior <- matrix(ncol=4, nrow=nDraws)
  
  for (i in 1:nDraws){
    sigma_beta_posterior[i,1] <- (nu_n*sigma_n_sqr)/ rchisq(1, nu_n)
    sigma_beta_posterior[i,2:4] <- rmvnorm(1, mu_n, sigma_beta_posterior[i,1]*solve(omega_n)) 
  }  
  return(sigma_beta_posterior)
}

posterior <- get_posterior_parameters(nDraws = 10000)
sigma_posterior <- posterior[,1]
beta_posterior <- posterior[,2:4]

hist(sigma_posterior, breaks = 30, main = "Posterior of Sigma")
hist(beta_posterior[,1], breaks = 30, main = "Posterior of Beta_0")
hist(beta_posterior[,2], breaks = 30, main = "Posterior of Beta_1")
hist(beta_posterior[,3], breaks = 30, main = "Posterior of Beta_2")

y_posterior <- beta_posterior %*% t(X) # getting predictions
y_posterior_median <- apply(y_posterior, 2, median) # median of the predictions

# calculating 90% equal tail probability 
posterior_5th_percentile <- apply(y_posterior, 2, quantile, probs = 0.05)
posterior_95th_percentile <- apply(y_posterior, 2, quantile, probs = 0.95)

plot(df$time, df$temp, ylab = 'Temperature', xlab = 'Time')
lines(df$time, y_posterior_median, col = 'red', lwd = 2)
lines(df$time, posterior_5th_percentile, col = 'blue', lwd = 2)
lines(df$time, posterior_95th_percentile, col = 'blue', lwd = 2)

# Part c)

# First derivative is 0 since it is a parabola
f_time_max <- -beta_posterior[,2] / (2*beta_posterior[,3])
hist(f_time_max, main = "Posterior distribution of time for the highest expected temp.")

plot(df$time, df$temp, ylab = 'Temperature', xlab = 'Time')
lines(df$time, y_posterior_median, col = 'red', lwd = 2)
abline(v = median(f_time_max), lwd = 3, lty = 2, col = 'blue')

# Part d)

#Too many variables leads to overfitting hence we will use regularization prior for higher 
#order terms. Larger $\lambda$ value gives smoother fit. We choose our priors as 
#\mu_0 = (-10,100,-100, 0, 0, 0, 0, 0, 0, 0, 0)  and 
#\Omega_0 = (0.01, 0.01, 0.01, 100, 100, 100, 100, 100, 100, 100, 100)

# Question 2: Posterior approximation for classification with logistic regression

df2 <- read.table("WomenAtWork.dat", header = TRUE)

# Part a)

Covs <- c(2:8) # Select which covariates/features to include
X <- as.matrix(df2[,Covs])
Xnames <- colnames(X)
y <- as.matrix(df2[,1])

nObs <- dim(df2)[1]
nPar <- dim(df2)[2] -1 # subtract y

# Setting up the prior
tau <- 2
mu <- as.matrix(rep(0, nPar)) # Prior mean vector
Sigma <- (tau^2)*diag(nPar) # Prior covariance matrix

# Now we will use optim. Inputs are;
# 1) log p(theta|y) function
# 2) initial values for betas

# input 1) 
LogPostLogistic <- function(betas, y, X, mu, Sigma){
  linPred <- X%*%betas
  logLik <- sum( linPred*y - log(1 + exp(linPred)) )
  logPrior <- dmvnorm(betas, mu, Sigma, log=TRUE) # densities are given as log, calculates density
  return(logLik + logPrior)
}

# input 2) 
initVal <- matrix(0, nPar, 1)

# Now optimize
OptimRes <- optim(initVal, LogPostLogistic, gr=NULL, y, X, mu, Sigma, method=c("BFGS"), 
                  control=list(fnscale=-1), hessian=TRUE)

# Printing the results to the screen
posterior_mode <- t(OptimRes$par)
colnames(posterior_mode) <- Xnames
approxPostStd <- sqrt(diag(solve(-OptimRes$hessian))) # Computing approximate standard deviations.
names(approxPostStd) <- Xnames # Naming the coefficient by covariates

print('The posterior mode is:')
print(posterior_mode)

print('The approximate posterior standard deviation is:')
print(approxPostStd)

lower_bound <- OptimRes$par[6] - 1.96*approxPostStd[6]
upper_bound <- OptimRes$par[6] + 1.96*approxPostStd[6]
cat("95% credible interval is [", lower_bound, ",", upper_bound, "]")

# testing approximation
glmModel <- glm(Work ~ 0 + ., data = df2, family = binomial)
glmModel$coefficients

# Part b)

mu_posterior <- OptimRes$par
Sigma2_posterior <- solve(-OptimRes$hessian)

input <- c(1, 18, 11, 7, 40, 1, 1)
X_input <- as.matrix(input, ncol = 1)

nDraws <- 10000

generated_beta <- rmvnorm(nDraws, mean = mu_posterior, sigma = Sigma2_posterior)
linPred <- generated_beta %*% X_input

sigmoid_fnc <- function(linPred){
  return(1/(1+exp(linPred)))
}

result <- apply(linPred, 1, sigmoid_fnc)
hist(result, main = "Posterior of predictive distribution of P(y=0|x)")

# Part c)

nDraws <- 10000
pred_draw <- rep(nDraws, 0)

for (i in 1:nDraws){
  # generate posterior draw
  generated_beta <- rmvnorm(1, mean = mu_posterior, sigma = Sigma2_posterior)
  linPred <- generated_beta %*% X_input
  prob_success <- 1/(1+exp(linPred))
  
  # generate predictive draw
  pred_draw[i] <- rbinom(n = 1, size = 13, prob = prob_success)
}

hist(pred_draw, main = "Posterior predictive dist. for # of women not working")




