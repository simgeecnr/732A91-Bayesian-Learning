# Bayesian Learning Lab 3
# nazbi056 & simci637

library(BayesLogit)
library(mvtnorm)
library(rstan)

# Question 1

df1 <- read.table("WomenAtWork.dat", header = TRUE)
Covs <- c(2:8) # Select which covariates/features to include
X <- as.matrix(df1[,Covs])
Xnames <- colnames(X)
y <- as.matrix(df1[,1])

nObs <- dim(df1)[1]
nPar <- dim(df1)[2] -1 # subtract y
# Setting up the prior
tau <- 3
b <- as.matrix(rep(0, nPar)) # Prior mean vector
B <- (tau^2)*diag(nPar) # Prior covariance matrix

# part a)

set.seed(12345)
# Generating beta draws
nDraws <- 1000
beta_prior <- matrix(0, nrow = nDraws, ncol = nPar)
beta_gibbs <- matrix(0, nrow = nDraws, ncol = nPar)

calc_w <- function(row, beta_prior) {
  w <- rpg(1, h = 1, z = row %*% t(beta_prior))
  return(w)
}

for (i in 1:nDraws){
  # draws from prior beta
  beta_prior[i,] <- rmvnorm(1, b, B)
  
  # w | beta_prior
  w <- apply(X, 1, calc_w, beta_prior = beta_prior[i,]) 
  
  # beta | y, w
  Omega <- diag(w)
  k <- y - 0.5
  V_w <- solve((t(X) %*% Omega %*% X) + solve(B))
  m_w <- V_w %*% (t(X) %*% k + solve(B) %*% b)
  
  beta_gibbs[i,] <- rmvnorm(1, m_w, V_w)
}

colors <- c("red", "blue", "darkgreen", "brown", "magenta", "orange", "purple")
for (i in 1:nPar){
  par(mfrow=c(1,1))
  # traceplot of Gibbs draws
  plot(1:nDraws, beta_gibbs[,i], 
       type = "l",
       col=colors[i], 
       main = paste("Traceplot of parameter", colnames(df1)[i+1]),
       xlab = "MCMC iteration",
       ylab = "Value") 
  
  par(mfrow=c(1,2))
  #histogram of Gibbs draws
  hist(beta_gibbs[,i],
       col=colors[i], 
       main = "Histogram", 
       xlab = paste(colnames(df1)[i+1])) 
  
  # Cumulative mean value, Gibbs draws
  cusumData =  cumsum(beta_gibbs[,i])/seq(1, nDraws) 
  plot(1:nDraws, cusumData, type = "l", 
       col=colors[i], 
       lwd = 2,
       main = paste("Convergence of", colnames(df1)[i+1]), 
       ylab = "Cumulative Mean", 
       xlab = "MCMC iteration")
}

colors <- c("red", "blue", "darkgreen", "brown", "magenta", "orange", "purple")
for (i in 1:nPar){
  par(mfrow=c(1,1))
  # traceplot of Gibbs draws
  plot(1:nDraws, beta_gibbs[,i], 
       type = "l",
       col=colors[i], 
       main = paste("Traceplot of parameter", colnames(df1)[i+1]),
       xlab = "MCMC iteration",
       ylab = "Value") 
  
  par(mfrow=c(1,2))
  #histogram of Gibbs draws
  hist(beta_gibbs[,i],
       col=colors[i], 
       main = "Histogram", 
       xlab = paste(colnames(df1)[i+1])) 
  
  # Cumulative mean value, Gibbs draws
  cusumData =  cumsum(beta_gibbs[,i])/seq(1, nDraws) 
  plot(1:nDraws, cusumData, type = "l", 
       col=colors[i], 
       lwd = 2,
       main = paste("Convergence of", colnames(df1)[i+1]), 
       ylab = "Cumulative Mean", 
       xlab = "MCMC iteration")
}

IF_Gibbs <- rep(NA, nPar)

for (i in 1:nPar) {
  a_Gibbs <- acf(beta_gibbs[,i], lag.max = 5, plot = FALSE)
  IF_Gibbs[i] <- 1 + 2 * sum(a_Gibbs$acf[-1])
}

cat("Inefficiency Factors:\n", IF_Gibbs)

# Part b)

input <- c(1, 22, 12, 7, 38, 1, 0)
X_input <- as.matrix(input, ncol = 1)

sigmoid_fnc <- function(linPred){
  return(exp(linPred)/(1+exp(linPred)))
}

linPred <- beta_gibbs %*% X_input
result <- apply(linPred, 1, sigmoid_fnc)
credible_interval <- quantile(result, c(0.05, 0.95))
cat("90% credible interval: (", credible_interval, ")")

hist(result, main = "Posterior of predictive distribution of P(y=1|x)")
abline(v = credible_interval, col = "red", lwd = 2, lty = 2)

# ------------------------------------------------------------------------------------------------
# Question 2

df2 <- read.table("eBayNumberOfBidderData_2024.dat", header = TRUE)

# part a)
# maximum likelihood estimator of beta
glmModel <- glm(nBids ~ 0 + ., data = df2, family = "poisson")
summary(glmModel)
cat("Model coefficients:\n", glmModel$coefficients)

# part b)
y <- as.matrix(df2$nBids) # y is (n x 1) matrix
X <- as.matrix(df2[2:ncol(df2)]) # X is (n x p) matrix
nObs <- nrow(df2)
nPar <- ncol(df2) - 1 # subtract y 

# prior hyperparameters
mu_prior <- rep(0, nPar)
Sigma_prior <- 100 * solve((t(X) %*% X))

# Now we will use optim. Inputs are;
# 1) log p(theta|y) function
# 2) initial values for thetas

# input 1) 
logPostFunc <- function(betas, y, X, mu, Sigma){
  lambda <- exp(X %*% betas) # (n x p)(p x 1) matrix multiplication
  logLik <- sum( log( ((lambda^y) * exp(-lambda) )/ factorial(y) ) )
  logPrior <- dmvnorm(betas, mu, Sigma, log=TRUE) # densities are given as log, calculates density
  return(logLik + logPrior)
}

# input 2) 
initVal <- matrix(0, nPar, 1) # (p x 1) matrix

# Now optimize
OptimRes <- optim(initVal, logPostFunc, gr=NULL, y, X, mu_prior, Sigma_prior, method=c("BFGS"), 
                  control=list(fnscale=-1), hessian=TRUE)

Xnames <- colnames(df2[2:ncol(df2)])
posterior_mode <- t(OptimRes$par)
colnames(posterior_mode) <- Xnames
print('The posterior mode is:')
print(posterior_mode)

J_y_inv <- solve(-OptimRes$hessian)
approxPostStd <- sqrt(diag(J_y_inv))
names(approxPostStd) <- Xnames
print('The approximate posterior standard deviation is:')
print(approxPostStd)

model_coef <- as.data.frame(glmModel$coefficients)
colnames(model_coef) <- 'glmModel'
model_coef$PosteriorMode <- t(posterior_mode)
print(model_coef)

# part c)

set.seed(12345)
MetropolisHasting <- function(y, X, mu_prior, Sigma_prior, logPostFunc, c, nDraws = 10000){
  
  sum_acc_prob <- 0
  theta_sample <- matrix(0, nrow = nDraws, ncol = nPar) # initialize draws
  theta_t <- rep(0, nPar) # initialize theta
  
  for (s in 1:nDraws){
    # step 1: sample from proposal distribution
    theta_t1 <- as.vector(rmvnorm(1, mean = theta_t, sigma = c * J_y_inv))
    
    # step 2: compute acceptance probability,
    log_density_t <- logPostFunc(theta_t, y, X, mu_prior, Sigma_prior)
    log_density_t1 <- logPostFunc(theta_t1, y, X, mu_prior, Sigma_prior)
    
    alpha <- exp(log_density_t1 - log_density_t)
    acc_prob <- min(1, alpha)
    sum_acc_prob <- sum_acc_prob + acc_prob
    # step 3:
    if (runif(1) < acc_prob){
      # accept
      theta_sample[s,] <- theta_t1
      theta_t <- theta_t1
    } else{
      # reject
      theta_sample[s,] <- theta_t
    }
  }
  
  cat("Average acceptance probability:", sum_acc_prob / nDraws, "\n")
  return(theta_sample[-1,])
}

nDraws <- 10000
c <- 0.6 # set c that average acceptance probability should is 25-30%
beta_samples <- MetropolisHasting(y, X, mu_prior, Sigma_prior, logPostFunc, c, nDraws)

par(mfrow = c(3, 3)) 
for (p in 1:nPar){
  plot(beta_samples[, p], 
       type = "l", 
       main = paste("Trajectory for Parameter", colnames(df2)[p+1]), 
       xlab = "MCMC iteration",
       ylab = "Value")
}

burn_in <- 500
beta_avg_values <- rep(0, nPar)

colors <- c("red", "blue", "darkgreen", "brown", "magenta", "orange", "purple", "yellow", "cyan")
for (i in 1:nPar){
  par(mfrow = c(1, 2)) 
  beta_avg <- cumsum(beta_samples[, i]) / (1:nrow(beta_samples))
  beta_avg_values[i] <- beta_avg[length(beta_avg)]
  
  plot(1:nrow(beta_samples), beta_avg, 
       type = "l", col=colors[i], 
       main = paste("Convergence of", colnames(df2)[i+1]), 
       lwd = 2.5, 
       xlab = "MCMC iteration",
       ylab = "Value")
  abline(h = glmModel$coefficients[i], col = "blue", lwd = 2, lty = 2)  
  
  hist(beta_samples[burn_in: nrow(beta_samples), i], 
       main = "Histogram without burn-in period", 
       xlab =  paste(colnames(df2)[i+1]), col = colors[i])
}

# part d)
X_input <- c(Const = 1, 
             PowerSeller = 1, 
             VerifyID = 0, 
             Sealed = 1, 
             Minblem = 0, 
             MajBlem = 1, 
             LargNeg = 0, 
             LogBook = 1.2, 
             MinBidShare = 0.8) 

pred_beta <- beta_samples[500:nrow(beta_samples),] # burn-in period is removed

# column 1: lambda, column 2: number of bids
generated_pois <- matrix(NA, nrow = nrow(pred_beta), ncol = 2) 
generated_pois[,1] <- exp(pred_beta %*% as.matrix(X_input)) # calculating lambda values

for (i in 1:nrow(pred_beta)){
  lambda <- generated_pois[i,1]
  generated_pois[i,2] <- rpois(1, lambda)
}

cat("Probability of no bidders:", sum(generated_pois[,2] == 0) / nrow(generated_pois))

hist(generated_pois[,2], 
     breaks = seq(min(generated_pois[,2])-0.5, max(generated_pois[,2])+0.5, by = 1),  
     col = "orange",  
     main = "Predictive Distribution",  
     xlab = "Values",  
     ylab = "Frequency"
)

cat("Number of expected bids from the glmModel:", predict(glmModel, newdata = as.data.frame(t(X_input)), type = "response"))

# ------------------------------------------------------------------------------------------------
# Question 3

# part a)
AR1_simulate <- function(phi, mu, sigma2, T){
  x_vector <- rep(NA, T)
  x_t <- mu
  x_vector[1] <- x_t
  for (i in 2:T){
    eps <- rnorm(1, mean = 0, sd = sqrt(sigma2))
    x_t1 <- mu + phi*(x_t - mu) + eps
    x_vector[i] <- x_t1
    x_t <- x_t1
  }    
  return(x_vector)
}

par(mfrow = c(3, 2))
phi_val <- seq(-1, 1, length.out = 6)
for (i in 1:length(phi_val)){
  x_simulation <- AR1_simulate(phi = phi_val[i], mu = 9, sigma2 = 4, T = 250)
  plot(x_simulation, 
       type="l", 
       xlab = "x_T", ylab = "Value",
       main = paste("phi =", phi_val[i]))
}

# part b)

# i) phi = 0.3
y = AR1_simulate(phi = 0.3, mu = 9, sigma2 = 4, T = 250)
N=length(y)

StanModel = '
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=-1, upper=1> phi;
  real<lower=0> sigma2;
}
model {
  mu ~ normal(9, 100); // non-informative prior
  phi ~ uniform(-1, 1); 
  sigma2 ~ scaled_inv_chi_square(0.0001, 0.0001); // as nu, tau^2 -> 0, it becomes jeffreys prior
  
  for (n in 2:N) { 
    y[n] ~ normal(mu + phi * (y[n-1] - mu), sqrt(sigma2));
  }
}'

data <- list(N=N, y=y)
warmup <- 1000
niter <- 2000
fit <- stan(model_code=StanModel, data=data, warmup=warmup, iter=niter, chains=4)

# Print the fitted model
print(fit, digits_summary=3)

# Extract posterior samples
postDraws <- extract(fit)

fit_summary <- summary(fit)
posterior_mean <- fit_summary$summary[, "mean"]
credible_intervals <- fit_summary$summary[, c("2.5%", "97.5%")]
effective_samples <- fit_summary$summary[, "n_eff"]

cat("posterior mean:\n")
print(posterior_mean[1:3])

cat("95% credible interval:\n")
print(credible_intervals[1:3, ])

cat("Effective sample size:\n")
print(effective_samples[1:3])

traceplot(fit)

mu_draws <- postDraws$mu
cum_mean_mu <- cumsum(mu_draws) / (1:length(mu_draws))

par(mfrow = c(1, 2))
plot(1:length(mu_draws), cum_mean_mu, lwd = 2,
     type = "l", col = "red", main = "Convergence for mu",
     xlab = "Total Draws", ylab = "Cumulative mean")
abline(h = posterior_mean[1], lwd = 2, lty = 2, col = "blue")

hist(mu_draws, main = "Histogram of mu draws")
abline(v = credible_intervals[1,1], lty = 2, lwd = 2, col = "red")
abline(v = credible_intervals[1,2], lty = 2, lwd = 2, col = "red")

phi_draws <- postDraws$phi
cum_mean_phi <- cumsum(phi_draws) / (1:length(phi_draws))

par(mfrow = c(1, 2))
plot(1:length(phi_draws), cum_mean_phi, lwd = 2, 
     type = "l", col = "red", main = "Convergence for phi",
     xlab = "Total draws", ylab = "Cumulative mean")
abline(h = posterior_mean[2], lwd = 2, lty = 2, col = "blue")

hist(phi_draws, main = "Histogram of phi draws")
abline(v = credible_intervals[2,1], lty = 2, lwd = 2, col = "red")
abline(v = credible_intervals[2,2], lty = 2, lwd = 2, col = "red")

sigma2_draws <- postDraws$sigma2
cum_mean_sigma2 <- cumsum(sigma2_draws) / (1:length(sigma2_draws))

par(mfrow = c(1, 2))
plot(1:length(sigma2_draws), cum_mean_sigma2, lwd = 2,
     type = "l", col = "red", main = "Convergence for sigma^2",
     xlab = "Total draws", ylab = "Cumulative mean")
abline(h = posterior_mean[3], lwd = 2, lty = 2, col = "blue")

hist(sigma2_draws, main = "Histogram of sigma^2 draws")
abline(v = credible_intervals[3,1], lty = 2, lwd = 2, col = "red")
abline(v = credible_intervals[3,2], lty = 2, lwd = 2, col = "red")

cat("mu converges to", cum_mean_mu[length(cum_mean_mu)], "while true value is 9\n")
cat("phi converges to", cum_mean_phi[length(cum_mean_phi)], "while true value is 0.3\n")
cat("sigma^2 converges to",cum_mean_sigma2[length(cum_mean_sigma2)], "while true value is 4\n")

# Joint posterior draws
plot(postDraws$mu, postDraws$phi, xlab = "mu", ylab = "phi")

# case ii) phi = 0.97

y = AR1_simulate(phi = 0.97, mu = 9, sigma2 = 4, T = 250)
N=length(y)

StanModel = '
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=-1, upper=1> phi;
  real<lower=0> sigma2;
}
model {
  mu ~ normal(9, 100); 
  phi ~ uniform(-1, 1); 
  sigma2 ~ scaled_inv_chi_square(0.0001, 0.0001); // as nu, tau^2 -> 0, it becomes jeffreys prior
  
  for (n in 2:N) { 
    y[n] ~ normal(mu + phi * (y[n-1] - mu), sqrt(sigma2));
  }
}'

data <- list(N=N, y=y)
warmup <- 1000
niter <- 2000
fit <- stan(model_code=StanModel,data=data, warmup=warmup,iter=niter,chains=4)

# Print the fitted model
print(fit, digits_summary=3)

# Extract posterior samples
postDraws <- extract(fit)

fit_summary <- summary(fit)
posterior_mean <- fit_summary$summary[, "mean"]
credible_intervals <- fit_summary$summary[, c("2.5%", "97.5%")]
effective_samples <- fit_summary$summary[, "n_eff"]

cat("posterior mean:\n")
print(posterior_mean[1:3])

cat("95% credible interval:\n")
print(credible_intervals[1:3, ])

cat("Effective sample size:\n")
print(effective_samples[1:3])

traceplot(fit)

mu_draws <- postDraws$mu
cum_mean_mu <- cumsum(mu_draws) / (1:length(mu_draws))

par(mfrow = c(1, 2))
plot(1:length(mu_draws), cum_mean_mu, 
     type = "l", col = "red", main = "Convergence for mu",
     xlab = "Total draws", ylab = "Cumulative mean")
abline(h = posterior_mean[1], lwd = 2, lty = 2, col = "blue")

hist(mu_draws, main = "Histogram of mu draws")
abline(v = credible_intervals[1,1], lty = 2, lwd = 2, col = "red")
abline(v = credible_intervals[1,2], lty = 2, lwd = 2, col = "red")

phi_draws <- postDraws$phi
cum_mean_phi <- cumsum(phi_draws) / (1:length(phi_draws))

par(mfrow = c(1, 2))
plot(1:length(phi_draws), cum_mean_phi, 
     type = "l", col = "red", main = "Convergence for phi",
     xlab = "Total draws", ylab = "Cumulative mean")
abline(h = posterior_mean[2], lwd = 2, lty = 2, col = "blue")

hist(phi_draws, main = "Histogram of phi draws")
abline(v = credible_intervals[2,1], lty = 2, lwd = 2, col = "red")
abline(v = credible_intervals[2,2], lty = 2, lwd = 2, col = "red")

sigma2_draws <- postDraws$sigma2
cum_mean_sigma2 <- cumsum(sigma2_draws) / (1:length(sigma2_draws))

par(mfrow = c(1, 2))
plot(1:length(sigma2_draws), cum_mean_sigma2, 
     type = "l", col = "red", main = "Convergence for sigma^2",
     xlab = "Total draws", ylab = "Cumulative mean")
abline(h = posterior_mean[3], lwd = 2, lty = 2, col = "blue")

hist(sigma2_draws, main = "Histogram of sigma^2 draws")
abline(v = credible_intervals[3,1], lty = 2, lwd = 2, col = "red")
abline(v = credible_intervals[3,2], lty = 2, lwd = 2, col = "red")

cat("mu converges to", cum_mean_mu[length(cum_mean_mu)], "while true value is 9\n")
cat("phi converges to", cum_mean_phi[length(cum_mean_phi)], "while true value is 0.97\n")
cat("sigma^2 converges to",cum_mean_sigma2[length(cum_mean_sigma2)], "while true value is 4\n")

# Joint posterior draws
plot(postDraws$mu, postDraws$phi, xlab = "mu", ylab = "phi")
















