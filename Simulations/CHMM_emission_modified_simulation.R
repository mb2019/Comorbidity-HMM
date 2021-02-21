library(modeest)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
source("helper-functions.R")
set.seed(123)

# Number of states of process 1
k1 <- 2

# Number of states of process 2
k2 <- 2

# Number of states of cartesian product of both states
K <- 4

# Transition probabilities
(Gamma0 <- t(matrix(c(0.70, 0.10, 0.10, 0.10, 
                      0.10, 0.70, 0.10, 0.10, 
                      0.10, 0.10, 0.70, 0.10, 
                      0.10, 0.10, 0.10, 0.70), K,K)))

# Initial distribution
delta <- c(0.25, 0.25, 0.25, 0.25)

# Number of observations per individual
nobs <- 30

# Number of individuals
nind <- 100

# Total number of observations
ntot <- nobs*nind

# ID of individuals
ID <- rep(1:nind, each=nobs)

# States
S <- c() 

# Loop through individuals
for (i in 1:nind){
  S_ind <- rep(NA, nobs)
  
  # First observation
  S_ind[1] <- sample(1:K, size=1, prob=delta)
  
  # Following observations
  for (t in 2:nobs){
    S_ind[t] <- sample(1:K, size=1, prob=Gamma0[S_ind[t-1],])
  }
  
  # Append results
  S <- append(S, S_ind)
}

# Mapping states
(mapping <- matrix(c(rep(1:k1, each=k2), rep(1:k2, k1)), ncol = 2))

# Map states to "original" two channels
S1 <- rep(NA,nobs) 
S2 <- rep(NA,nobs)

for (i in 1:ntot){
  S1[i] <- mapping[S[i],1] 
  S2[i] <- mapping[S[i],2] 
}

# Vectors for observations
y1 <- rep(NA, ntot) 
y2 <- rep(NA, ntot)

# Covariates
covar <- cbind(rnorm(ntot), rnorm(ntot),                       # Random
               rep(c(rep(0, nobs/2), rep(1, nobs/2)), nind))   # Covar

# State-dependent mus based on covariate
mu1 <- list(c(5, 10), c(7.5, 12.5))
mu2 <- list(c(5, 10), c(7.5, 12.5))

# Simulate observations
for(t in 1:ntot){
  y1[t] <- rnorm(1, mu1[[covar[t,3]+1]][S1[t]], 1)
  y2[t] <- rnorm(1, mu2[[covar[t,3]+1]][S2[t]], 1)
}

# Stan Data
stan_data <- stan_list_chmm(y1=y1, y2=y2, k1=2, k2=2, covar=covar, ID=ID)

# Parameters
params <- c("alpha1", "alpha2", 
            "beta1", "beta2",
            "sigma1", "sigma2",
            "pi_initial", "gamma",
            "z_star")

# Function for initial values based on kmeans
init_fn <- function() {
  list(alpha1 = sort(kmeans(stan_data$y1, centers = stan_data$k1)$centers), 
       alpha2 = sort(kmeans(stan_data$y2, centers = stan_data$k2)$centers),
       theta1 = matrix(data=0, nrow=stan_data$k1, ncol=stan_data$C),
       theta2 = matrix(data=0, nrow=stan_data$k2, ncol=stan_data$C))
}

# Fit
fit <- stan(file="CHMM_emission_modified.stan", 
            data=stan_data,
            pars=params, 
            init = init_fn,
            chains = 4, iter = 3000, seed = 123, control = list(max_treedepth = 15))

# Results
print(fit, probs = c(0.05, 0.95), pars="z_star", include=F)

# Summary
fit_summary <- summary(fit)$summary

# Mu of Process 1 / State 1 / Covar = 0 (-0.5 due to centering)
fit_summary["alpha1[1]", "mean"] + (-0.5)*fit_summary["beta1[1,3]", "mean"]

# Mu of Process 2 / State 2 / Covar = 1 (0.5 due to centering)
fit_summary["alpha2[2]", "mean"] + (0.5)*fit_summary["beta2[2,3]", "mean"]

# Samples
samples <- extract(fit)

# Viterbi (most frequent value)
z_star <- apply(samples$z_star, FUN=mfv1, MARGIN=2)
sum(z_star==S)