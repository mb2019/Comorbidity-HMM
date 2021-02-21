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

# Transition probabilities: Covar=0
(Gamma0 <- t(matrix(c(0.70, 0.10, 0.10, 0.10, 
                      0.10, 0.70, 0.10, 0.10, 
                      0.10, 0.10, 0.70, 0.10, 
                      0.10, 0.10, 0.10, 0.70), K,K)))

# Transition probabilities: Covar=1
(Gamma1 <- t(matrix(c(0.97, 0.01, 0.01, 0.01, 
                      0.01, 0.97, 0.01, 0.01, 
                      0.01, 0.01, 0.97, 0.01, 
                      0.01, 0.01, 0.01, 0.97), K,K)))

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
  
  # Following observations: Covar=0
  for (t in 2:(nobs/2)){
    S_ind[t] <- sample(1:K, size=1, prob=Gamma0[S_ind[t-1],])
  }
  
  # Following observations: Covar=1
  for (t in (nobs/2+1):nobs){
    S_ind[t] <- sample(1:K, size=1, prob=Gamma1[S_ind[t-1],])
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

# State-dependent mus
mu1 <- c(5, 10)
mu2 <- c(5, 10)

# Simulate observations
for(t in 1:ntot){
  y1[t] <- rnorm(1, mu1[S1[t]], 1)
  y2[t] <- rnorm(1, mu2[S2[t]], 1)
}

# Covariates
covar <- cbind(rnorm(ntot), rnorm(ntot),                       # Random
               rep(c(rep(0, nobs/2), rep(1, nobs/2)), nind))   # Covar

# Stan Data
stan_data <- stan_list_chmm(y1=y1, y2=y2, k1=2, k2=2, covar=covar, ID=ID)

# Fit
fit <- stan(file="CHMM_comorbidity.stan", data=stan_data,
            pars=c("mu1", "mu2", "sigma1", "sigma2", "alpha", "beta", "z_star"),
            chains = 4, iter = 3000, seed = 123, control = list(max_treedepth = 15))

# Results
print(fit, probs = c(0.05, 0.95), pars="z_star", include=F)

# Samples
samples <- extract(fit)

# Transition probability matrix (based on posterior means)
covs <- cbind(1, scale(stan_data$Covar, center = TRUE, scale=FALSE))
coefs <- cbind(colMeans(samples$alpha), colMeans(samples$beta))
tpms <- moveHMM:::trMatrix_rcpp(nbStates=4, beta = t(coefs), covs=covs)

# Average transition probability matrix
round(apply(tpms[,,stan_data$Covar[,3]==0], FUN=mean, MARGIN=1:2), 2)         # Covar == 0
round(apply(tpms[,,stan_data$Covar[,3]==1], FUN=mean, MARGIN=1:2), 2)         # Covar == 1

# Difference transition probability matrix
round(apply(tpms[,,stan_data$Covar[,3]==0], FUN=mean, MARGIN=1:2)-Gamma0, 2)  # Covar == 0
round(apply(tpms[,,stan_data$Covar[,3]==1], FUN=mean, MARGIN=1:2)-Gamma1, 2)  # Covar == 1

# Viterbi (most frequent value)
z_star <- apply(samples$z_star, FUN=mfv1, MARGIN=2)
sum(z_star==S)