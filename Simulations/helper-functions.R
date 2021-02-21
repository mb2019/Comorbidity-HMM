# stan_list_chmm():
# Function that creates stan data list suitable for CHMM.

stan_list_chmm <- function(y1, y2, k1, k2, covar, ID){
  
  # States
  K <- k1*k2                        # Overall number of states
  map_proc1 <- rep(1:k1, each=k2)   # State mapping of process 1
  map_proc2 <- rep(1:k2, k1)        # State mapping of process 2
  
  # Data
  N <- length(y1)                         # Number of observations
  nind <- length(unique(ID))              # Number of individuals
  ID <- factor(ID, labels = c(1:nind))    # Identifier of individuals
  ID <- as.numeric(ID)
  Covar <- covar                          # Covariates
  C <- ncol(Covar)                        # Number of covariates
  y1 <- y1                                # Observation of process 1
  y2 <- y2                                # Observation of process 2
  
  return(list(# States & Mapping
              K=K, k1=k1, k2=k2, map_proc1=map_proc1, map_proc2=map_proc2,
    
              # Data
              N=N, nind=nind, ID=ID, Covar=Covar, C=C, y1=y1, y2=y2))
}
