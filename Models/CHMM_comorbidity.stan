data {
  
  // States
  int<lower=0> k1;          // Number of states of first process
  int<lower=0> k2;          // Number of states of second process
  int<lower=0> K;           // Overall number of states
  int map_proc1[K];         // Vector for state mapping of process 1
  int map_proc2[K];         // Vector for state mapping of process 2
  
  // Data
  int<lower=0> N;           // Length of observations/states
  int ID[N];                // Identifier of individuals
  int nind;                 // Number of individuals
  vector[N] y1;             // Observations of first process
  vector[N] y2;             // Observations of second process
  int<lower=0> C;           // Number of covariates excluding intercept
  matrix[N, C] Covar;       // Covariate matrix excluding intercept
  
}

transformed data {
  
  // Centering
  matrix[N, C] Covar_cent;
  
  // QR Decomposition
  matrix[N, C] Q_ast;
  matrix[C, C] R_ast;
  matrix[C, C] R_ast_inverse;
  
  // Centering
  for (i in 1:C) {
    Covar_cent[,i] = Covar[,i] - mean(Covar[,i]);
  }
  
  // QR
  Q_ast = qr_thin_Q(Covar_cent) * sqrt(N - 1);
  R_ast = qr_thin_R(Covar_cent) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
  
}

parameters {
  
  positive_ordered[k1] mu1;  // Mus of normal distribution of first process
  positive_ordered[k2] mu2;  // Mus of normal distribution of second process 
  real<lower=0> sigma1;      // Sigma of normal distribution of first process
  real<lower=0> sigma2;      // Sigma of normal distribution of second process
  simplex[K] pi_initial;     // Initial state probabilities
  matrix[K*(K-1), C] theta;  // Thetas for transition probabilities
  real alpha[K*(K-1)];       // Alphas (Intercepts) for transition probabilities
  
}

transformed parameters{
  
  // Transition probability matrices
  matrix[K,K] gamma[N];           // individual transition probability matrix for each t
  matrix[K,K] log_gamma[N];       // log
  matrix[K,K] log_gamma_tr[N];    // transpose
  
  // Log_lik
  vector[nind] log_lik;
  
  /////////////
  //// tpm ////
  /////////////
  
  // Derive array of (log-) transition prob. matrices
  for (t in 1:N) { 
    int row_tpm = 1;
    
    for (i in 1:K){
      for (j in 1:K){
        if(i==j) {
          gamma[t,i,j] = 1;
        } else {
            gamma[t,i,j] = exp(alpha[row_tpm] + Q_ast[t]*to_vector(theta[row_tpm]));
            row_tpm += 1;
          }
        }
      }
    
    // Rows must sum up to 1
    for (i in 1:K){
      log_gamma[t][i] = log(gamma[t][i]/sum(gamma[t][i]));
    }
  }  
    
  // Transpose log gamma (transition prob. matrix)
  for (t in 1:N){
      log_gamma_tr[t] = log_gamma[t]'; 
  }
  
  ///////////////
  /// Log_lik ///
  ///////////////
  
 {
  // Variables for forward algorithm & log_lik
  vector[K] lp;
  vector[K] lp_p1;
  int iter_ind = 1; 
  
  // Forward algorithm
  for (t in 1:N) {
    
    // First observation of each individual
    if (t==1 || ID[t] != ID[t-1]){
      
      for (k in 1:K) // initial state prob. + emission prob.
          lp[k] = log(pi_initial[k]) +
                  normal_lpdf(y1[t] | mu1[map_proc1[k]], sigma1) + 
                  normal_lpdf(y2[t] | mu2[map_proc2[k]], sigma2); 
    }
    
    // Following observations
    for (k in 1:K)  // Previous forward path prob. + transition prob. + emission prob.
        lp_p1[k] = log_sum_exp(to_vector(log_gamma_tr[t,k]) + lp) +
                               normal_lpdf(y1[t] | mu1[map_proc1[k]], sigma1) + 
                               normal_lpdf(y2[t] | mu2[map_proc2[k]], sigma2);
      lp=lp_p1;
      
    // Add forward path prob to target at end of each individual
     if (t==N || ID[t] != ID[t+1]){
       
        // Log_lik per inidividual
        log_lik[iter_ind] = log_sum_exp(lp);
        iter_ind += 1;
        
        }
    }
  }
}

model {
  
  // Forward algorithm: Sum of each individual log_lik
  target += sum(log_lik);
  
  // Prior initial state probabilities
  pi_initial ~ dirichlet(rep_vector(1, K));

  // Prior mus
  mu1 ~ normal(0, 10);
  mu2 ~ normal(0, 10);
  
  // Prior sigmas
  sigma1 ~ std_normal();
  sigma2 ~ std_normal();
  
  // Prior alphas
  to_vector(alpha) ~ normal(0, 2.5);

  // Prior thetas
  to_vector(theta) ~ std_normal();
  
}

generated quantities{
  
  // QR Decomposition: Betas
  matrix[K*(K-1), C] beta; 
  
  // Specify variables: Viterbi
  int<lower=1,upper=K> z_star[N];
  
  //////////////
  ///// QR /////
  //////////////
  
  // Transform thetas from QR decomposition to betas
  for (i in 1:(K*(K-1))) {
    beta[i] = to_row_vector(R_ast_inverse * to_vector(theta[i]));
  }
  
  /////////////
  // Viterbi //
  /////////////
  
  {
    int back_ptr[N, K];     // Backpointer to keep track of states
    real best_logp[N, K];   // Best log_p for each time step and each state
    real log_p_z_star;      // Max log p at end of sequence of each individual
    real logp;              // Variable for storing viterbi prob.
    
    // Looping over whole sequence
    for (t in 1:N) {
      
    // First observation of each individual
    if (t==1 || ID[t] != ID[t-1]){
      for (k in 1:K){
         best_logp[t, k] = log(pi_initial[k]) +
                           normal_lpdf(y1[t] | mu1[map_proc1[k]], sigma1) + 
                           normal_lpdf(y2[t] | mu2[map_proc2[k]], sigma2);
      }
      
    } else {
    
    // Following observations
    for (k in 1:K) {
      best_logp[t, k] = negative_infinity();

        for (j in 1:K) {
          
          // Previous viterbi path + transition prob. + emission prob.
          logp = best_logp[t-1, j] + log_gamma[t, j, k] + 
                 normal_lpdf(y1[t] | mu1[map_proc1[k]], sigma1) + 
                 normal_lpdf(y2[t] | mu2[map_proc2[k]], sigma2);
                 
          if (logp > best_logp[t, k]) { 
            back_ptr[t, k] = j; 
            best_logp[t, k] = logp; 
          }
         }
        }
      }
    }

    // Determining best sequence starting at end of sequence
    for (t0 in 1:N){
      int t = N - t0 + 1;
      
      // End of each individual sequence
      if (t==N || ID[t+1] != ID[t]) {
        log_p_z_star = max(best_logp[t]);
      
      for (k in 1:K)
          if (best_logp[t, k]==log_p_z_star)
            z_star[t] = k;
            
          } else {
            z_star[t] = back_ptr[t+1, z_star[t+1]];
       }
    }
  }
}

