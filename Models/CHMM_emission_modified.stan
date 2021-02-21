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
  
  simplex[K] pi_initial;        // Initial state probabilities
  simplex[K] gamma[K];          // Transition probability matrix
  positive_ordered[k1] alpha1;  // Intercept for mu of process 1
  positive_ordered[k2] alpha2;  // Intercept for mu of process 2
  matrix[k1, C] theta1;         // Thetas for mu of process 1
  matrix[k2, C] theta2;         // Thetas for mu of process 2
  real<lower=0> sigma1;         // Sigma of process 1
  real<lower=0> sigma2;         // Sigma of process 2
  
}

transformed parameters{
  
  // Mus of both processes for each t
  positive_ordered[k1] mu1[N];
  positive_ordered[k2] mu2[N];
  
  // Transition probability matrix
  matrix[K,K] log_gamma;       // log
  matrix[K,K] log_gamma_tr;    // transpose
  
  // Log_lik
  vector[nind] log_lik;
  
  /////////////
  //// Mus ////
  /////////////
  
  for (t in 1:N){
    
    // Process 1
    for (i in 1:k1){
      mu1[t,i] = alpha1[i] + Q_ast[t]*to_vector(theta1[i]);
    }
    
    // Process 2
    for (i in 1:k2){
      mu2[t,i] = alpha2[i] + Q_ast[t]*to_vector(theta2[i]);
    }
    
  }
  
  /////////////
  //// tpm ////
  /////////////
  
  // log of gamma (transition prob. matrix)
  for (i in 1:K){
    log_gamma[i] = to_row_vector(log(gamma[i])); 
  }
  
  // Transpose log gamma (transition prob. matrix)
  log_gamma_tr = log_gamma'; 
  
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
                  normal_lpdf(y1[t] | mu1[t, map_proc1[k]], sigma1) + 
                  normal_lpdf(y2[t] | mu2[t, map_proc2[k]], sigma2); 
    }
    
    // Following observations
    for (k in 1:K)  // Previous forward path prob. + transition prob. + emission prob.
        lp_p1[k] = log_sum_exp(to_vector(log_gamma_tr[k]) + lp) +
                               normal_lpdf(y1[t] | mu1[t, map_proc1[k]], sigma1) + 
                               normal_lpdf(y2[t] | mu2[t, map_proc2[k]], sigma2);
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
  
  // Prior transition probability matrix
  for (i in 1:K){
    gamma[i] ~ dirichlet(rep_vector(1, K)); 
  }
  
  // Prior alphas
  alpha1 ~ normal(0, 10);
  alpha2 ~ normal(0, 10);
  
  // Prior thetas
  to_vector(theta1) ~ std_normal();
  to_vector(theta2) ~ std_normal();
  
  // Prior sigmas
  sigma1 ~ std_normal();
  sigma2 ~ std_normal();
  
}

generated quantities{
  
  // QR Decomposition: Betas
  matrix[k1, C] beta1;
  matrix[k2, C] beta2;
  
  // Specify variables: Viterbi
  int<lower=1,upper=K> z_star[N];
  
  //////////////
  ///// QR /////
  //////////////
  
  // Transform thetas from QR decomposition to betas
  for (i in 1:k1) {
    beta1[i] = to_row_vector(R_ast_inverse * to_vector(theta1[i]));
  }
  
  for (i in 1:k2) {
    beta2[i] = to_row_vector(R_ast_inverse * to_vector(theta2[i]));
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
                           normal_lpdf(y1[t] | mu1[t, map_proc1[k]], sigma1) + 
                           normal_lpdf(y2[t] | mu2[t, map_proc2[k]], sigma2);
      }
      
    } else {
    
    // Following observations
    for (k in 1:K) {
      best_logp[t, k] = negative_infinity();

        for (j in 1:K) {
          
          // Previous viterbi path + transition prob. + emission prob.
          logp = best_logp[t-1, j] + log_gamma[j, k] + 
                 normal_lpdf(y1[t] | mu1[t, map_proc1[k]], sigma1) + 
                 normal_lpdf(y2[t] | mu2[t, map_proc2[k]], sigma2);
                 
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

