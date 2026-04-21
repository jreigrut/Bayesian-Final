data {
  int<lower=1> N;               // number of time steps
  vector<lower=0>[N] C;         // catch
  vector<lower=0>[N] I;         // abundance index
}

parameters {
  real<lower=0> K;     // carrying capacity
  real<lower=0> r;     // growth rate

  // Time-varying catchability
  vector[N] log_q;              
  real<lower=0> sigma_q;        

  // Variances
  real<lower=0> sigma_proc;     
  real<lower=0> sigma_obs;      

  vector<lower=0.001, upper=2.0>[N] P;   // population process (latent)
}

transformed parameters {
  vector[N] q;
  vector[N] Pmed;
  vector[N] Imed;

  // Convert log_q → q
  for (t in 1:N)
    q[t] = exp(log_q[t]);

  Pmed[1] = log(P[1]);    // log scale mean for first latent abundance
  for (t in 2:N) {
    real pred;
    pred = P[t - 1] 
           + r * P[t - 1] * (1 - P[t - 1]) 
           - C[t - 1] / K;
    pred = fmax(pred, 0.001);   // Prevent log(0)
    Pmed[t] = log(pred);
  }
  // Observation model
  for (t in 1:N) {
    Imed[t] = log(q[t] * K * P[t]);
  }
}

model {
  // Priors
  K ~ lognormal(15, 0.5);  // conversion of priors from JABBA
  r ~ lognormal(-1.07, 0.2);  // conversion of priors from JABBA
  q ~ lognormal(-6.91, 1);
  
  sigma_proc ~ normal(0, 1);
  sigma_obs ~ normal(0, 1);
  sigma_q ~ normal(0, 0.2);   

  // State process
  P[1] ~ lognormal(-0.02, 0.198); // conversion of priors from JABBA
  for (t in 2:N)
    P[t] ~ lognormal(Pmed[t], sigma_proc);
  // Observation model
  for (t in 1:N)
    I[t] ~ lognormal(Imed[t], sigma_obs);
}

generated quantities {
  real MSY = r * K / 4;
  real HMSY = r / 2;
  vector[N] B;
  vector[N] Irep;
  vector[N] log_lik;
  vector[N] exploitation;
  for (t in 1:N) {
    B[t] = K * P[t];
    Irep[t] = lognormal_rng(Imed[t], sigma_obs);
    log_lik[t] = lognormal_lpdf(I[t] | Imed[t], sigma_obs);
    exploitation[t] = C[t] / (K * P[t]);
  }
}
