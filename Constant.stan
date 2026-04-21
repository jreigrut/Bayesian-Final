data {
  int<lower=1> N;
  vector<lower=0>[N] C;
  vector<lower=0>[N] I;
}

parameters {
  real<lower=0> K;
  real<lower=0> r;
  real log_q;                            
  real<lower=0> sigma_proc;
  real<lower=0> sigma_obs;
  vector<lower=0.001, upper=2.0>[N] P;
}

transformed parameters {
  real q = exp(log_q);                     
  vector[N] Pmed;
  vector[N] Imed;

  Pmed[1] = log(P[1]);
  for (t in 2:N) {
    real pred;
    pred = P[t-1] + r * P[t-1] * (1 - P[t-1]) - C[t-1] / K;
    pred = fmax(pred, 0.001);
    Pmed[t] = log(pred);
  }
  for (t in 1:N)
    Imed[t] = log(q * K * P[t]);
}

model {
  // Priors
  K ~ lognormal(15, 0.5);
  r ~ lognormal(-1.07, 0.2);
  log_q ~ normal(-6.91, 0.5);              
                                            
  sigma_proc ~ normal(0, 1);             
  sigma_obs  ~ normal(0, 1);

  // State process
  P[1] ~ lognormal(-0.02, 0.198);
  for (t in 2:N)
    P[t] ~ lognormal(Pmed[t], sigma_proc);

  // Observation model
  for (t in 1:N)
    I[t] ~ lognormal(Imed[t], sigma_obs);
}

generated quantities {
  real MSY  = r * K / 4;
  real HMSY = r / 2;
  vector[N] B;
  vector[N] Irep;
  vector[N] log_lik;
  vector[N] exploitation;
  for (t in 1:N) {
    B[t]           = K * P[t];
    Irep[t]        = lognormal_rng(Imed[t], sigma_obs);
    log_lik[t]     = lognormal_lpdf(I[t] | Imed[t], sigma_obs);
    exploitation[t] = C[t] / (K * P[t]);
  }
}