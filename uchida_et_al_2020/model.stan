functions{
  real ZINB_lpmf(int Y, real q, real lambda, real phi){
    if(Y==0){
      return log_sum_exp(
        bernoulli_lpmf(0|q),
        bernoulli_lpmf(1|q) + neg_binomial_2_log_lpmf(0|lambda,phi)
      );
    }else{
      return bernoulli_lpmf(1|q) + neg_binomial_2_log_lpmf(Y|lambda,phi);
    }
  }
}
data{
  int<lower=1> Site;
  int<lower=1> Time;
  int<lower=0> ACg[Site, Time];
  int<lower=0> ACt[Site, Time];
  int<lower=0> MAg[Site, Time];
  int<lower=0> MAt[Site, Time];
  int<lower=0> PM[Site, 5, Time];
  matrix<lower=0, upper=1>[Site, Time-1] EXT;
  int<lower=1> K[Time];
  matrix<lower=0>[Site, Site] W;
  vector<lower=0>[Site] rowSumW;
  int<lower=1> N;
  matrix [Site, N-3] ENV;
  int<lower=0, upper=1> Veg[Site, 3];
}
transformed data{
  matrix[Site,N] ENV1 = append_col(ENV, to_matrix(Veg));
}
parameters{
  vector<lower=0, upper=10>[11] sigma;
  vector[2] season_pm;
  vector[2] season_ac;
  vector[2] season_ma;
  vector<lower=0, upper=1>[3] k;
  real pre[2];
  real ext[3];
  vector<lower=0>[5] phi;
  real<lower=0, upper=1> theta;
  vector[N] alpha_ac;
  vector[N] alpha_ma;
  vector[N] beta_ac;
  vector[N] beta_pm;
  vector[N] beta_ma;
  matrix[Site,5] ab_raw;
  matrix[Site, Time-1] s_raw[3];
  vector[Site] mu1[3];
  vector[2] gamma;
  vector[N-3] delta[3];
}
transformed parameters{
  vector[3] Season_ac;
  vector[3] Season_pm;
  vector[3] Season_ma;
  vector[Site] tree_ac = ENV1*alpha_ac + sigma[4]*ab_raw[,1];
  vector[Site] tree_ma = ENV1*alpha_ma + sigma[5]*ab_raw[,2];
  vector[Site] r_pm = ENV1* beta_pm + sigma[6]*ab_raw[,3];
  vector[Site] r_ac = ENV1* beta_ac + sigma[7]*ab_raw[,4];
  vector[Site] r_ma = ENV1* beta_ma + sigma[8]*ab_raw[,5];
  matrix[Site, Time] mu_pm;
  matrix[Site, Time] mu_ac;
  matrix[Site, Time] mu_ma;
  Season_pm[1:2] = season_pm[1:2];
  Season_pm[3] = - sum(season_pm);
  Season_ac[1:2] = season_ac[1:2];
  Season_ac[3] = - sum(season_ac);
  Season_ma[1:2] = season_ma[1:2];
  Season_ma[3] = - sum(season_ma);
  mu_ac[,1] = mu1[1,];
  mu_pm[,1] = mu1[2,];
  mu_ma[,1] = mu1[3,];
  for(t in 2:Time){
    mu_pm[,t] = r_pm + k[1]*mu_pm[,t-1]                           + ext[1]*EXT[,t-1] + Season_pm[K[t-1]] + sigma[ 9]*s_raw[1,,t-1];
    mu_ac[,t] = r_ac + k[2]*mu_ac[,t-1] + pre[1]*exp(mu_pm[,t-1]) + ext[2]*EXT[,t-1] + Season_ac[K[t-1]] + sigma[10]*s_raw[2,,t-1];
    mu_ma[,t] = r_ma + k[3]*mu_ma[,t-1] + pre[2]*exp(mu_pm[,t-1]) + ext[3]*EXT[,t-1] + Season_ma[K[t-1]] + sigma[11]*s_raw[3,,t-1];
  }
}
model{
  phi ~ cauchy(0, 5);
  pre ~ normal(0, 100);
  ext ~ normal(0, 100);
  beta_pm ~ normal(0,100);
  beta_ac ~ normal(0,100);
  beta_ma ~ normal(0,100);
  alpha_ac ~ normal(0,100);
  alpha_ma ~ normal(0,100);
  season_ma ~ normal(0, 100);
  season_pm ~ normal(0, 100);
  season_ac ~ normal(0, 100);
  gamma ~ normal(0, 100);
  to_vector(ab_raw) ~ normal(0,1);
  Veg[,1] ~ bernoulli_logit(ENV*delta[1,]);
  Veg[,2] ~ bernoulli_logit(ENV*delta[2,]);
  Veg[,3] ~ bernoulli_logit(ENV*delta[3,]);
  for(i in 1:3){
    delta[i,] ~ normal(0,100);
    to_vector(s_raw[i,,]) ~ normal(0,1);
    mu1[i,] ~ normal(W*mu1[i,] ./ rowSumW, sigma[i] ./ sqrt(rowSumW));
  }
  for(t in 1:Time){
    if(ACg[1,t]<999){
      for(s in 1:Site){
        if(ACg[s,t]<999){
          ACg[s,t] ~ neg_binomial_2_log(mu_ac[s,t], phi[1]);
          ACt[s,t] ~ neg_binomial_2_log(mu_ac[s,t] + tree_ac[s], phi[2]);
          MAg[s,t] ~ neg_binomial_2_log(mu_ma[s,t], phi[3]);
          MAt[s,t] ~ ZINB(theta, mu_ma[s,t] + tree_ma[s], phi[4]);
        }
        if(PM[s,1,t]<999)
          for(i in 1:5)
            PM[s,i,t] ~ ZINB(inv_logit(gamma[1] + gamma[2]*mu_pm[s,t]), mu_pm[s,t], phi[5]);
      }
    }
  }
}
