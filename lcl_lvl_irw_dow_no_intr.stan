// Bayesian Structural Time Series
// local level integrated random walk trend with slope
//
data {
    int<lower=1> T;                 // number of time periods
    int<lower=1> S;                 // length of season
    vector[T] Y;
}
parameters {
    vector[T] v_err;    // innovations of slope
    vector[T] w_err;    // innovations of season
    real<lower=0> s_v;  // variance of slope innovations
    real<lower=0> s_w;  // variance of seasonal innovations
    real<lower=0> s_Y;  // variance of observation error
}
transformed parameters {
    vector[T] u;        // local level
    vector[T] v;        // slope of trend
    vector[T] w;        // seasons
    vector[T] Y_hat;    // mean prediction for each time period
    
    // construct slope and level series
    
    // v[t] = v[t-1] + v_err[t] * v_err
    // with a little algebra this boils down
    // to a cumulative sum
    v = s_v * cumulative_sum(v_err);
    
    // u[t] = u[t-1] + v[t]
    // with a little more algebra this
    // also boils down to a cumulative sum
    u = cumulative_sum(v);
    
    // seasonal effects
    // note: seasonal effect have to sum to zero to be identified
    w[1:(S-1)] = s_w * w_err[1:(S-1)];
    
    for (t in S:T) {
        w[t] = -sum(w[t-S+1:t-1]) + s_w * w_err[t];
    }
    
    // put it all together
    // this is hear in case we want to add 
    // cycles or seasonality later
    Y_hat = u + w;
}
model {
    // priors on slope innovations
    v_err ~ normal(0,1);
    w_err ~ normal(0,1);
    
    // priors on variances
    s_v ~ normal(0,10);
    s_w ~ normal(0,10);
    s_Y ~ normal(0,10);
    
    // likelihood
    Y ~ normal(Y_hat,s_Y);
}
generated quantities {
    vector[T] Y_sim;
    vector[T] Y_sim_exp;
    vector[T] mu_exp;
    vector[T] slope_exp;
    vector[T] dow_exp;
    
    // simulate values from the posterior
    for (t in 1:T) {
        Y_sim[t] = normal_rng(Y_hat[t],s_Y);
    }
    
    // transform from log scale
    Y_sim_exp = exp(Y_sim);
    mu_exp = exp(u);
    dow_exp = exp(w);
    slope_exp = exp(v);
}
