// BSTS
// ---------------------------------------------
// local level 
// estimated with Kalman Smoother
// notation follows Durbin and Koopman 2012, 
// section 4.3, 4.4
// 
// *** simulate from prior by setting prior_only = 1 ***
//
data {
    // dimensions
    int<lower=1> n;                     // time periods
    int<lower=1> p;                     // time series
    int<lower=1> m;                     // unobserved states per time period
    int<lower=1> r;                     // length of variance on state evolution

    // system constants
    matrix[p,m] Z[n];                   // Y_t = Z_t * alpha_t + s_eta ** allows for regression **
    matrix[m,m] T;                      // alpha_(t+1) = T_t * alpha_t + R_t * eta_t
    matrix[m,r] R;                      //

    // data
    vector[p] Y[n];                     // observed time series

    // sample from prior?
    int<lower=0,upper=1> prior_only;    // if 1, sample from prior, if 0, performance inference
}

parameters {
    vector<lower=0>[p] s_eps;           // variance of observation error
    vector<lower=0>[r] s_q;             // variance components of state
    vector<lower=0>[m] p_one;           // variance of intial state values
    vector[m] a_one;                    // mean of initial state values
}

transformed parameters {
    matrix[p,p] H;                      // observation error variance
    matrix[r,r] Q;                      // state error variance
    matrix[m,m] P_one;                  // variance of initial state
    matrix[p,p] F[n];                   // variance of prediction error
    matrix[p,p] F_inv[n];               // inverse of F
    matrix[m,p] K[n];                   // Kalman gain
    matrix[m,p] L[n];                   // 1 minus Kalman gain
    matrix[m,m] P[n];                   // variance of mean of alpha_t given Y_(t-1) 

    vector[p] v[n];                     // one-step ahead forecase error of Y_t given Y_(t-1)
    vector[m] a[n];                     // mean of alpha_t given Y_(t-1)
    vector[p] mu_a[n];                  // mean of Y_t = Z * a_t

    vector[m] a_tt[n];                  // mean of alpha_t given Y_t
    matrix[m,m] P_tt[n];                // variance of alpha_t given Y_t
    vector[p] mu_a_tt[n];               // mean of Y_t = Z * alpha_t

    // Kalman smoother variables
    vector[m] a_hat[n];                 // smoothed mean of alpha_t given Y_1:t
    vector[p] mu_a_hat[n];              // mean of Y_t = Z * alpha_hat_t
    vector[m] r_t_zero;                 // value of r at t = 0
    vector[m] r_t[n];                   // weighted sum of innovations v_t occurring after t -1
    //vector[p] u[n];                     // smoothing error
    
    // construct matrices of variances
    H = diag_matrix(s_eps);
    Q = diag_matrix(s_q);
    P_one = diag_matrix(p_one);

    // initialize vectors
    a[1] = a_one;
    v[1] = Y[1] - Z[1] * a_one;
    mu_a[1] = Z[1] * a[1];

    // initialize matrices
    P[1] = P_one;
    F[1] = Z[1] * P[1] * Z[1]' + H;
    F_inv[1] = inverse(F[1]);
    K[1] = T * P[1] * Z[1]' * F_inv[1];
    L[1] = T - K[1] * Z[1];

    // initialize distribution of state    
    a_tt[1] = a_one + P[1] * Z[1]' * F_inv[1] * v[1];
    P_tt[1] = P[1] - P[1] * Z[1]' * F_inv[1] * Z[1] * P[1];
    mu_a_tt[1] = Z[1] * a_tt[1];

    { // Kalman filter
        
        for (t in 2:n) {
            // a and v
            a[t] = T * a[t-1] + K[t-1] * v[t-1];
            v[t] = Y[t] - Z[t] * a[t];
            mu_a[t] = Z[t] * a[t];
        
            // P, F, K
            P[t] = T * P[t-1] * (T - K[t-1] * Z[t])' + R * Q * R';
            F[t] = Z[t] * P[t] * Z[t]' + H;
            F_inv[t] = inverse(F[t]);
            K[t] = T * P[t] * Z[t]' * F_inv[t];
            L[t] = T - K[t] * Z[t];

            // construct state mean and variance
            a_tt[t] = a[t] + P[t] * Z[t]' * F_inv[t] * v[t];
            P_tt[t] = P[t] - P[t] * Z[t]' * F_inv[t] * Z[t] * P[t];
            mu_a_tt[t] = Z[t] * a_tt[t];
        }

    } // filter
       
    { // Kalman smoother recursions

        // counter to allow us to go from n down to 1
        int c = n - 1;

        // initialize r
        r_t[n] = rep_vector(0,m);
        
        // backwards smoother
        for (t in 1:(n-1)) {
            // eq 4.69 in DK2012, page 96
            r_t[c] = (Z[c+1]' * F_inv[c+1]) * v[c+1] + L[c+1]' * r_t[c+1];
            
            // go back in time
            c = c - 1;
        }

        // compute initial value for r
        r_t_zero = (Z[1]' * F_inv[1]) * v[1] + L[1]' * r_t[1];

        // initialize smoothed state
        a_hat[1] = a[1] + P[1] * r_t_zero;
        mu_a_hat[1] = Z[1] * a_hat[1];

        // forward for smoothed state
        for (t in 2:n) {
            a_hat[t] = a[t] + P[t] * r_t[t-1];
            mu_a_hat[t] = Z[t] * a_hat[t];
        }
    } // smoother
}

model {
    // observation and state variances
    s_eps ~ normal(0,5);
    s_q ~ normal(0,5);
    
    // initial state mean and variance
    a_one ~ normal(0,5);
    p_one ~ normal(0,5);
    
    // likelihood
    // Stan parameterizes normal in terms of standard deviation
    // so we're taking the square root
    //
    // conduct inference if prior_only = 0
    if (prior_only == 0)
        for (t in 1:n)
            Y[t] ~ multi_normal(mu_a[t],sqrt(H));
}

generated quantities {
    // simulated values from the filtered state
    vector[p] Y_sim[n];       
    vector[p] Y_sim_exp[n];

    // draw values from the filtered state
    for (t in 1:n) {
        Y_sim[t] = multi_normal_rng(mu_a_hat[t],sqrt(H));
    }

    // exponentiate
    Y_sim_exp = exp(Y_sim);
}