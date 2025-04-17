functions {
  #include util.stanfunctions
  #include cormack-jolly-seber.stanfunctions
  #include jolly-seber.stanfunctions
}

data {
  int<lower=1> I_max,  // maximum number of individuals per site
               K_max,  // maximum numberof secondaries
               J,  // number of primaries
               M;  // number of sites
  array[M] int<lower=1> I;  // number of individuals
  array[M, J] int<lower=0, upper=K_max> K;  // number of secondaries
  array[M, I_max] int<lower=1, upper=J> f;  // first primary of detection
  array[M, I_max] int<lower=f, upper=J> l;  // last primary of detection
  matrix<lower=0>[J - 1, M] tau;  // survey intervals
  array[M, I_max, J, K_max] int<lower=1, upper=2> y;  // detection history
  array[M] int<lower=0> I_aug;  // number of augmented individuals
  int<lower=0, upper=1> intervals;  // account for tau (0 = no, 1 = yes)
}

transformed data {
  int JS = sum(I_aug) > 0,
      I_sum = sum(I),
      I_aug_max = max(I_aug),
      K_sum = 0;
  matrix[J - 1, M] log_tau_star;  // log scaled time intervals
  array[M, I_max] int f_k;  // first secondary detected
  for (m in 1:M) {
    K_sum += sum(K[m]);
    if (JS) {
      log_tau_star[:, m] = log(tau[:, m] / sum(tau[:, m]) * (J - 1));
    } else {
      f_k[m, 1:I[m]] = first_sec(y[m, 1:I[m]], K[m], f[m, 1:I[m]]);
    }
  }
}

parameters {
  vector<lower=0>[M] h;  // mortality hazard rates
  vector[M] l_p_a;  // logit detection intercepts
  vector<lower=0>[2] l_p_t;  // detection log odds scales
  matrix<multiplier=l_p_t[1]>[M, J] l_p_j;  // primary effects
  vector<multiplier=l_p_t[2]>[K_sum] l_p_k;  // secondary effects
  vector<lower=0, upper=1>[JS * M] psi;  // inclusion probabilities
  matrix<lower=0>[JS * J, M] u;  // (unnormalised) entry rates
  row_vector<lower=0>[JS * intervals * M] mu;  // baseline entry rates
}

transformed parameters {
  // (log) concentration vectors and entry probabilities
  matrix[JS * J, M] log_alpha, beta;
  if (JS) {
    log_alpha = rep_matrix(0, J, M);
    if (intervals) {
      log_alpha[2:J] = rep_matrix(log(mu), J - 1) + log_tau_star;
    }
    for (m in 1:M) {
      beta[:, m] = u[:, m] / sum(u[:, m]);
    }
  }
  
  // survival and detection probabilities
  matrix[J - 1, M] phi_tau = exp(diag_post_multiply(tau, -h));
  array[M] matrix[K_max, J] p;
  {
    int kk = 1;
    for (m in 1:M) {
      p[m] = l_p_a[m] + rep_matrix(l_p_j[m], K_max);
      for (j in 1:J) {
        for (k in 1:K[m, j]) {
          p[m][k, j] += l_p_k[kk];
          kk += 1;
        }
      }
    }
    p = inv_logit(p);
  }
}

model {
  // priors
  target += exponential_lupdf(h | 1);
  target += logistic_lupdf(l_p_a | 0, 1);
  target += exponential_lupdf(l_p_t | 2);
  target += normal_lupdf(to_vector(l_p_j) | 0, l_p_t[1]);
  target += normal_lupdf(l_p_k | 0, l_p_t[2]);
  if (JS) {
    target += beta_lupdf(psi | 1, 1);
    target += gamma_lupdf(to_vector(u) | to_vector(exp(log_alpha)), 1);
    if (intervals) {
      target += gamma_lupdf(mu | 1, 1);
    }
  }
  
  // likelihood
  for (m in 1:M) {
    if (JS) {
      // tuple(vector[I[m]], matrix[2, I_aug[m]], matrix[J, I[m]], matrix[J, I_aug[m]],
      //       array[I[m]] matrix[2, J], array[I_aug[m], J] matrix[2, J])
      //   lp = js_rd(y[m, 1:I[m]], f[m, 1:I[m]], l[m, 1:I[m]], K[m], psi[m],
      //              beta[:, m], rep_matrix(phi_tau[:, m], I[m] + I_aug[m]), rep_array(p[m], I[m] + I_aug[m]));
      tuple(vector[I[m]], vector[2], array[I[m]] matrix[2, J], 
            array[J] matrix[2, J], matrix[J, I[m]], vector[J])
        lp = js_rd(y[m, 1:I[m]], f[m, 1:I[m]], l[m, 1:I[m]], K[m], psi[m],
                   beta[:, m], phi_tau[:, m], p[m]);
      target += sum(lp.1) + I_aug[m] * log_sum_exp(lp.2);
    } else {
      vector[I[m]] log_lik = cjs_rd(y[m, 1:I[m]], f[m, 1:I[m]], l[m, 1:I[m]], 
                                    K[m], f_k[m, 1:I[m]], phi_tau[:, m], p[m]);
      target += sum(log_lik);
    }
  }
}

generated quantities {
  vector[I_sum] log_lik;
  array[JS * M] tuple(array[J] int, array[J] int, int) latent;
  for (m in 1:M) {
    int mm = sum(I[1:m]) - I[m];
    array[I[m]] int idx = linspaced_int_array(I[m], mm + 1, mm + I[m]);
    if (JS) {
      tuple(vector[I[m]], vector[2], array[I[m]] matrix[2, J], 
            array[J] matrix[2, J], matrix[J, I[m]], vector[J])
        lp = js_rd(y[m, 1:I[m]], f[m, 1:I[m]], l[m, 1:I[m]], K[m], psi[m],
                   beta[:, m], phi_tau[:, m], p[m]);
      log_lik[idx] = lp.1;
      latent[m] = js_rd_rng(lp, f[m, 1:I[m]],  l[m, 1:I[m]], I_aug[m],
                            phi_tau[:, m]);
    } else {
      log_lik[idx] = cjs_rd(y[m, 1:I[m]], f[m, 1:I[m]], l[m, 1:I[m]], K[m], 
                            f_k[m, 1:I[m]], phi_tau[:, m], p[m]);
    }
  }
}
