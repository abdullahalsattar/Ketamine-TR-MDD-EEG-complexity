# ==============================================================================
# Bayesian hierarchical analysis of EEG neural complexity
# Ketamine (0.5 / 1.0 mg/kg) vs Fentanyl (active control) in TR-MDD
# Measures: HFD, LZC, MSE (fine / intermediate / coarse)
# Timepoints: Baseline, 2 h, 24 h post-dose
#
# Model:       DV ~ Dose * Time + (1 | Participant), Gaussian (brms / Stan)
# Priors:      Weakly informative, scaled to each outcome's SD
# Sampling:    4 chains x 15,000 iterations (7,500 warm-up)
# References:  Dose = Fentanyl; Time = Baseline (treatment coding)
# Inference:   Dose and Time effects are tested on MARGINAL contrasts (averaged
#              over the other factor); Dose x Time on the interaction contrasts.
#              Post hoc contrasts are computed from posterior cell means.
# Change log (Nov 2026): omnibus Dose/Time tests changed from
#   reference-cell coefficients to marginal contrasts with implied priors;
#   added marginal, within-dose and change-vs-fentanyl contrasts.
# ==============================================================================

# ---- Packages ----------------------------------------------------------------
library(brms)        # Bayesian mixed models via Stan
library(posterior)   # posterior draws
library(bayestestR)  # Savage-Dickey Bayes factors
library(mvtnorm)     # multivariate normal densities (joint Bayes factors)
library(dplyr)

set.seed(12345)

# ---- Data --------------------------------------------------------------------
df <- read.csv(".............................", check.names = FALSE)

names(df) <- c("Participant", "Dose", "Time",
               "HFD", "LZC", "MSE_fine", "MSE_mid", "MSE_coarse")

dvs <- c("HFD", "LZC", "MSE_fine", "MSE_mid", "MSE_coarse")

df <- df %>%
  mutate(
    Participant = factor(Participant),
    Dose = factor(Dose,
                  levels = c("Fent", "05K", "1K"),
                  labels = c("Fentanyl", "0.5mg_Ketamine", "1.0mg_Ketamine")),
    Time = factor(Time,
                  levels = c("0hr", "2hr", "24hr"),
                  labels = c("Baseline", "2h_Post", "24h_Post"))
  )

# Model coefficients under treatment coding
dose_b  <- c("Dose0.5mg_Ketamine", "Dose1.0mg_Ketamine")   # dose effect at Baseline
time_b  <- c("Time2h_Post", "Time24h_Post")                 # time effect under Fentanyl
inter_b <- as.vector(outer(dose_b, time_b, paste, sep = ":"))

# ---- 1. Priors ---------------------------------------------------------------
# Intercept:     Normal(mean at Fentanyl-Baseline, SD)
# Main effects:  Normal(0, 2 x SD)
# Interactions:  Normal(0, SD)
# sigma, sd:     Student-t(3, 0, SD)

make_priors <- function(y) {
  s  <- sd(df[[y]], na.rm = TRUE)
  m0 <- mean(df[[y]][df$Dose == "Fentanyl" & df$Time == "Baseline"], na.rm = TRUE)
  c(
    set_prior(sprintf("normal(%g, %g)", m0, s), class = "Intercept"),
    do.call(c, lapply(c(dose_b, time_b), function(k)
      set_prior(sprintf("normal(0, %g)", 2 * s), class = "b", coef = k))),
    do.call(c, lapply(inter_b, function(k)
      set_prior(sprintf("normal(0, %g)", s), class = "b", coef = k))),
    set_prior(sprintf("student_t(3, 0, %g)", s), class = "sigma"),
    set_prior(sprintf("student_t(3, 0, %g)", s), class = "sd")
  )
}

# ---- 2. Fit models -----------------------------------------------------------
dir.create("models", showWarnings = FALSE)

fit_bhm <- function(y)
  brm(as.formula(paste(y, "~ Dose * Time + (1 | Participant)")),
      data = df, family = gaussian(), prior = make_priors(y),
      chains = 4, iter = 15000, warmup = 7500, cores = 4, seed = 12345,
      control = list(adapt_delta = 0.99, max_treedepth = 15),
      save_pars = save_pars(all = TRUE),
      file = file.path("models", paste0(y, "_model")))   # reloads if already fitted

models <- lapply(setNames(dvs, dvs), fit_bhm)

# ---- Helpers -----------------------------------------------------------------
# Posterior summary: mean, median, 95% credible interval, probability of direction (%)
summ <- function(x) {
  x <- as.numeric(x)
  c(Mean = mean(x), Median = median(x),
    CrI_2.5 = unname(quantile(x, 0.025)), CrI_97.5 = unname(quantile(x, 0.975)),
    pd = 100 * max(mean(x > 0), mean(x < 0)))
}

# Matrix of posterior draws for selected coefficients
b_draws <- function(m, coefs) {
  d <- as_draws_df(m)
  sapply(paste0("b_", coefs), function(k) as.numeric(d[[k]]))
}

# ---- 3. Convergence, model fit and ICC ---------------------------------------
model_fit <- bind_rows(lapply(dvs, function(y) {
  m   <- models[[y]]
  d   <- as_draws_df(m)
  r2  <- summ(bayes_R2(m, summary = FALSE))
  icc <- summ(d$sd_Participant__Intercept^2 /
                (d$sd_Participant__Intercept^2 + d$sigma^2))
  l   <- loo(m, moment_match = TRUE)
  data.frame(
    DV            = y,
    max_Rhat      = max(rhat(m), na.rm = TRUE),
    min_ESS_ratio = min(neff_ratio(m), na.rm = TRUE),
    R2            = r2[["Median"]],  R2_low  = r2[["CrI_2.5"]],  R2_high  = r2[["CrI_97.5"]],
    ICC           = icc[["Median"]], ICC_low = icc[["CrI_2.5"]], ICC_high = icc[["CrI_97.5"]],
    elpd_loo      = l$estimates["elpd_loo", "Estimate"],
    elpd_loo_se   = l$estimates["elpd_loo", "SE"],
    looic         = l$estimates["looic", "Estimate"],
    max_pareto_k  = max(l$diagnostics$pareto_k)
  )
}))

# ---- 4. Fixed effects --------------------------------------------------------
fixed <- bind_rows(lapply(dvs, function(y) {
  s <- t(apply(b_draws(models[[y]], c("Intercept", dose_b, time_b, inter_b)), 2, summ))
  data.frame(DV = y, Parameter = sub("^b_", "", rownames(s)), s, row.names = NULL)
}))

# ---- 5. Posterior contrasts --------------------------------------------------
# The 9 Dose x Time cell means are rebuilt from the coefficients for every draw,
# then contrasted (marginal, within-dose, change vs fentanyl, dose at each time).

cell_means <- function(m) {
  B  <- b_draws(m, c("Intercept", dose_b, time_b, inter_b))
  b  <- function(k) B[, paste0("b_", k)]
  dl <- levels(df$Dose); tl <- levels(df$Time)
  mu <- array(NA_real_, c(nrow(B), 3, 3), dimnames = list(NULL, dl, tl))
  for (d in 1:3) for (t in 1:3)
    mu[, d, t] <- b("Intercept") +
      (if (d > 1) b(paste0("Dose", dl[d])) else 0) +
      (if (t > 1) b(paste0("Time", tl[t])) else 0) +
      (if (d > 1 && t > 1) b(paste0("Dose", dl[d], ":Time", tl[t])) else 0)
  mu
}

contrast_draws <- function(m) {
  mu <- cell_means(m)
  Fe <- "Fentanyl"; K05 <- "0.5mg_Ketamine"; K10 <- "1.0mg_Ketamine"
  BL <- "Baseline"; H2 <- "2h_Post"; H24 <- "24h_Post"
  mD <- function(d) rowMeans(mu[, d, ])          # dose, averaged over time
  mT <- function(t) rowMeans(mu[, , t])          # time, averaged over dose
  ch <- function(d, t2, t1) mu[, d, t2] - mu[, d, t1]
  qd <- function(d) mu[, d, H2] - (mu[, d, BL] + mu[, d, H24]) / 2

  out <- list(
    "Marginal: 0.5mg vs Fentanyl" = mD(K05) - mD(Fe),
    "Marginal: 1.0mg vs Fentanyl" = mD(K10) - mD(Fe),
    "Marginal: 1.0mg vs 0.5mg"    = mD(K10) - mD(K05),
    "Marginal: 2h vs Baseline"    = mT(H2)  - mT(BL),
    "Marginal: 24h vs Baseline"   = mT(H24) - mT(BL),
    "Marginal: 24h vs 2h"         = mT(H24) - mT(H2),
    "Marginal: quadratic (2h vs mean of Baseline and 24h)" = mT(H2) - (mT(BL) + mT(H24)) / 2)

  for (d in c(Fe, K05, K10)) {
    out[[paste0("Within ", d, ": 2h vs Baseline")]]  <- ch(d, H2,  BL)
    out[[paste0("Within ", d, ": 24h vs Baseline")]] <- ch(d, H24, BL)
    out[[paste0("Within ", d, ": 24h vs 2h")]]       <- ch(d, H24, H2)
  }
  for (d in c(K05, K10)) {
    out[[paste0(d, " vs Fentanyl: change 2h - Baseline")]]  <- ch(d, H2,  BL) - ch(Fe, H2,  BL)
    out[[paste0(d, " vs Fentanyl: change 24h - Baseline")]] <- ch(d, H24, BL) - ch(Fe, H24, BL)
    out[[paste0(d, " vs Fentanyl: change 24h - 2h")]]       <- ch(d, H24, H2) - ch(Fe, H24, H2)
    out[[paste0(d, " vs Fentanyl: quadratic")]]             <- qd(d) - qd(Fe)
  }
  for (t in c(BL, H2, H24)) {
    out[[paste0("At ", t, ": 0.5mg vs Fentanyl")]] <- mu[, K05, t] - mu[, Fe, t]
    out[[paste0("At ", t, ": 1.0mg vs Fentanyl")]] <- mu[, K10, t] - mu[, Fe, t]
    out[[paste0("At ", t, ": 1.0mg vs 0.5mg")]]    <- mu[, K10, t] - mu[, K05, t]
  }
  out
}

contrasts <- bind_rows(lapply(dvs, function(y) {
  s <- t(sapply(contrast_draws(models[[y]]), summ))
  data.frame(DV = y, Contrast = rownames(s), s, row.names = NULL)
}))

# ---- 6. Joint tests: ROPE and Savage-Dickey Bayes factors --------------------
# Dose and Time are tested on MARGINAL contrasts (averaged over the other
# factor); Dose x Time on the four interaction contrasts. Each contrast set is
# a linear map L of the coefficients; its prior is the one implied by the
# coefficient priors: Cov = L diag(prior_sd^2) L'.
# ROPE: all contrasts in a set within +/- 5% of the outcome SD.
# BF10 = prior density at 0 / posterior density at 0 (Gaussian approximation).

joint_rope <- function(M, halfwidth) {
  p_null <- mean(apply(abs(M) <= halfwidth, 1, all))
  c(P_against_null = 1 - p_null, Evidence_ratio = (1 - p_null) / p_null)
}

bf10_joint <- function(C, prior_cov) {
  z <- rep(0, ncol(C))
  dmvnorm(z, sigma = prior_cov) /                          # prior density at 0
    dmvnorm(z, mean = colMeans(C), sigma = cov(C))         # posterior density at 0
}

# Coefficient order: dose_b (2), time_b (2), inter_b (05:2h, 10:2h, 05:24h, 10:24h)
coefs <- c(dose_b, time_b, inter_b)
L_sets <- list(
  "Dose (marginal)" = rbind(c(1, 0, 0, 0, 1/3, 0,   1/3, 0),
                            c(0, 1, 0, 0, 0,   1/3, 0,   1/3)),
  "Time (marginal)" = rbind(c(0, 0, 1, 0, 1/3, 1/3, 0,   0),
                            c(0, 0, 0, 1, 0,   0,   1/3, 1/3)),
  "Dose x Time"     = cbind(matrix(0, 4, 4), diag(4)))

omnibus <- bind_rows(lapply(dvs, function(y) {
  s   <- sd(df[[y]], na.rm = TRUE)
  psd <- c(rep(2 * s, 4), rep(s, 4))                       # main: 2*SD; interactions: SD
  B   <- b_draws(models[[y]], coefs)
  bind_rows(lapply(names(L_sets), function(e) {
    L <- L_sets[[e]]
    C <- B %*% t(L)                                        # posterior draws of contrasts
    data.frame(DV = y, Effect = e,
               t(joint_rope(C, 0.05 * s)),
               BF10 = bf10_joint(C, L %*% diag(psd^2) %*% t(L)))
  }))
}))

# ---- 7. Per-coefficient Savage-Dickey Bayes factors --------------------------
# Note: these are reference-cell coefficients (dose effect at Baseline; time
# effect under Fentanyl), as listed in the fixed-effects table.
       
bf10_coef <- function(x, prior_sd)
  exp(bayesfactor_parameters(x, prior = distribution_normal(1e5, 0, prior_sd),
                             direction = "two-sided", null = 0)$log_BF)

bf_coef <- bind_rows(lapply(dvs, function(y) {
  s <- sd(df[[y]], na.rm = TRUE)
  data.frame(DV = y, Parameter = c(dose_b, time_b),
             BF10 = apply(b_draws(models[[y]], c(dose_b, time_b)), 2, bf10_coef,
                          prior_sd = 2 * s),
             row.names = NULL)
}))

# ---- 8. Export results -------------------------------------------------------
dir.create("results", showWarnings = FALSE)

write.csv(model_fit, "results/bhm_model_fit.csv",          row.names = FALSE)
write.csv(fixed,     "results/bhm_fixed_effects.csv",      row.names = FALSE)
write.csv(contrasts, "results/bhm_contrasts.csv",          row.names = FALSE)
write.csv(omnibus,   "results/bhm_joint_tests.csv",        row.names = FALSE)
write.csv(bf_coef,   "results/bhm_bayes_factors_coef.csv", row.names = FALSE)
