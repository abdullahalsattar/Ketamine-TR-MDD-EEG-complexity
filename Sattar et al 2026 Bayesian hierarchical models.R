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
# ==============================================================================

# ---- Packages ----------------------------------------------------------------
library(brms)        # Bayesian mixed models via Stan
library(posterior)   # posterior draws
library(bayestestR)  # Savage-Dickey Bayes factors
library(mvtnorm)     # multivariate normal densities (joint Bayes factors)
library(dplyr)

set.seed(12345)

# ---- Data --------------------------------------------------------------------
df <- read.csv("data/mdd_complexity_data.csv", check.names = FALSE)

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
# Dose contrasts at each timepoint (dose coefficient + Dose x Time interaction),
# and time contrasts within the Fentanyl condition.

contrast_draws <- function(m) {
  M <- b_draws(m, c(dose_b, time_b, inter_b))
  b <- function(k) M[, paste0("b_", k)]
  
  dose_at <- function(t) {
    e05 <- b("Dose0.5mg_Ketamine") + if (t == "Baseline") 0 else b(paste0("Dose0.5mg_Ketamine:Time", t))
    e10 <- b("Dose1.0mg_Ketamine") + if (t == "Baseline") 0 else b(paste0("Dose1.0mg_Ketamine:Time", t))
    setNames(list(e05, e10, e10 - e05),
             paste(c("0.5mg vs Fentanyl", "1.0mg vs Fentanyl", "1.0mg vs 0.5mg"), "at", t))
  }
  
  c(dose_at("Baseline"), dose_at("2h_Post"), dose_at("24h_Post"),
    list("2h vs Baseline (Fentanyl)"  = b("Time2h_Post"),
         "24h vs Baseline (Fentanyl)" = b("Time24h_Post"),
         "24h vs 2h (Fentanyl)"       = b("Time24h_Post") - b("Time2h_Post")))
}

contrasts <- bind_rows(lapply(dvs, function(y) {
  s <- t(sapply(contrast_draws(models[[y]]), summ))
  data.frame(DV = y, Contrast = rownames(s), s, row.names = NULL)
}))

# ---- 6. Joint tests: ROPE and Savage-Dickey Bayes factors --------------------
# ROPE: all coefficients in a set within +/- 5% of the outcome SD.
# BF10 (joint): Gaussian approximation to the posterior at the null (all = 0).

joint_rope <- function(M, halfwidth) {
  p_null <- mean(apply(abs(M) <= halfwidth, 1, all))
  c(P_against_null = 1 - p_null, Evidence_ratio = (1 - p_null) / p_null)
}

bf10_joint <- function(M, prior_sd) {
  z <- rep(0, ncol(M))
  dmvnorm(z, sigma = diag(prior_sd^2, ncol(M))) /          # prior density at 0
    dmvnorm(z, mean = colMeans(M), sigma = cov(M))         # posterior density at 0
}

omnibus <- bind_rows(lapply(dvs, function(y) {
  s    <- sd(df[[y]], na.rm = TRUE)
  sets <- list("Dose (at Baseline)"   = list(dose_b,  2 * s),
               "Time (under Fentanyl)" = list(time_b,  2 * s),
               "Dose x Time"           = list(inter_b, s))
  bind_rows(lapply(names(sets), function(e) {
    M <- b_draws(models[[y]], sets[[e]][[1]])
    data.frame(DV = y, Effect = e,
               t(joint_rope(M, 0.05 * s)),
               BF10 = bf10_joint(M, sets[[e]][[2]]))
  }))
}))

# ---- 7. Per-coefficient Savage-Dickey Bayes factors --------------------------
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