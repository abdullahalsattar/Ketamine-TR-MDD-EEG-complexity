# ==============================================================================
# Linear mixed-effects analysis of EEG neural complexity
# Ketamine (0.5 / 1.0 mg/kg) vs Fentanyl (active control) in TR-MDD
# Measures: HFD, LZC, MSE (fine / intermediate / coarse)
# Timepoints: Baseline, 2 h, 24 h post-dose
#
# Design:      Within-subject, three-way crossover
# Estimation:  REML
# Inference:   Type III ANOVA, Satterthwaite degrees of freedom
# References:  Dose = Fentanyl; Time = Baseline
# ==============================================================================

# ---- Packages ----------------------------------------------------------------
library(lme4)         # mixed-effects models
library(lmerTest)     # Satterthwaite df, Type III ANOVA
library(emmeans)      # estimated marginal means, pairwise contrasts
library(performance)  # marginal / conditional R² and ICC
library(effectsize)   # standardized coefficients
library(dplyr)        # data manipulation

# ---- Data --------------------------------------------------------------------
df <- read.csv("data/mdd_complexity_data.csv", check.names = FALSE)

names(df) <- c("Participant", "Dose", "Time",
               "HFD", "LZC", "MSE_fine", "MSE_mid", "MSE_coarse")

dvs <- c("HFD", "LZC", "MSE_fine", "MSE_mid", "MSE_coarse")

# Convert to factors with explicit reference levels (Fentanyl; Baseline)
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

# ---- Helper functions --------------------------------------------------------
fit_lmm <- function(dv, rhs, data)
  lmer(as.formula(paste(dv, "~", rhs)), data = data, REML = TRUE)

type3 <- function(model)
  anova(model, type = "III", ddf = "Satterthwaite")

# ---- 1. Primary omnibus model: Dose × Time -----------------------------------
# Formula: DV ~ Dose * Time + (1|Participant)
# This is the primary, pre-specified analysis testing the a priori hypotheses.

primary <- lapply(setNames(dvs, dvs), function(dv) {
  
  m <- fit_lmm(dv, "Dose * Time + (1|Participant)", df)

  
  list(
    model        = m,
    anova        = type3(m),
    dose_main    = pairs(emmeans(m, ~ Dose), adjust = "bonferroni"),
    time_main    = pairs(emmeans(m, ~ Time), adjust = "bonferroni"),
    dose_by_time = pairs(emmeans(m, ~ Dose | Time), adjust = "bonferroni"),
    time_by_dose = pairs(emmeans(m, ~ Time | Dose), adjust = "bonferroni"),
    r2           = performance::r2(m),
    icc          = performance::icc(m),
    std_beta     = standardize_parameters(m, method = "refit")
  )
})

# ---- 2.  Polynomial contrasts -----------------------------------------
# Orthogonal polynomial components of Time (equally spaced: 0, 1, 2) and
# Dose (actual ketamine values: 0, 0.5, 1.0) allow direct tests of the
# pre-specified inverted-U (quadratic) temporal hypothesis

time_poly <- poly(0:2, 2)             # Baseline, 2 h, 24 h (evenly spaced)
dose_poly <- poly(c(0, 0.5, 1), 2)    # Fentanyl, 0.5 mg/kg, 1.0 mg/kg

df_poly <- df %>%
  mutate(
    Time_lin  = time_poly[as.integer(Time), 1],
    Time_quad = time_poly[as.integer(Time), 2],
    Dose_lin  = dose_poly[as.integer(Dose), 1],
    Dose_quad = dose_poly[as.integer(Dose), 2]
  )

polynomial <- lapply(setNames(dvs, dvs), function(dv) {
  m <- fit_lmm(dv,
               "(Dose_lin + Dose_quad) * (Time_lin + Time_quad) + (1|Participant)",
               df_poly)
  type3(m)
})

# ---- 3. Time-stratified dose analyses --------------------------------
# Dose effects are tested separately at 2 h (acute) and 24 h (recovery).


stratified <- lapply(setNames(c("2h_Post", "24h_Post"), c("2h", "24h")),
                     function(timepoint) {
                       
                       d <- droplevels(filter(df, Time == timepoint))
                       
                       lapply(setNames(dvs, dvs), function(dv) {
                         
                         m <- fit_lmm(dv, "Dose + (1|Participant)", d)
                         
                         list(
                           model    = m,
                           anova    = type3(m),
                           dose     = pairs(emmeans(m, ~ Dose), adjust = "bonferroni"),
                           r2       = performance::r2(m),
                           std_beta = standardize_parameters(m, method = "posthoc")
                         )
                       })
                     })

# ---- 4. Export results -------------------------------------------------------
dir.create("results", showWarnings = FALSE)

stack_anova <- function(lst) {
  bind_rows(lapply(names(lst), function(dv)
    data.frame(DV = dv, Effect = rownames(lst[[dv]]), lst[[dv]], check.names = FALSE)))
}

write.csv(stack_anova(lapply(primary, `[[`, "anova")),
          "results/primary_anova.csv", row.names = FALSE)

write.csv(stack_anova(polynomial),
          "results/polynomial_anova.csv", row.names = FALSE)

for (tp_name in names(stratified))
  write.csv(stack_anova(lapply(stratified[[tp_name]], `[[`, "anova")),
            paste0("results/dose_anova_", tp_name, ".csv"), row.names = FALSE)