"""
Permutation tests of EEG neural complexity
Ketamine (0.5 / 1.0 mg/kg) vs Fentanyl (active control) in TR-MDD
Measures:   HFD, LZC, MSE (fine / intermediate / coarse)
Timepoints: Baseline, 2 h, 24 h post-dose

Condition labels are permuted within each participant (repeated-measures
exchangeability). 25,000 permutations; p = (1 + #null >= observed) / (1 + N).
Multiple comparisons: Westfall-Young single-step maxT within each family of tests.
"""

import numpy as np
import pandas as pd
from pathlib import Path

N_PERM = 25000
rng = np.random.default_rng(42)

DOSES    = ["Fentanyl", "0.5mg_Ketamine", "1.0mg_Ketamine"]
TIMES    = ["Baseline", "2h_Post", "24h_Post"]
MEASURES = ["HFD", "LZC", "MSE_fine", "MSE_mid", "MSE_coarse"]
PAIRS    = [(0, 1), (0, 2), (1, 2)]          # contrast = level i minus level j

# ---- Data --------------------------------------------------------------------
df = pd.read_csv("data/mdd_complexity_data.csv")
df.columns = ["Participant", "Dose", "Time"] + MEASURES
df["Dose"] = df["Dose"].map({"Fent": DOSES[0], "05K": DOSES[1], "1K": DOSES[2]})
df["Time"] = df["Time"].map({"0hr": TIMES[0], "2hr": TIMES[1], "24hr": TIMES[2]})


def cube(measure):
    """Participants x Dose x Time array (participants with all 9 cells)."""
    wide = (df.pivot_table(index="Participant", columns=["Dose", "Time"], values=measure)
              .reindex(columns=pd.MultiIndex.from_product([DOSES, TIMES]))
              .dropna())
    return wide.to_numpy().reshape(-1, len(DOSES), len(TIMES))


# ---- Test statistics ---------------------------------------------------------
def rm_F(Y):
    """One-way repeated-measures ANOVA F. Y: participants x conditions."""
    n, k = Y.shape
    gm = Y.mean()
    ss_cond = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_subj = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_err  = ((Y - gm) ** 2).sum() - ss_cond - ss_subj
    return (ss_cond / (k - 1)) / (ss_err / ((n - 1) * (k - 1)))


def pair_diffs(Y):
    """Paired mean differences for all level pairs on axis 1 (repeated over any axis 2)."""
    M = Y.mean(0)
    return np.stack([M[i] - M[j] for i, j in PAIRS], -1).ravel()


# ---- Permutation engine and p-values -----------------------------------------
def permute(Y, stat):
    """Observed statistic and null distribution. Axis 1 (condition) is permuted
    within each participant; the same permutation is applied across axis 2, so
    all tests in a family share each permutation (needed for maxT)."""
    n, k = Y.shape[:2]
    shape = (n, k) + (1,) * (Y.ndim - 2)
    order = np.tile(np.arange(k), (n, 1))
    null = np.array([stat(np.take_along_axis(Y, rng.permuted(order, axis=1).reshape(shape), axis=1))
                     for _ in range(N_PERM)])
    return np.atleast_1d(stat(Y)), null.reshape(N_PERM, -1)


def p_perm(obs, null):                        # uncorrected, one test per column
    return (1 + (null >= obs).sum(0)) / (1 + len(null))


def p_maxT(obs, null):                        # Westfall-Young maxT across columns
    return (1 + (null.max(1)[:, None] >= obs).sum(0)) / (1 + len(null))


def interaction_freedman_lane(V):
    """Dose x Time omnibus F (subject + Dose + Time vs + Dose x Time), with
    Freedman-Lane permutation of reduced-model residuals within participant."""
    n = V.shape[0]
    d, t = np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)
    D = (d[:, None] == [1, 2]).astype(float)                  # treatment coding
    T = (t[:, None] == [1, 2]).astype(float)
    DxT = (D[:, :, None] * T[:, None, :]).reshape(9, 4)
    X_red  = np.hstack([np.kron(np.eye(n), np.ones((9, 1))), np.tile(np.hstack([D, T]), (n, 1))])
    X_full = np.hstack([X_red, np.tile(DxT, (n, 1))])
    R_red, R_full = (np.eye(9 * n) - X @ np.linalg.pinv(X) for X in (X_red, X_full))
    df_err = 9 * n - np.linalg.matrix_rank(X_full)

    def F(y):
        rss_red, rss_full = y @ R_red @ y, y @ R_full @ y
        return ((rss_red - rss_full) / 4) / (rss_full / df_err)

    y = V.reshape(-1)
    resid = (R_red @ y).reshape(n, 9)
    fitted = y - resid.ravel()
    null = np.array([F(fitted + rng.permuted(resid, axis=1).ravel()) for _ in range(N_PERM)])
    return np.atleast_1d(F(y)), null[:, None], df_err


# ---- Analyses ----------------------------------------------------------------
results = []


def record(measure, analysis, tests, obs, p, p_adj=None):
    for i, test in enumerate(tests):
        results.append({"Measure": measure, "Analysis": analysis, "Test": test,
                        "Statistic": obs[i], "p_perm": p[i],
                        "p_maxT": np.nan if p_adj is None else p_adj[i]})


dose_pairs = [f"{DOSES[i]} - {DOSES[j]}" for i, j in PAIRS]
time_pairs = [f"{TIMES[i]} - {TIMES[j]}" for i, j in PAIRS]

for m in MEASURES:
    V = cube(m)                                                   # participants x dose x time

    # 1. Main effects: RM-ANOVA F on participant means, collapsing the other factor
    for name, Y in [("Dose main effect", V.mean(2)), ("Time main effect", V.mean(1))]:
        obs, null = permute(Y, rm_F)
        record(m, name, ["F"], obs, p_perm(obs, null))

    # 2. Dose pairwise within each time (two-sided; maxT over 9 tests)
    obs, null = permute(V, pair_diffs)
    record(m, "Dose pairwise within time", [f"{p} at {t}" for t in TIMES for p in dose_pairs],
           obs, p_perm(abs(obs), abs(null)), p_maxT(abs(obs), abs(null)))

    # 3. Time pairwise within each dose (two-sided; maxT over 9 tests)
    obs, null = permute(V.transpose(0, 2, 1), pair_diffs)
    record(m, "Time pairwise within dose", [f"{p} under {d}" for d in DOSES for p in time_pairs],
           obs, p_perm(abs(obs), abs(null)), p_maxT(abs(obs), abs(null)))

    # 4. Time-stratified dose effects: F and pairwise (maxT over 3 tests) at 2 h and 24 h
    for k, label in [(1, "2h"), (2, "24h")]:
        Y = V[:, :, k]
        obs, null = permute(Y, rm_F)
        record(m, f"Dose effect at {label}", ["F"], obs, p_perm(obs, null))
        obs, null = permute(Y, pair_diffs)
        record(m, f"Dose pairwise at {label}", dose_pairs,
               obs, p_perm(abs(obs), abs(null)), p_maxT(abs(obs), abs(null)))

    # 5a. Dose x Time omnibus (Freedman-Lane)
    obs, null, df_err = interaction_freedman_lane(V)
    record(m, "Dose x Time omnibus (Freedman-Lane)", [f"F(4, {df_err})"], obs, p_perm(obs, null))

    # 5b. Dose x Time by interval: RM-ANOVA F across doses on change scores
    #     (dose labels permuted jointly across intervals; maxT over 3 intervals)
    C = V[:, :, [1, 2, 2]] - V[:, :, [0, 0, 1]]
    obs, null = permute(C, lambda Y: np.array([rm_F(Y[:, :, i]) for i in range(3)]))
    record(m, "Dose x Time by interval", ["2h - Baseline", "24h - Baseline", "24h - 2h"],
           obs, p_perm(obs, null), p_maxT(obs, null))

# ---- Export ------------------------------------------------------------------
Path("results").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv("results/permutation_results.csv", index=False)