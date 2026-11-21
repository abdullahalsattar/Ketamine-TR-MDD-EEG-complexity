"""
Permutation tests of EEG neural complexity
Ketamine (0.5 / 1.0 mg/kg) vs Fentanyl (active control) in TR-MDD
Measures:   HFD, LZC, MSE (fine / intermediate / coarse)
Timepoints: Baseline, 2 h, 24 h post-dose

Condition labels are permuted within each participant (repeated-measures
exchangeability). 25,000 permutations; p = (1 + #null >= observed) / (1 + N).
Permutation p-values vary slightly between runs.
Multiple comparisons: Westfall-Young single-step maxT within each family of tests (within each measure).
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


# ---- 6. Interval tests: maxT across all measures, and pairwise contrasts ------
#  6a. Same interval F tests as 5b, but one dose permutation per participant is
#      shared by all 5 measures x 3 intervals, so maxT covers all 15 tests.
#  6b. Pairwise interval contrasts: per-participant difference in change between
#      two treatments, exact sign-flip test (all 2^n sign patterns); maxT over
#      the 3 treatment pairs, and over 5 measures x 3 pairs for 24h - 2h.
import itertools

INTERVALS = ["2h - Baseline", "24h - Baseline", "24h - 2h"]
V_all = np.stack([cube(m) for m in MEASURES], axis=2)                # n x dose x measure x time
C_all = V_all[..., [1, 2, 2]] - V_all[..., [0, 0, 1]]                # n x dose x measure x interval

obs, null = permute(C_all, lambda Y: np.array([rm_F(Y[:, :, mi, i])
                                               for mi in range(len(MEASURES)) for i in range(3)]))
p_all = p_maxT(obs, null)
for mi, m in enumerate(MEASURES):
    s = slice(3 * mi, 3 * mi + 3)
    record(m, "Dose x Time by interval, maxT across measures", INTERVALS,
           obs[s], p_perm(obs[s], null[:, s]), p_all[s])

COMPARE = [(2, 0), (1, 0), (2, 1)]                                   # 1.0 vs Fent, 0.5 vs Fent, 1.0 vs 0.5
comp_names = [f"{DOSES[i]} - {DOSES[j]}" for i, j in COMPARE]
D = np.stack([C_all[:, i] - C_all[:, j] for i, j in COMPARE], axis=-1)   # n x measure x interval x pair
n = D.shape[0]
signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))     # all 2^n patterns
obs_pw = np.abs(D.mean(0))                                           # measure x interval x pair
null_pw = np.abs(np.tensordot(signs, D, axes=(1, 0)) / n)            # 2^n x measure x interval x pair
tol = 1e-12
p_raw = (null_pw >= obs_pw - tol).mean(0)
p_3pairs = (null_pw.max(3, keepdims=True) >= obs_pw - tol).mean(0)
mx_24 = null_pw[:, :, 2, :].reshape(len(signs), -1).max(1)
p_24_all = (mx_24[:, None, None] >= obs_pw[:, 2, :] - tol).mean(0)  # measure x pair
for mi, m in enumerate(MEASURES):
    for i, iv in enumerate(INTERVALS):
        for q, c in enumerate(comp_names):
            results.append({"Measure": m, "Analysis": "Pairwise interval contrast (exact sign-flip)",
                            "Test": f"{c}, change {iv}", "Statistic": D[:, mi, i, q].mean(),
                            "p_perm": p_raw[mi, i, q], "p_maxT": p_3pairs[mi, i, q],
                            "p_maxT_across_measures": p_24_all[mi, q] if i == 2 else np.nan})


# ---- Export ------------------------------------------------------------------
Path("results").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv("results/permutation_results.csv", index=False)
