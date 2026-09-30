# Ketamine and EEG Neural Complexity in TR-MDD

Analysis code for:

> Sattar et al. (2026). Ketamine Induces Changes in Neural Complexity: A Multi-metric Cross-over EEG Study Analyzed with Mixed, Bayesian, and Permutation Approaches. (Manuscript under review September 2026).

The same data are analysed with three complementary methods.



## Study design

* **Participants:** 20 adults with treatment-resistant major depressive disorder
* **Design:** double-blind, active-controlled, within-subject, three-way crossover
* **Conditions:** Fentanyl 50 µg (active control), ketamine 0.5 mg/kg, ketamine 1.0 mg/kg (intramuscular)
* **Timepoints:** Baseline (pre-dose), 2 h and 24 h post-dose
* **Outcomes:** Higuchi's fractal dimension (HFD), Lempel-Ziv complexity (LZC), and multiscale entropy at fine (scales 1–3), intermediate (6–8) and coarse (9–10) scales



## Repository contents



.
├── README.md	
├── Sattar et al 2026 Linear mixed-effects models.R        	# 1. Linear mixed-effects models
├── Sattar et al 2026 Bayesian hierarchical models.R            # 2. Bayesian hierarchical models
├── Sattar et al 2026 Permutation tests.py   			# 3. Permutation tests





## Methods



### 1\. Linear mixed-effects models (`lmm\\\\\\\_analysis.R`)

* **Model:** `DV \\\\\\\~ Dose \\\\\\\* Time + (1 | Participant)`, REML
* **Tests:** Type III ANOVA with Satterthwaite degrees of freedom
* **Follow-up:** Bonferroni-corrected pairwise contrasts (`emmeans`)
* **Effect sizes:** marginal and conditional R², ICC, standardised coefficients
* **Also:** orthogonal polynomial contrasts (linear and quadratic Dose and Time), and dose models fitted separately at 2 h and 24 h



### 2\. Bayesian hierarchical models (`bhm\\\\\\\_analysis.R`)

* **Model:** `DV \\\\\\\~ Dose \\\\\\\* Time + (1 | Participant)`, Gaussian, fitted with `brms`
* **Priors:** weakly informative, scaled to each outcome's SD
* **Sampling:** 4 chains × 15,000 iterations (7,500 warm-up)
* **Outputs:** posterior medians, 95% credible intervals and probability of direction; posterior contrasts at each timepoint; Bayesian R², ICC and LOO
* **Hypothesis tests:** joint ROPE tests and Savage-Dickey Bayes factors
* **Note:** under treatment coding, the Dose coefficients are dose differences at Baseline, and the Time coefficients are time effects under Fentanyl.



### 3\. Permutation tests (`permutation\\\\\\\_analysis.py`)

* **Permutation scheme:** condition labels permuted within each participant, 25,000 permutations
* **Tests:** repeated-measures F for main effects and for dose effects at 2 h and 24 h; paired mean differences for pairwise comparisons
* **Interaction:** Freedman–Lane permutation for the omnibus test, plus a test for each time interval
* **Multiple comparisons:** Westfall–Young maxT within each family of tests

Reference levels in all three methods: **Fentanyl** for Dose and **Baseline** for Time.





## Data format

All scripts read the same file: `data/mdd\\\\\\\_complexity\\\\\\\_data.csv`.

It must be in long format (one row per participant, dose and timepoint), with a header row and **8 columns in this order**:

|Column|Content|Values|
|-|-|-|
|1|Participant ID|any|
|2|Dose|`Fent`, `05K`, `1K`|
|3|Time|`0hr`, `2hr`, `24hr`|
|4|HFD|numeric|
|5|LZC|numeric|
|6|MSE fine|numeric|
|7|MSE intermediate|numeric|
|8|MSE coarse|numeric|

Column names don't matter, because the scripts rename them.





## Citation

\--



## Contact

Abdullah Al Sattar, University of New England, Australia. *\[asattar2@myune.edu.au; abdullahalsattar@gmail.com]*



## License

MIT. See `LICENSE`.

