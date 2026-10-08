"""Builds notebooks/01_causal_and_stats.ipynb from version-controlled cell sources.

Keeping cells here (rather than hand-editing JSON) makes notebook diffs reviewable.
Run:  python scripts/build_notebook.py && jupyter nbconvert --execute --inplace notebooks/01_causal_and_stats.ipynb
"""
from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = [
("md", """# TalentPulse — 01 · Attrition statistics & causal analysis of overtime

**Question:** Is overtime the top attrition driver in the IBM HR Attrition dataset (n = 1,470), and how much of
the raw gap survives adjustment for confounders?

Pipeline:
1. Base rates and the raw overtime gap
2. Chi-square tests of independence across all categorical drivers (Holm-corrected), with Cramér's V
3. Mann-Whitney U tests across numeric drivers (Holm-corrected), with rank-biserial effect sizes
4. Crude and adjusted logistic-regression odds ratios (statsmodels)
5. Propensity-score matching (1:1 nearest neighbour, caliper) for the ATT of overtime, with balance diagnostics,
   bootstrap CI, and an IPW cross-check

> Runs on Databricks (attach the repo, `%pip install -r requirements-dev.txt`) or locally. The final cell writes
> `artifacts/causal_results.json`, which the analytics service serves through the MCP `get_overtime_impact` tool."""),
("code", """import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common.data import load_hr, TARGET, TREATMENT, CATEGORICAL, NUMERIC, ARTIFACTS_DIR

SEED = 42
rng = np.random.default_rng(SEED)
pd.set_option("display.width", 140)

df = load_hr()
print(df.shape)
print(f"Base attrition rate: {df[TARGET].mean():.3%}  ({df[TARGET].sum()} / {len(df)})")"""),
("md", "## 1 · The raw overtime gap"),
("code", """ct = pd.crosstab(df[TREATMENT], df[TARGET], margins=True)
rates = df.groupby(TREATMENT)[TARGET].agg(n="size", attrited="sum", rate="mean")
print(ct, "\\n")
print(rates)
rate_yes, rate_no = rates.loc["Yes", "rate"], rates.loc["No", "rate"]
print(f"\\nOvertime attrition {rate_yes:.1%} vs {rate_no:.1%}  -> raw gap {100*(rate_yes-rate_no):+.1f} pts, "
      f"risk ratio {rate_yes/rate_no:.2f}")"""),
("md", "## 2 · Chi-square tests across categorical drivers"),
("code", """def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

cat_rows = []
for col in CATEGORICAL:
    table = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    cat_rows.append({"feature": col, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v(table)})

chi_df = pd.DataFrame(cat_rows)
chi_df["p_holm"] = multipletests(chi_df["p_value"], method="holm")[1]
chi_df = chi_df.sort_values("cramers_v", ascending=False).reset_index(drop=True)
chi_df"""),
("md", "## 3 · Mann-Whitney U tests across numeric drivers (attrited vs. stayed)"),
("code", """mw_rows = []
left, stayed = df[df[TARGET] == 1], df[df[TARGET] == 0]
for col in NUMERIC:
    u, p = stats.mannwhitneyu(left[col], stayed[col], alternative="two-sided")
    # rank-biserial correlation: positive => attriters tend to have HIGHER values
    rbc = 2 * u / (len(left) * len(stayed)) - 1
    mw_rows.append({"feature": col, "median_left": left[col].median(), "median_stayed": stayed[col].median(),
                    "U": u, "p_value": p, "rank_biserial": rbc})

mw_df = pd.DataFrame(mw_rows)
mw_df["p_holm"] = multipletests(mw_df["p_value"], method="holm")[1]
mw_df["abs_effect"] = mw_df["rank_biserial"].abs()
mw_df = mw_df.sort_values("abs_effect", ascending=False).reset_index(drop=True)
mw_df.head(12)"""),
("md", "## 4 · Logistic regression odds ratios (crude and adjusted)"),
("code", """model_df = df.copy()
model_df["OverTimeYes"] = (model_df[TREATMENT] == "Yes").astype(int)

crude = smf.logit(f"{TARGET} ~ OverTimeYes", data=model_df).fit(disp=False)

adj_formula = (f"{TARGET} ~ OverTimeYes + Age + np.log(MonthlyIncome) + JobLevel + TotalWorkingYears + "
               "YearsAtCompany + DistanceFromHome + NumCompaniesWorked + JobSatisfaction + "
               "EnvironmentSatisfaction + WorkLifeBalance + JobInvolvement + StockOptionLevel + "
               "C(MaritalStatus) + C(BusinessTravel) + C(Department) + C(Gender)")
adjusted = smf.logit(adj_formula, data=model_df).fit(disp=False, maxiter=200)

def or_table(res):
    ci = np.exp(res.conf_int())
    return pd.DataFrame({"odds_ratio": np.exp(res.params), "ci_low": ci[0], "ci_high": ci[1],
                         "p_value": res.pvalues}).drop(index="Intercept")

crude_or = or_table(crude).loc["OverTimeYes"]
adj_or = or_table(adjusted)
print(f"Crude OR (overtime):    {crude_or.odds_ratio:.2f}  [{crude_or.ci_low:.2f}, {crude_or.ci_high:.2f}]")
print(f"Adjusted OR (overtime): {adj_or.loc['OverTimeYes','odds_ratio']:.2f}  "
      f"[{adj_or.loc['OverTimeYes','ci_low']:.2f}, {adj_or.loc['OverTimeYes','ci_high']:.2f}]")
print(f"Pseudo-R2 adjusted model: {adjusted.prsquared:.3f}")
adj_or.sort_values("odds_ratio", ascending=False)"""),
("md", """### Which driver matters most? Drop-one likelihood-ratio tests in a multivariable logit
Univariate effect sizes (Cramér's V vs. rank-biserial r) are not on a common scale, so the ranking is done inside one
adjusted model: each term is removed in turn and the likelihood-ratio χ² measures how much unique explanatory power it
carried. `Department` is omitted because it is nested in `JobRole` (keeping both causes quasi-separation)."""),
("code", """import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

rank_terms = (["OverTimeYes"] + NUMERIC +
              ["C(BusinessTravel)", "C(EducationField)", "C(Gender)", "C(JobRole)", "C(MaritalStatus)"])
full_fit = smf.logit(f"{TARGET} ~ " + " + ".join(rank_terms), data=model_df).fit(disp=False, maxiter=500)
lr_rows = []
for term in rank_terms:
    reduced = smf.logit(f"{TARGET} ~ " + " + ".join(t for t in rank_terms if t != term),
                        data=model_df).fit(disp=False, maxiter=500)
    lr = 2 * (full_fit.llf - reduced.llf)
    dof = int(full_fit.df_model - reduced.df_model)
    lr_rows.append({"feature": term, "lr_chi2": lr, "dof": dof, "p_value": stats.chi2.sf(lr, dof)})
ranking = pd.DataFrame(lr_rows).sort_values("p_value").reset_index(drop=True)
ranking["p_holm"] = multipletests(ranking["p_value"], method="holm")[1]
ranking.head(10)"""),
("md", """## 5 · Propensity-score matching: ATT of overtime on attrition

Treatment = `OverTime == Yes`. Covariates are restricted to plausibly **pre-treatment** characteristics
(demographics, role, level, tenure, pay, commute). Satisfaction / work-life-balance scores are excluded because
overtime can plausibly *cause* them (they are mediators, and adjusting for them would bias the total effect)."""),
("code", """PS_NUM = ["Age", "DistanceFromHome", "Education", "JobLevel", "MonthlyIncome", "NumCompaniesWorked",
          "TotalWorkingYears", "YearsAtCompany", "StockOptionLevel"]
PS_CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]

T = (df[TREATMENT] == "Yes").astype(int).to_numpy()
Y = df[TARGET].to_numpy()

ps_pre = ColumnTransformer([("num", StandardScaler(), PS_NUM),
                            ("cat", OneHotEncoder(drop="first", sparse_output=False), PS_CAT)])
Xps = ps_pre.fit_transform(df)
ps_model = LogisticRegression(max_iter=5000, C=1.0).fit(Xps, T)
ps = ps_model.predict_proba(Xps)[:, 1]
logit_ps = np.log(ps / (1 - ps))
print(f"Propensity range treated [{ps[T==1].min():.3f}, {ps[T==1].max():.3f}]  "
      f"control [{ps[T==0].min():.3f}, {ps[T==0].max():.3f}]")"""),
("code", """def match_att(logit_ps, T, Y, caliper_sd=0.2, seed=SEED):
    \"\"\"Greedy 1:1 nearest-neighbour matching on logit(PS) WITHOUT replacement, caliper = 0.2 SD (Austin 2011).
    Treated units are visited in random order to avoid order bias.\"\"\"
    caliper = caliper_sd * logit_ps.std()
    order = np.random.default_rng(seed).permutation(np.where(T == 1)[0])
    available = (T == 0).copy()
    matched_t, matched_c = [], []
    for i in order:
        cand = np.flatnonzero(available)
        if cand.size == 0:
            break
        d = np.abs(logit_ps[cand] - logit_ps[i])
        j = int(d.argmin())
        if d[j] <= caliper:
            matched_t.append(i)
            matched_c.append(cand[j])
            available[cand[j]] = False
    matched_t, matched_c = np.asarray(matched_t), np.asarray(matched_c)
    att = Y[matched_t].mean() - Y[matched_c].mean()
    return att, matched_t, matched_c

att, mt, mc = match_att(logit_ps, T, Y)
print(f"Matched treated: {len(mt)} / {T.sum()}")
print(f"Attrition | matched overtime: {Y[mt].mean():.1%}   matched controls: {Y[mc].mean():.1%}")
print(f"ATT (PSM): {100*att:+.1f} pts")"""),
("md", "### Covariate balance (standardized mean differences; |SMD| < 0.1 is the usual target)"),
("code", """def smd(a, b):
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled

Xbal = pd.DataFrame(Xps, columns=ps_pre.get_feature_names_out())
balance = pd.DataFrame({
    "smd_before": [smd(Xbal.loc[T == 1, c], Xbal.loc[T == 0, c]) for c in Xbal],
    "smd_after":  [smd(Xbal.iloc[mt][c], Xbal.iloc[mc][c]) for c in Xbal],
}, index=Xbal.columns)
print(f"Max |SMD| before: {balance.smd_before.abs().max():.3f}   after: {balance.smd_after.abs().max():.3f}")
balance.reindex(balance.smd_before.abs().sort_values(ascending=False).index).head(10)"""),
("md", "### Bootstrap CI for the ATT (re-estimating the propensity model each replicate)"),
("code", """B = 300
boot = []
n = len(df)
for _ in range(B):
    idx = rng.integers(0, n, n)
    Tb, Yb = T[idx], Y[idx]
    Xb = ps_pre.fit_transform(df.iloc[idx])
    psb = LogisticRegression(max_iter=5000).fit(Xb, Tb).predict_proba(Xb)[:, 1]
    psb = np.clip(psb, 1e-6, 1 - 1e-6)
    boot.append(match_att(np.log(psb / (1 - psb)), Tb, Yb, seed=int(rng.integers(1e9)))[0])
ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
print(f"ATT (PSM) = {100*att:+.1f} pts, 95% bootstrap CI [{100*ci_low:+.1f}, {100*ci_high:+.1f}]")"""),
("md", "### Robustness: inverse-probability weighting (ATT weights, trimmed propensities)"),
("code", """ps_t = np.clip(ps, 0.02, 0.98)
w = np.where(T == 1, 1.0, ps_t / (1 - ps_t))
att_ipw = Y[T == 1].mean() - np.average(Y[T == 0], weights=w[T == 0])
print(f"ATT (IPW) = {100*att_ipw:+.1f} pts")"""),
("md", "## 6 · Summary & export"),
("code", """top_driver = ranking.iloc[0]
results = {
    "n_records": int(len(df)),
    "base_attrition_rate": round(float(df[TARGET].mean()), 4),
    "overtime": {
        "attrition_rate_overtime": round(float(rate_yes), 4),
        "attrition_rate_no_overtime": round(float(rate_no), 4),
        "raw_gap_pts": round(float(100 * (rate_yes - rate_no)), 1),
        "chi2_p_value": float(chi_df.set_index("feature").loc["OverTime", "p_value"]),
        "cramers_v": round(float(chi_df.set_index("feature").loc["OverTime", "cramers_v"]), 3),
        "crude_odds_ratio": round(float(crude_or.odds_ratio), 2),
        "adjusted_odds_ratio": round(float(adj_or.loc["OverTimeYes", "odds_ratio"]), 2),
        "adjusted_or_ci": [round(float(adj_or.loc["OverTimeYes", "ci_low"]), 2),
                           round(float(adj_or.loc["OverTimeYes", "ci_high"]), 2)],
        "psm_att_pts": round(float(100 * att), 1),
        "psm_att_ci_pts": [round(float(100 * ci_low), 1), round(float(100 * ci_high), 1)],
        "psm_matched_treated": int(len(mt)),
        "psm_max_abs_smd_after": round(float(balance.smd_after.abs().max()), 3),
        "ipw_att_pts": round(float(100 * att_ipw), 1),
    },
    "top_driver": {"feature": top_driver.feature, "lr_chi2": round(float(top_driver.lr_chi2), 1),
                   "runner_up": ranking.iloc[1].feature, "runner_up_lr_chi2": round(float(ranking.iloc[1].lr_chi2), 1)},
    "driver_ranking": ranking.head(10).round(4).to_dict(orient="records"),
}
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "causal_results.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps(results["overtime"], indent=2))
print("Top driver (LR test):", results["top_driver"])"""),
]


def build(out: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in CELLS]
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "notebooks" / "01_causal_and_stats.ipynb"
    build(target)
    print(f"Wrote {target}")
