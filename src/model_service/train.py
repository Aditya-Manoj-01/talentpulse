"""Benchmark calibrated Logistic Regression vs. calibrated XGBoost for attrition prediction.

- Stratified 5-fold outer CV -> out-of-fold (OOF) probabilities for honest ROC-AUC / Brier / PR-AUC.
- Each candidate is wrapped in CalibratedClassifierCV (inner CV), so calibration never sees the outer test fold.
- Everything is logged to MLflow (one parent run, one nested run per candidate).
- The winning model is refit on all data and exported with a SHAP explainer + global importances.
- A Fairlearn audit (Gender, AgeBand) runs on the winner's OOF predictions.

Usage:
    PYTHONPATH=src python -m model_service.train --tracking-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common.data import (
    ARTIFACTS_DIR,
    ID_COL,
    MODEL_CATEGORICAL,
    NUMERIC,
    SENSITIVE,
    feature_target,
    load_hr,
)

log = logging.getLogger("train")


# --------------------------------------------------------------------------- model definitions
def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), MODEL_CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def base_estimators(seed: int) -> dict[str, Pipeline]:
    """Uncalibrated pipelines (also used for SHAP explanations)."""
    return {
        "logreg": Pipeline(
            [
                ("pre", make_preprocessor()),
                ("clf", LogisticRegression(C=0.3, max_iter=5000, solver="lbfgs")),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("pre", make_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        min_child_weight=3,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=2,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


CALIBRATION_METHOD = {"logreg": "sigmoid", "xgboost": "sigmoid"}  # isotonic overfits at n≈1.2k per fold


def calibrated(name: str, pipe: Pipeline) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(estimator=pipe, method=CALIBRATION_METHOD[name], cv=5)


# --------------------------------------------------------------------------- evaluation
def fold_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "log_loss": log_loss(y_true, p),
    }


def cross_validate(name: str, X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    per_fold = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        model = calibrated(name, base_estimators(seed)[name])
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        m = fold_metrics(y.iloc[te].to_numpy(), oof[te])
        per_fold.append(m)
        log.info("%s fold %d: AUC=%.3f Brier=%.3f", name, fold, m["roc_auc"], m["brier"])
    summary = {}
    for k in per_fold[0]:
        vals = np.array([f[k] for f in per_fold])
        summary[f"cv_{k}_mean"] = float(vals.mean())
        summary[f"cv_{k}_std"] = float(vals.std(ddof=1))
    summary.update({f"oof_{k}": float(v) for k, v in fold_metrics(y.to_numpy(), oof).items()})
    return oof, summary, per_fold


def plot_calibration(y: np.ndarray, oofs: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in oofs.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed attrition rate")
    ax.set_title("Calibration (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- SHAP
def export_shap(name: str, X: pd.DataFrame, y: pd.Series, seed: int, out: Path) -> dict[str, float]:
    """Explain the uncalibrated base estimator (log-odds space; sigmoid calibration is monotone)."""
    pipe = base_estimators(seed)[name].fit(X, y)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out(), index=X.index)

    if name == "xgboost":
        explainer = shap.TreeExplainer(clf)
    else:
        background = shap.maskers.Independent(Xt, max_samples=200)
        explainer = shap.LinearExplainer(clf, background)
    sv = explainer(Xt)

    # Aggregate one-hot columns back to their source feature for a human-readable global ranking.
    contrib = pd.DataFrame(np.abs(sv.values), columns=Xt.columns)
    source = {}
    for col in Xt.columns:
        source[col] = next((c for c in MODEL_CATEGORICAL if col.startswith(f"{c}_")), col)
    global_imp = contrib.T.groupby(pd.Series(source)).sum().T.mean().sort_values(ascending=False)

    shap.summary_plot(sv.values, Xt, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    joblib.dump({"explainer": explainer, "pipeline": pipe}, out / "shap_explainer.joblib")
    importance = {k: round(float(v), 5) for k, v in global_imp.items()}
    (out / "shap_global_importance.json").write_text(json.dumps(importance, indent=2))
    return importance


# --------------------------------------------------------------------------- Fairlearn
def fairness_audit(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y_pred = (p >= threshold).astype(int)
    report: dict = {"threshold": threshold, "groups": {}, "disparities": {}}
    for feature in SENSITIVE:
        sf = df[feature].astype(str)
        mf = MetricFrame(
            metrics={
                "count": count,
                "selection_rate": selection_rate,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
            },
            y_true=y,
            y_pred=y_pred,
            sensitive_features=sf,
        )
        auc_by_group = {
            g: float(roc_auc_score(y[sf == g], p[sf == g])) if len(np.unique(y[sf == g])) == 2 else None
            for g in sorted(sf.unique())
        }
        groups = mf.by_group.round(4).to_dict(orient="index")
        for g, auc in auc_by_group.items():
            groups[g]["roc_auc"] = None if auc is None else round(auc, 4)
        report["groups"][feature] = groups
        report["disparities"][feature] = {
            "demographic_parity_difference": round(
                float(demographic_parity_difference(y, y_pred, sensitive_features=sf)), 4
            ),
            "equalized_odds_difference": round(
                float(equalized_odds_difference(y, y_pred, sensitive_features=sf)), 4
            ),
        }
    return report


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.8, 76)
    scores = [f1_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    ap.add_argument("--experiment", default="talentpulse-attrition")
    ap.add_argument("--out", default=str(ARTIFACTS_DIR))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fairness-tolerance", type=float, default=0.10,
                    help="Flag if equalized-odds difference exceeds this.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_hr(args.data)
    X, y = feature_target(df)
    log.info("Loaded %d rows, positive class = %.1f%%", len(y), 100 * y.mean())

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    results: dict[str, dict] = {}
    oofs: dict[str, np.ndarray] = {}
    with mlflow.start_run(run_name="benchmark") as parent:
        mlflow.log_params({"n_rows": len(y), "positive_rate": round(float(y.mean()), 4),
                           "n_splits": args.n_splits, "seed": args.seed})
        for name in ("logreg", "xgboost"):
            with mlflow.start_run(run_name=f"{name}_calibrated", nested=True):
                clf_params = base_estimators(args.seed)[name].named_steps["clf"].get_params()
                mlflow.log_params({f"clf__{k}": v for k, v in clf_params.items()
                                   if isinstance(v, (int, float, str, bool)) and v is not None})
                mlflow.log_param("calibration", CALIBRATION_METHOD[name])
                oof, summary, per_fold = cross_validate(name, X, y, args.n_splits, args.seed)
                mlflow.log_metrics(summary)
                for i, f in enumerate(per_fold):
                    mlflow.log_metrics({f"fold_{k}": v for k, v in f.items()}, step=i)
                results[name], oofs[name] = summary, oof
                log.info("%s: AUC %.3f ± %.3f | Brier %.3f", name,
                         summary["cv_roc_auc_mean"], summary["cv_roc_auc_std"], summary["cv_brier_mean"])

        # Winner: highest mean AUC; Brier breaks near-ties (< 0.005 AUC).
        ranked = sorted(results, key=lambda n: (-round(results[n]["cv_roc_auc_mean"], 3),
                                                results[n]["cv_brier_mean"]))
        best = ranked[0]
        mlflow.set_tag("best_model", best)
        log.info("Best model: %s", best)

        plot_calibration(y.to_numpy(), oofs, out / "calibration_curve.png")
        mlflow.log_artifact(str(out / "calibration_curve.png"))

        final = calibrated(best, base_estimators(args.seed)[best]).fit(X, y)
        joblib.dump(final, out / "model.joblib")
        try:
                        mlflow.sklearn.log_model(
                final,
                name="model",
                input_example=X.head(3),
                # Our own sklearn/xgboost classes; safe to deserialize.
                skops_trusted_types=[
                    "sklearn.calibration._CalibratedClassifier",
                    "sklearn.calibration._SigmoidCalibration",
                    "xgboost.sklearn.XGBClassifier",
                    "xgboost.core.Booster",
                ],
            )
        except Exception as exc:  # mlflow-skinny / older servers: still keep the joblib artifact
            log.warning("mlflow.sklearn.log_model failed (%s); logging joblib instead", exc)
            mlflow.log_artifact(str(out / "model.joblib"))

        importance = export_shap(best, X, y, args.seed, out)
        mlflow.log_artifacts(str(out), artifact_path="explainability")
        log.info("Top SHAP drivers: %s", list(importance)[:5])

        threshold = best_f1_threshold(y.to_numpy(), oofs[best])
        fairness = fairness_audit(df, y.to_numpy(), oofs[best], threshold)
        flagged = {f: d for f, d in fairness["disparities"].items()
                   if d["equalized_odds_difference"] > args.fairness_tolerance}
        fairness["flagged"] = flagged
        (out / "fairness_report.json").write_text(json.dumps(fairness, indent=2))
        mlflow.log_artifact(str(out / "fairness_report.json"))
        for f, d in fairness["disparities"].items():
            mlflow.log_metrics({f"fair_{f}_{k}": v for k, v in d.items()})

        # OOF scores feed the serving layer: cohort risk is reported from predictions the model
        # never trained on, so served aggregates are not inflated by in-sample fit.
        pd.DataFrame({ID_COL: df[ID_COL], "risk": oofs[best]}).to_csv(out / "oof_scores.csv", index=False)

        metrics = {
            "best_model": best,
            "decision_threshold": threshold,
            "positive_rate": round(float(y.mean()), 4),
            "candidates": {n: {k: round(v, 4) for k, v in r.items()} for n, r in results.items()},
            "mlflow_run_id": parent.info.run_id,
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(out / "metrics.json"))

    print(json.dumps(metrics["candidates"], indent=2))
    print("Fairness disparities:", json.dumps(fairness["disparities"], indent=2))
    if flagged:
        print(f"WARNING: equalized-odds difference above {args.fairness_tolerance} for {list(flagged)}")


if __name__ == "__main__":
    main()
