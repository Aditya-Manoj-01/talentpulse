"""Shared data loading and feature definitions for the IBM HR Attrition dataset."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/"
    "master/data/emp_attrition.csv"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = Path(
    os.getenv("HR_DATA_PATH", REPO_ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
)
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", REPO_ROOT / "artifacts"))

EXPECTED_ROWS = 1470
ID_COL = "EmployeeNumber"
TARGET = "Attrition"
TREATMENT = "OverTime"
DROP_CONSTANT = ["EmployeeCount", "Over18", "StandardHours"]

CATEGORICAL = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]
NUMERIC = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]
# Gender is a protected attribute: it is audited with Fairlearn, never used as a model input.
SENSITIVE = ["Gender", "AgeBand"]
MODEL_CATEGORICAL = [c for c in CATEGORICAL if c != "Gender"]
MODEL_FEATURES = MODEL_CATEGORICAL + NUMERIC

AGE_BINS = [17, 30, 40, 50, 120]
AGE_LABELS = ["18-30", "31-40", "41-50", "51+"]

DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
JOB_ROLES = [
    "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
    "Manufacturing Director", "Research Director", "Research Scientist",
    "Sales Executive", "Sales Representative",
]


def ensure_dataset(path: Path = DEFAULT_DATA_PATH) -> Path:
    """Download the public IBM HR Attrition CSV if it is not present locally."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_hr(path: Path | str | None = None) -> pd.DataFrame:
    """Load, validate and lightly clean the dataset (target encoded 0/1)."""
    path = ensure_dataset(Path(path) if path else DEFAULT_DATA_PATH)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(df)} in {path}")
    missing = set(CATEGORICAL + NUMERIC + [ID_COL, TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    df = df.drop(columns=[c for c in DROP_CONSTANT if c in df.columns])
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    df["AgeBand"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return df


def feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[MODEL_FEATURES].copy(), df[TARGET].copy()
