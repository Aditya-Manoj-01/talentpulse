"""Cohort filter schema shared by model, analytics and gateway services."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

Department = Literal["Human Resources", "Research & Development", "Sales"]
JobRole = Literal[
    "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
    "Manufacturing Director", "Research Director", "Research Scientist",
    "Sales Executive", "Sales Representative",
]
AgeBand = Literal["18-30", "31-40", "41-50", "51+"]
YesNo = Literal["Yes", "No"]


class CohortFilter(BaseModel):
    """Only coarse, enumerated attributes are filterable. No IDs, names, or free-form values."""

    model_config = ConfigDict(extra="forbid")

    department: Department | None = None
    job_role: JobRole | None = None
    overtime: YesNo | None = None
    age_band: AgeBand | None = None
    job_level: int | None = None
    business_travel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"] | None = None


_COLUMN = {
    "department": "Department", "job_role": "JobRole", "overtime": "OverTime",
    "age_band": "AgeBand", "job_level": "JobLevel", "business_travel": "BusinessTravel",
}


def apply_filter(df: pd.DataFrame, f: CohortFilter) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for field, value in f.model_dump(exclude_none=True).items():
        mask &= df[_COLUMN[field]] == value
    return df[mask]
