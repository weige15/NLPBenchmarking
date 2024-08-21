from typing import Sequence

import pandas as pd


class ModelAssumptions:
    query: str
    ground_truth: str
    context: Sequence[str]

    @classmethod
    def from_series(cls, series: pd.Series):
        return cls(
            query=series["question"],
            ground_truth=series["ground_truth"],
            context=series["context"],
        )
