"""Keep fitted scaler feature names when Forecasting generates target scales."""

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import RobustScaler, StandardScaler


class InferenceTimeSeriesDataSet(TimeSeriesDataSet):
    def transform_values(
        self, name, values, data=None, inverse=False, group_id=False, **kwargs,
    ):
        # Forecasting passes close_center/close_scale as scalar tensors during
        # __getitem__, although their scalers were fitted on named DataFrames.
        # Restore the actual feature name without refitting or changing metadata.
        transformer = self.get_transformer(name, group_id=group_id)
        if (
            name in self.reals
            and isinstance(transformer, (StandardScaler, RobustScaler))
            and hasattr(transformer, 'feature_names_in_')
            and not isinstance(values, (pd.Series, pd.DataFrame))
        ):
            # sklearn converts tensor inputs to float64. Preserve that conversion
            # so target-scale values and Torch scalar assignment stay identical.
            dtype = np.float64 if isinstance(values, torch.Tensor) else None
            values = pd.Series(np.asarray(values, dtype=dtype).reshape(-1), name=name)
        return super().transform_values(
            name, values, data=data, inverse=inverse, group_id=group_id, **kwargs,
        )
