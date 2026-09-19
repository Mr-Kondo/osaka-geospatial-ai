"""Temporal partitions are defined by label year, never by random row splits."""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or actual.ndim != 1 or len(actual) == 0:
        raise ValueError("Expected nonempty 1D actual/predicted arrays of identical shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Predictions and labels must be finite")
    return {
        "n": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
    }


def temporal_fold(frame, target_year, target, first_origin_year):
    eligible = frame[frame[target].notna() & frame.year.ge(first_origin_year)]
    train = eligible[eligible.target_year < target_year].copy()
    test = eligible[eligible.target_year.eq(target_year)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty temporal fold for target year {target_year}")
    if train.target_year.max() > test.year.min():
        raise ValueError("Temporal leakage: a training label is not available at prediction origin")
    if not (eligible.target_year == eligible.year + 1).all():
        raise ValueError("Only consecutive next-year labels are supported")
    return train, test
