from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("spatialforecast.core_engine")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except Exception:
    ARIMA = SARIMAX = SimpleExpSmoothing = Holt = ExponentialSmoothing = None
    adfuller = acf = pacf = acorr_ljungbox = None
    STATSMODELS_AVAILABLE = False


MODEL_CODES = {
    "SMA": 1,
    "WMA": 2,
    "SES": 3,
    "HOLT": 4,
    "HW_ADD": 5,
    "HW_MUL": 6,
    "ARIMA": 7,
    "SARIMA": 8,
}
MODEL_NAMES = {v: k for k, v in MODEL_CODES.items()}
MODEL_LABELS = {
    "SMA": "Simple Moving Average",
    "WMA": "Weighted Moving Average",
    "SES": "Simple Exponential Smoothing",
    "HOLT": "Holt Linear Trend",
    "HW_ADD": "Holt-Winters Additive",
    "HW_MUL": "Holt-Winters Multiplicative",
    "ARIMA": "ARIMA",
    "SARIMA": "SARIMA",
}


@dataclass
class ForecastResult:
    forecast: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    rmse: float
    mae: float
    smape: float
    model_name: str
    aic: float = np.nan
    bic: float = np.nan
    ljungbox_p: float = np.nan
    adf_p: float = np.nan
    mean_interval_width: float = np.nan


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    den = np.abs(yt) + np.abs(yp)
    valid = den > 1e-12
    if not np.any(valid):
        return 0.0
    return float(200.0 * np.mean(np.abs(yp[valid] - yt[valid]) / den[valid]))


def prepare_series(series: Sequence[float], min_n: int = 6) -> Optional[np.ndarray]:
    y = np.asarray(series, dtype=float).ravel()
    finite = np.isfinite(y)
    if finite.sum() < min_n:
        return None
    first = np.where(finite)[0][0]
    last = np.where(finite)[0][-1] + 1
    y = y[first:last].astype(float, copy=True)
    finite = np.isfinite(y)
    if finite.sum() < min_n:
        return None
    idx = np.arange(len(y))
    if not finite.all():
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    return y if np.all(np.isfinite(y)) else None


def _holdout_split(y: np.ndarray, holdout: int) -> Tuple[np.ndarray, np.ndarray]:
    holdout = max(1, min(int(holdout), max(1, len(y) // 3)))
    train, test = y[:-holdout], y[-holdout:]
    if len(train) < 3:
        raise ValueError("Not enough training observations after holdout split.")
    return train, test


def _validation_stats(test, pred) -> Tuple[float, float, float]:
    return rmse(test, pred), mae(test, pred), smape(test, pred)


def _residual_interval(forecast, residuals, z=1.96):
    forecast = np.asarray(forecast, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    if not np.isfinite(sigma):
        sigma = 0.0
    horizon = np.sqrt(np.arange(1, len(forecast) + 1, dtype=float))
    spread = z * sigma * horizon
    return forecast - spread, forecast + spread


def _diagnostics(y, residuals, lower, upper, aic=np.nan, bic=np.nan):
    adf_p = np.nan
    lb_p = np.nan
    try:
        if STATSMODELS_AVAILABLE and len(y) >= 8 and np.nanstd(y) > 0:
            adf_p = float(adfuller(y, autolag="AIC")[1])
    except Exception as exc:
        logger.debug("ADF stationarity test failed: %s", exc)
    try:
        resid = np.asarray(residuals, dtype=float)
        resid = resid[np.isfinite(resid)]
        if STATSMODELS_AVAILABLE and len(resid) >= 6 and np.nanstd(resid) > 0:
            lag = max(1, min(10, len(resid) // 5))
            lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[-1])
    except Exception as exc:
        logger.debug("Ljung-Box test failed: %s", exc)
    width = float(np.nanmean(np.asarray(upper) - np.asarray(lower))) if len(lower) else np.nan
    return float(aic) if np.isfinite(aic) else np.nan, float(bic) if np.isfinite(bic) else np.nan, lb_p, adf_p, width


def _sma_forecast(train, steps, window):
    history = list(map(float, train))
    window = max(1, min(int(window), len(history)))
    preds = []
    for _ in range(steps):
        pred = float(np.mean(history[-window:]))
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds)


def _sma_fitted(y, window):
    fitted = np.full(len(y), np.nan)
    window = max(1, min(int(window), len(y)))
    for i in range(window, len(y)):
        fitted[i] = np.mean(y[i-window:i])
    return fitted


def _wma_weights(window):
    w = np.arange(1, int(window) + 1, dtype=float)
    return w / w.sum()


def _wma_forecast(train, steps, window):
    history = list(map(float, train))
    window = max(1, min(int(window), len(history)))
    weights = _wma_weights(window)
    preds = []
    for _ in range(steps):
        vals = np.asarray(history[-window:], dtype=float)
        pred = float(np.dot(vals, weights[-len(vals):] / weights[-len(vals):].sum()))
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds)


def _wma_fitted(y, window):
    fitted = np.full(len(y), np.nan)
    window = max(1, min(int(window), len(y)))
    weights = _wma_weights(window)
    for i in range(window, len(y)):
        fitted[i] = float(np.dot(y[i-window:i], weights))
    return fitted


def _result_from_classical(name, y, train, test, valid_pred, forecast, fitted, lower, upper, aic=np.nan, bic=np.nan):
    score_rmse, score_mae, score_smape = _validation_stats(test, valid_pred)
    residuals = y - fitted
    aic, bic, lb_p, adf_p, width = _diagnostics(y, residuals, lower, upper, aic, bic)
    return ForecastResult(
        forecast=np.asarray(forecast, dtype=float), lower=np.asarray(lower, dtype=float), upper=np.asarray(upper, dtype=float),
        rmse=score_rmse, mae=score_mae, smape=score_smape, model_name=name,
        aic=aic, bic=bic, ljungbox_p=lb_p, adf_p=adf_p, mean_interval_width=width,
    )


def fit_sma(series, steps, window, holdout):
    y = prepare_series(series, min_n=max(6, window + 3))
    if y is None: raise ValueError("Insufficient observations for SMA.")
    train, test = _holdout_split(y, holdout)
    valid_pred = _sma_forecast(train, len(test), window)
    fitted = _sma_fitted(y, window)
    forecast = _sma_forecast(y, steps, window)
    lower, upper = _residual_interval(forecast, y - fitted)
    return _result_from_classical("SMA", y, train, test, valid_pred, forecast, fitted, lower, upper)


def fit_wma(series, steps, window, holdout):
    y = prepare_series(series, min_n=max(6, window + 3))
    if y is None: raise ValueError("Insufficient observations for WMA.")
    train, test = _holdout_split(y, holdout)
    valid_pred = _wma_forecast(train, len(test), window)
    fitted = _wma_fitted(y, window)
    forecast = _wma_forecast(y, steps, window)
    lower, upper = _residual_interval(forecast, y - fitted)
    return _result_from_classical("WMA", y, train, test, valid_pred, forecast, fitted, lower, upper)


def _fit_hw_model(y, kind, seasonal_period):
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required.")
    if kind == "SES":
        return SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
    if kind == "HOLT":
        return Holt(y, initialization_method="estimated", damped_trend=False).fit(optimized=True)
    if kind == "HW_ADD":
        return ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=seasonal_period, initialization_method="estimated").fit(optimized=True)
    if kind == "HW_MUL":
        if np.any(np.asarray(y) <= 0):
            raise ValueError("Multiplicative Holt-Winters requires strictly positive values.")
        return ExponentialSmoothing(y, trend="add", seasonal="mul", seasonal_periods=seasonal_period, initialization_method="estimated").fit(optimized=True)
    raise ValueError(kind)


def fit_exponential(series, steps, holdout, kind, seasonal_period):
    min_n = 8
    if kind in ("HW_ADD", "HW_MUL"):
        min_n = max(2 * seasonal_period + 2, 10)
    y = prepare_series(series, min_n=min_n)
    if y is None: raise ValueError(f"Insufficient observations for {kind}.")
    train, test = _holdout_split(y, holdout)
    val_model = _fit_hw_model(train, kind, seasonal_period)
    valid_pred = np.asarray(val_model.forecast(len(test)), dtype=float)
    model = _fit_hw_model(y, kind, seasonal_period)
    forecast = np.asarray(model.forecast(steps), dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    lower, upper = _residual_interval(forecast, y - fitted)
    return _result_from_classical(
        kind, y, train, test, valid_pred, forecast, fitted, lower, upper,
        getattr(model, "aic", np.nan), getattr(model, "bic", np.nan)
    )


def fit_arima(series, steps, order, holdout):
    if not STATSMODELS_AVAILABLE: raise ImportError("statsmodels is required for ARIMA.")
    y = prepare_series(series, min_n=max(10, sum(order) + 6))
    if y is None: raise ValueError("Insufficient observations for ARIMA.")
    train, test = _holdout_split(y, holdout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        valid_pred = np.asarray(val.forecast(len(test)), dtype=float)
        model = ARIMA(y, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        pred = model.get_forecast(steps=steps)
        forecast = np.asarray(pred.predicted_mean, dtype=float)
        ci = np.asarray(pred.conf_int(alpha=0.05), dtype=float)
        fitted = np.asarray(model.fittedvalues, dtype=float)
    return _result_from_classical("ARIMA", y, train, test, valid_pred, forecast, fitted, ci[:,0], ci[:,1], getattr(model,"aic",np.nan), getattr(model,"bic",np.nan))


def fit_sarima(series, steps, order, seasonal_order, holdout):
    if not STATSMODELS_AVAILABLE: raise ImportError("statsmodels is required for SARIMA.")
    s = int(seasonal_order[3])
    y = prepare_series(series, min_n=max(2*s + 2 if s > 1 else 12, sum(order)+sum(seasonal_order[:3])+8))
    if y is None: raise ValueError("Insufficient observations for SARIMA.")
    train, test = _holdout_split(y, holdout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=75)
        valid_pred = np.asarray(val.forecast(len(test)), dtype=float)
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=75)
        pred = model.get_forecast(steps=steps)
        forecast = np.asarray(pred.predicted_mean, dtype=float)
        ci = np.asarray(pred.conf_int(alpha=0.05), dtype=float)
        fitted = np.asarray(model.fittedvalues, dtype=float)
    return _result_from_classical("SARIMA", y, train, test, valid_pred, forecast, fitted, ci[:,0], ci[:,1], getattr(model,"aic",np.nan), getattr(model,"bic",np.nan))


def fit_and_forecast(series, model, steps, holdout, sma_window, wma_window, arima_order, sarima_order, seasonal_period):
    model = model.upper()
    if model == "SMA": return fit_sma(series, steps, sma_window, holdout)
    if model == "WMA": return fit_wma(series, steps, wma_window, holdout)
    if model in ("SES", "HOLT", "HW_ADD", "HW_MUL"):
        return fit_exponential(series, steps, holdout, model, seasonal_period)
    if model == "ARIMA": return fit_arima(series, steps, arima_order, holdout)
    if model == "SARIMA": return fit_sarima(series, steps, sarima_order, (1,0,0,seasonal_period), holdout)
    if model == "AUTO":
        best = None
        for candidate in MODEL_CODES.keys():
            try:
                cur = fit_and_forecast(series, candidate, steps, holdout, sma_window, wma_window, arima_order, sarima_order, seasonal_period)
                if np.isfinite(cur.rmse) and (best is None or cur.rmse < best.rmse):
                    best = cur
            except Exception as exc:
                logger.debug("Model '%s' failed during AUTO selection: %s", candidate, exc)
                continue
        if best is None: raise ValueError("No candidate model could be fitted.")
        return best
    raise ValueError(f"Unknown model {model}")


def aggregate_stack(stack: np.ndarray, block: int, statistic="mean") -> np.ndarray:
    stack = np.asarray(stack, dtype=float)
    if block <= 1: return stack
    t, rows, cols = stack.shape
    out_rows, out_cols = math.ceil(rows/block), math.ceil(cols/block)
    out = np.full((t, out_rows, out_cols), np.nan, dtype=float)
    reducer = np.nanmedian if statistic == "median" else np.nanmean
    for r in range(out_rows):
        r0, r1 = r*block, min(rows, (r+1)*block)
        for c in range(out_cols):
            c0, c1 = c*block, min(cols, (c+1)*block)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[:,r,c] = reducer(stack[:,r0:r1,c0:c1], axis=(1,2))
    return out


def forecast_cube(stack, model, steps, holdout, sma_window, wma_window, arima_order, sarima_order, seasonal_period, progress=None, canceled=None):
    stack = np.asarray(stack, dtype=float)
    _, rows, cols = stack.shape
    f3 = lambda: np.full((steps, rows, cols), np.nan, dtype=np.float32)
    f2 = lambda: np.full((rows, cols), np.nan, dtype=np.float32)
    result = {
        "forecast": f3(), "lower": f3(), "upper": f3(),
        "rmse": f2(), "mae": f2(), "smape": f2(), "aic": f2(), "bic": f2(),
        "ljungbox_p": f2(), "adf_p": f2(), "uncertainty_width": f2(),
        "model_code": np.zeros((rows, cols), dtype=np.int16),
    }
    total, done = rows*cols, 0
    for r in range(rows):
        for c in range(cols):
            if canceled and canceled(): raise InterruptedError("Canceled")
            try:
                fr = fit_and_forecast(stack[:,r,c], model, steps, holdout, sma_window, wma_window, arima_order, sarima_order, seasonal_period)
                result["forecast"][:,r,c] = fr.forecast
                result["lower"][:,r,c] = fr.lower
                result["upper"][:,r,c] = fr.upper
                for key in ("rmse","mae","smape","aic","bic","ljungbox_p","adf_p","mean_interval_width"):
                    target = "uncertainty_width" if key == "mean_interval_width" else key
                    result[target][r,c] = getattr(fr, key)
                result["model_code"][r,c] = MODEL_CODES[fr.model_name]
            except Exception as exc:
                logger.warning("Pixel (%s,%s) forecast failed: %s", r, c, exc)
            done += 1
            if progress: progress(100.0*done/total)
    return result


def refit_for_plot(series, result_model, steps, sma_window, wma_window, arima_order, sarima_order, seasonal_period):
    y = prepare_series(series)
    if y is None: raise ValueError("Insufficient observations.")
    name = result_model.upper()
    if name == "SMA":
        fitted = _sma_fitted(y, sma_window); fc = _sma_forecast(y, steps, sma_window); lo,hi = _residual_interval(fc, y-fitted)
    elif name == "WMA":
        fitted = _wma_fitted(y, wma_window); fc = _wma_forecast(y, steps, wma_window); lo,hi = _residual_interval(fc, y-fitted)
    elif name in ("SES","HOLT","HW_ADD","HW_MUL"):
        model = _fit_hw_model(y, name, seasonal_period); fitted=np.asarray(model.fittedvalues,float); fc=np.asarray(model.forecast(steps),float); lo,hi=_residual_interval(fc,y-fitted)
    elif name == "ARIMA":
        model=ARIMA(y,order=arima_order,enforce_stationarity=False,enforce_invertibility=False).fit(); fitted=np.asarray(model.fittedvalues,float); pr=model.get_forecast(steps); fc=np.asarray(pr.predicted_mean,float); ci=np.asarray(pr.conf_int(alpha=.05),float); lo,hi=ci[:,0],ci[:,1]
    else:
        model=SARIMAX(y,order=sarima_order,seasonal_order=(1,0,0,seasonal_period),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=75); fitted=np.asarray(model.fittedvalues,float); pr=model.get_forecast(steps); fc=np.asarray(pr.predicted_mean,float); ci=np.asarray(pr.conf_int(alpha=.05),float); lo,hi=ci[:,0],ci[:,1]
    residuals = y - fitted
    return y, fitted, fc, lo, hi, residuals


def representative_diagnostics(series, max_lag=24):
    y = prepare_series(series)
    if y is None or not STATSMODELS_AVAILABLE: return None
    nlags = max(1, min(max_lag, len(y)//3))
    return {
        "acf": np.asarray(acf(y, nlags=nlags, fft=True), dtype=float),
        "pacf": np.asarray(pacf(y, nlags=min(nlags, max(1, len(y)//2-1)), method="ywm"), dtype=float),
    }
