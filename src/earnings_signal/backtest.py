from __future__ import annotations

import pandas as pd


def build_long_short_backtest(
    signals: pd.DataFrame,
    model_name: str = "NLP-enhanced",
    top_quantile: float = 0.25,
) -> pd.DataFrame:
    model_signals = signals.query("model == @model_name").copy()
    if model_signals.empty:
        columns = ["period", "long_return", "short_return", "long_short_return"]
        return pd.DataFrame(columns=columns)

    model_signals["period"] = (
        pd.to_datetime(model_signals["call_date"]).dt.to_period("Q").astype(str)
    )
    rows = []
    for period, group in model_signals.groupby("period"):
        group = group.sort_values("signal_score", ascending=False)
        bucket_size = max(1, int(round(len(group) * top_quantile)))
        longs = group.head(bucket_size)
        shorts = group.tail(bucket_size)
        rows.append(
            {
                "period": period,
                "long_return": longs["market_adjusted_return_5d"].mean(),
                "short_return": shorts["market_adjusted_return_5d"].mean(),
                "long_short_return": longs["market_adjusted_return_5d"].mean()
                - shorts["market_adjusted_return_5d"].mean(),
                "events": len(group),
            }
        )

    backtest = pd.DataFrame(rows).sort_values("period")
    backtest["cumulative_long_short"] = (1 + backtest["long_short_return"]).cumprod() - 1
    return backtest
