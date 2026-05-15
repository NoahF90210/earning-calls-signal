from __future__ import annotations

import re

import pandas as pd

POSITIVE_TERMS = {
    "accelerating",
    "beat",
    "confidence",
    "demand",
    "expanding",
    "growth",
    "improved",
    "momentum",
    "record",
    "resilient",
    "strong",
    "upside",
}
NEGATIVE_TERMS = {
    "challenging",
    "decline",
    "delayed",
    "headwind",
    "inflation",
    "miss",
    "pressure",
    "risk",
    "slowdown",
    "soft",
    "uncertain",
    "weak",
}
UNCERTAINTY_TERMS = {
    "could",
    "expect",
    "guidance",
    "may",
    "might",
    "outlook",
    "possible",
    "uncertain",
    "visibility",
    "volatility",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", str(text).lower())


def count_terms(tokens: list[str], terms: set[str]) -> int:
    return sum(token in terms for token in tokens)


def add_text_features(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for text in events["transcript"]:
        tokens = tokenize(text)
        total_words = max(len(tokens), 1)
        positive_count = count_terms(tokens, POSITIVE_TERMS)
        negative_count = count_terms(tokens, NEGATIVE_TERMS)
        uncertainty_count = count_terms(tokens, UNCERTAINTY_TERMS)
        rows.append(
            {
                "word_count": total_words,
                "positive_term_rate": positive_count / total_words,
                "negative_term_rate": negative_count / total_words,
                "uncertainty_term_rate": uncertainty_count / total_words,
                "net_sentiment": (positive_count - negative_count) / total_words,
                "question_count": str(text).count("?"),
                "exclamation_count": str(text).count("!"),
            }
        )

    feature_df = pd.DataFrame(rows, index=events.index)
    return pd.concat([events.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


def top_language_terms(
    events: pd.DataFrame,
    target_col: str = "target_positive_5d",
    min_count: int = 1,
    limit: int = 15,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for label, group in events.groupby(target_col):
        token_counts: dict[str, int] = {}
        for text in group["transcript"]:
            for token in set(tokenize(text)):
                if len(token) >= 4:
                    token_counts[token] = token_counts.get(token, 0) + 1
        for token, count in token_counts.items():
            records.append({"term": token, "target": int(label), "doc_count": count})

    if not records:
        return pd.DataFrame(columns=["term", "positive_count", "negative_count", "lift"])

    pivot = (
        pd.DataFrame(records)
        .pivot_table(
            index="term",
            columns="target",
            values="doc_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns={0: "negative_count", 1: "positive_count"})
    )
    for column in ["negative_count", "positive_count"]:
        if column not in pivot:
            pivot[column] = 0
    pivot["lift"] = pivot["positive_count"] - pivot["negative_count"]
    return (
        pivot.reset_index()
        .query("positive_count + negative_count >= @min_count")
        .sort_values("lift", ascending=False)
        .head(limit)
    )
