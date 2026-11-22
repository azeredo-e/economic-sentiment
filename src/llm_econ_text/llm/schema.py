"""Per-article output schema. Enforced via the provider's structured output / tool calling
and validated here; free text is never parsed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Topic = Literal[
    "industria",
    "agronegocio",
    "mercado_financeiro",
    "mercado_trabalho",
    "servicos",
    "none",
]
SentimentLabel = Literal["negative", "neutral", "positive"]

# Score bands that define the label; keeps the continuous score and the 3-class label consistent.
NEUTRAL_BAND = 0.2


def label_from_score(score: float) -> SentimentLabel:
    if score <= -NEUTRAL_BAND:
        return "negative"
    if score >= NEUTRAL_BAND:
        return "positive"
    return "neutral"


def normalize_label(label: ArticleLabel) -> tuple[ArticleLabel, bool]:
    """An irrelevant article carries no sentiment and is never EPU (indices count relevant
    articles only). Returns (label, coerced); the model's raw output stays in the ledger."""
    if label.relevant or (label.sentiment == 0 and not label.epu and label.epu_reason is None):
        return label, False
    fixed = label.model_copy(
        update={"sentiment": 0.0, "sentiment_label": "neutral", "epu": False, "epu_reason": None}
    )
    return fixed, True


class ArticleLabel(BaseModel):
    relevant: bool
    topic: Topic
    sentiment: float = Field(ge=-1.0, le=1.0)
    sentiment_label: SentimentLabel
    epu: bool
    epu_reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _consistent(self) -> ArticleLabel:
        if self.relevant == (self.topic == "none"):
            raise ValueError("relevant must be False exactly when topic is 'none'")
        if self.sentiment_label != label_from_score(self.sentiment):
            raise ValueError(
                f"sentiment_label {self.sentiment_label!r} inconsistent with "
                f"sentiment {self.sentiment} (neutral band ±{NEUTRAL_BAND})"
            )
        if self.epu and not self.epu_reason:
            raise ValueError("epu_reason is required when epu is True")
        return self
