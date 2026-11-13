from typing import Any, Literal

from pydantic import BaseModel

from src.config.configs import settings

Urgency = Literal["low", "medium", "high", "critical"]
Sentiment = Literal["negative", "neutral", "positive"]


class WeakLabel(BaseModel):
    """
    A weak label for a support ticket.
    """

    rationale: str  # Just one sentence, makes debugging and auditing easier
    urgency: Urgency
    sentiment: Sentiment


LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "urgency": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "sentiment": {"type": "string", "enum": ["negative", "neutral", "positive"]},
    },
    "required": ["rationale", "urgency", "sentiment"],
    "additionalProperties": False,
}


def build_body(text: str, guidelines: str) -> dict:
    """
    Builds the body of the request to the labeling model.
    """
    return {
        "model": settings.ml.LABELLING_MODEL,
        "instructions": guidelines,
        "input": f"<complaint>\n{text}\n</complaint>",
        "reasoning": {"effort": "low"},
        "max_output_tokens": 2048,  # NOTE: Set a reasonable limit for the output tokens
        "text": {
            "format": {
                "type": "json_schema",
                "name": "weak_label",
                "strict": True,
                "schema": LABEL_SCHEMA,
            }
        },
    }


def load_guidelines() -> str:
    """
    Loads the labeling guidelines from a text file.
    """
    with open(settings.ml.LABELLING_GUIDELINES_PATH, encoding="utf-8") as f:
        return f.read()
