from typing import get_args

import pytest
from pydantic import ValidationError

from src.model.labelling.schema import (
    LABEL_SCHEMA,
    Sentiment,
    Urgency,
    WeakLabel,
    build_body,
)


def test_schema_enums_match_label_types() -> None:
    properties = LABEL_SCHEMA["properties"]

    assert properties["urgency"]["enum"] == list(get_args(Urgency))
    assert properties["sentiment"]["enum"] == list(get_args(Sentiment))


def test_schema_meets_strict_mode_rules() -> None:
    assert set(LABEL_SCHEMA["required"]) == set(LABEL_SCHEMA["properties"])
    assert LABEL_SCHEMA["additionalProperties"] is False


def test_build_body_wraps_complaint_and_uses_guidelines() -> None:
    body = build_body("My card was stolen", "THE RULES")

    assert body["instructions"] == "THE RULES"
    assert body["input"] == "<complaint>\nMy card was stolen\n</complaint>"
    assert body["text"]["format"]["schema"] is LABEL_SCHEMA
    assert body["text"]["format"]["strict"] is True


def test_label_rejects_value_outside_enum() -> None:
    with pytest.raises(ValidationError):
        WeakLabel.model_validate(
            {"rationale": "x", "urgency": "urgent", "sentiment": "negative"}
        )
