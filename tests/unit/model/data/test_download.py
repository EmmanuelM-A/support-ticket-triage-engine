from pathlib import Path

import pandas as pd
import pytest

from src.config.configs import settings
from src.model.data.download import CATEGORY_COLUMN, TEXT_COLUMN, sample_complaints
from src.model.data.label_map import UnknownProductError


def _write_source(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    df = pd.DataFrame(
        [
            {
                "Complaint ID": str(i),
                "Date received": "2024-01-01",
                "Product": product,
                "Sub-product": sub_product,
                "Issue": "Some issue",
                "Sub-issue": "",
                TEXT_COLUMN: text,
                "Company": "Some bank",
            }
            for i, (product, sub_product, text) in enumerate(rows)
        ]
    )
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def source(tmp_path: Path) -> Path:
    rows = (
        [("Credit reporting", "", f"old name {i}") for i in range(5)]
        + [
            ("Credit reporting or other personal consumer reports", "", f"new {i}")
            for i in range(5)
        ]
        + [("Mortgage", "FHA mortgage", f"mortgage {i}") for i in range(2)]
        + [("Consumer Loan", "Vehicle loan", "car loan")]
        + [("Other financial service", "", "excluded")]
        + [("Mortgage", "", ""), ("Mortgage", "", "   ")]
    )
    return _write_source(tmp_path / "complaints.csv", rows)


def test_caps_each_category_after_merging_renamed_products(
    source: Path, tmp_path: Path
) -> None:
    out = sample_complaints(source, tmp_path / "out" / "sample.csv", n_per_category=4)

    counts = pd.read_csv(out)[CATEGORY_COLUMN].value_counts().to_dict()

    assert counts == {"credit_reporting": 4, "mortgage": 2, "vehicle_loan": 1}


def test_drops_empty_narratives_and_excluded_products(
    source: Path, tmp_path: Path
) -> None:
    out = sample_complaints(source, tmp_path / "sample.csv", n_per_category=100)

    df = pd.read_csv(out, keep_default_na=False)

    assert len(df) == 13
    assert "Other financial service" not in set(df["Product"])
    assert (df[TEXT_COLUMN].str.strip() != "").all()


def test_sample_is_reproducible_with_same_seed(source: Path, tmp_path: Path) -> None:
    first = sample_complaints(source, tmp_path / "a.csv", n_per_category=3, seed=1)
    second = sample_complaints(source, tmp_path / "b.csv", n_per_category=3, seed=1)

    assert first.read_text() == second.read_text()


def test_skips_when_destination_exists(source: Path, tmp_path: Path) -> None:
    destination = tmp_path / "sample.csv"
    destination.write_text("existing")

    sample_complaints(source, destination)

    assert destination.read_text() == "existing"


def test_unknown_product_raises_and_writes_nothing(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path / "complaints.csv", [("Crypto staking", "", "some text")]
    )
    destination = tmp_path / "sample.csv"

    with pytest.raises(UnknownProductError):
        sample_complaints(source, destination)

    assert not destination.exists()


def test_raises_when_no_narratives(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "complaints.csv", [("Mortgage", "", "")])
    destination = tmp_path / "sample.csv"

    with pytest.raises(ValueError):
        sample_complaints(source, destination)

    assert not destination.exists()


def test_default_source_is_inside_project() -> None:
    project_root = Path(__file__).resolve().parents[4]

    assert settings.ml.ORIGINAL_DATA_PATH.is_relative_to(project_root)
