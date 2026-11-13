from pathlib import Path

import pandas as pd
import pytest

from src.config.configs import settings
from src.model.data.download import CATEGORY_COLUMN, TEXT_COLUMN, sample_complaints
from src.model.data.label_map import UnknownProductError

Row = tuple[str, str, str]


def _write_zip(path: Path, rows: list[Row], first_id: int = 0) -> Path:
    df = pd.DataFrame(
        [
            {
                "Complaint ID": str(first_id + i),
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
    df.to_csv(
        path,
        index=False,
        compression={"method": "zip", "archive_name": path.stem + ".csv"},
    )
    return path


def _counts(path: Path) -> dict[str, int]:
    counts = pd.read_csv(path)[CATEGORY_COLUMN].value_counts()
    return {str(category): int(n) for category, n in counts.items()}


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
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
    _write_zip(directory / "export_1.zip", rows)
    return directory


def test_caps_each_category_after_merging_renamed_products(
    source_dir: Path, tmp_path: Path
) -> None:
    out = sample_complaints(source_dir, tmp_path / "out" / "sample.csv", 4)

    assert _counts(out) == {"credit_reporting": 4, "mortgage": 2, "vehicle_loan": 1}


def test_drops_empty_narratives_and_excluded_products(
    source_dir: Path, tmp_path: Path
) -> None:
    out = sample_complaints(source_dir, tmp_path / "sample.csv", 100)

    df = pd.read_csv(out, keep_default_na=False)

    assert len(df) == 13
    assert "Other financial service" not in set(df["Product"])
    assert (df[TEXT_COLUMN].str.strip() != "").all()


def test_reads_every_zip_in_the_directory(source_dir: Path, tmp_path: Path) -> None:
    _write_zip(
        source_dir / "export_2.zip",
        [("Mortgage", "", f"more mortgage {i}") for i in range(3)],
        first_id=1_000,
    )

    out = sample_complaints(source_dir, tmp_path / "sample.csv", 100)

    assert _counts(out)["mortgage"] == 5


def test_ignores_files_that_are_not_zips(source_dir: Path, tmp_path: Path) -> None:
    (source_dir / "notes.txt").write_text("not data")
    pd.DataFrame({"Product": ["Mortgage"]}).to_csv(source_dir / "loose.csv")

    out = sample_complaints(source_dir, tmp_path / "sample.csv", 100)

    assert len(pd.read_csv(out)) == 13


def test_complaint_in_two_exports_is_kept_once(
    source_dir: Path, tmp_path: Path
) -> None:
    _write_zip(source_dir / "export_2.zip", [("Credit reporting", "", "old name 0")])

    out = sample_complaints(source_dir, tmp_path / "sample.csv", 100)

    df = pd.read_csv(out, dtype=str)
    assert len(df) == 13
    assert not df["Complaint ID"].duplicated().any()


def test_sample_is_reproducible_with_same_seed(
    source_dir: Path, tmp_path: Path
) -> None:
    first = sample_complaints(source_dir, tmp_path / "a.csv", 3, seed=1)
    second = sample_complaints(source_dir, tmp_path / "b.csv", 3, seed=1)

    assert first.read_text() == second.read_text()


def test_skips_when_destination_exists(source_dir: Path, tmp_path: Path) -> None:
    destination = tmp_path / "sample.csv"
    destination.write_text("existing")

    sample_complaints(source_dir, destination)

    assert destination.read_text() == "existing"


def test_force_rebuilds_existing_sample(source_dir: Path, tmp_path: Path) -> None:
    destination = tmp_path / "sample.csv"
    destination.write_text("existing")

    sample_complaints(source_dir, destination, 100, force=True)

    assert len(pd.read_csv(destination)) == 13


def test_empty_directory_raises_and_writes_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    destination = tmp_path / "sample.csv"

    with pytest.raises(FileNotFoundError):
        sample_complaints(empty, destination)

    assert not destination.exists()


def test_unknown_product_raises_and_writes_nothing(tmp_path: Path) -> None:
    _write_zip(tmp_path / "export.zip", [("Crypto staking", "", "some text")])
    destination = tmp_path / "sample.csv"

    with pytest.raises(UnknownProductError):
        sample_complaints(tmp_path, destination)

    assert not destination.exists()


def test_raises_when_no_narratives(tmp_path: Path) -> None:
    _write_zip(tmp_path / "export.zip", [("Mortgage", "", "")])
    destination = tmp_path / "sample.csv"

    with pytest.raises(ValueError):
        sample_complaints(tmp_path, destination)

    assert not destination.exists()


def test_default_source_dir_is_inside_project() -> None:
    project_root = Path(__file__).resolve().parents[4]

    assert settings.ml.SOURCE_DATA_DIR.is_relative_to(project_root)
