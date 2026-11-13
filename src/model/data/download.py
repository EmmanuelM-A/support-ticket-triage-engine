import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.configs import settings
from src.model.data.label_map import LabelMap, load_label_map

TEXT_COLUMN = "Consumer complaint narrative"
CATEGORY_COLUMN = "category"
ID_COLUMN = "Complaint ID"

_logger = logging.getLogger(__name__)


def sample_complaints(
    source_dir: Path,
    destination: Path,
    n_per_category: int = settings.ml.SAMPLES_PER_CATEGORY,
    label_map: LabelMap | None = None,
    seed: int = settings.ml.RANDOM_SEED,
    force: bool = False,
) -> Path:
    """
    Stream every CFPB export zip in `source_dir`, keep rows with a narrative,
    map each product to a category and write a uniform sample of up to
    `n_per_category` rows per category to `destination` (CSV).
    """
    if destination.exists() and not force:
        _logger.info(
            "Sample already exists at %s, skipping (use force to rebuild)",
            destination,
        )
        return destination

    sources = sorted(source_dir.glob("*.zip"))
    if not sources:
        raise FileNotFoundError(f"No .zip files found in {source_dir}")

    label_map = label_map or load_label_map()
    rng = np.random.default_rng(seed)
    kept: pd.DataFrame | None = None

    for chunk in _read_chunks(sources):
        chunk = chunk.dropna(subset=[TEXT_COLUMN])
        chunk = chunk[chunk[TEXT_COLUMN].str.strip() != ""]

        chunk = chunk.assign(**{CATEGORY_COLUMN: label_map.map_frame(chunk)})
        chunk = chunk.dropna(subset=[CATEGORY_COLUMN])

        if chunk.empty:
            continue

        chunk = chunk.assign(_r=rng.random(len(chunk)))

        combined = (
            chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        )
        kept = (
            combined.sort_values("_r")
            .drop_duplicates(subset=ID_COLUMN)
            .groupby(CATEGORY_COLUMN, sort=False)
            .head(n_per_category)
        )

    if kept is None:
        raise ValueError(f"No rows with narrative text found in {source_dir}")

    sampled = (
        kept.drop(columns="_r")
        .sort_values([CATEGORY_COLUMN, ID_COLUMN])
        .reset_index(drop=True)
    )

    _check_counts(sampled, label_map, n_per_category)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    sampled.to_csv(tmp, index=False)
    tmp.replace(destination)

    _logger.info(
        "Wrote %d rows from %d source files to %s",
        len(sampled),
        len(sources),
        destination,
    )

    return destination


def _read_chunks(sources: list[Path]) -> Iterator[pd.DataFrame]:
    for source in sources:
        _logger.info("Reading %s", source.name)
        yield from pd.read_csv(
            source,
            chunksize=settings.ml.DOWNLOAD_CHUNK_SIZE,
            usecols=settings.ml.REQUIRED_COLUMNS,
            dtype=str,
        )


def _check_counts(df: pd.DataFrame, label_map: LabelMap, n_per_category: int) -> None:
    counts = df[CATEGORY_COLUMN].value_counts()

    missing = set(label_map.categories) - set(counts.index)
    if missing:
        _logger.warning("Categories with no rows: %s", sorted(missing))

    short = counts[counts < n_per_category]
    if not short.empty:
        _logger.warning(
            "Categories with fewer than %d rows: %s", n_per_category, short.to_dict()
        )


if __name__ == "__main__":
    from src.config.logger import setup_logging

    parser = argparse.ArgumentParser(description="Sample CFPB complaints.")
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if the sample exists"
    )
    args = parser.parse_args()

    setup_logging()
    sample_complaints(
        settings.ml.SOURCE_DATA_DIR,
        settings.ml.RAW_DATA_DIR / "complaints_sample.csv",
        force=args.force,
    )
