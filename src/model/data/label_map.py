from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, model_validator

LABEL_MAP_PATH = Path(__file__).with_name("label_map.yaml")


class UnknownProductError(ValueError):
    pass


class _SplitProduct(BaseModel):
    default: str
    sub_products: dict[str, str]


class LabelMap(BaseModel):
    version: int
    categories: list[str]
    products: dict[str, str | _SplitProduct]
    excluded: list[str] = []

    @model_validator(mode="after")
    def _check_consistency(self) -> "LabelMap":
        targets: set[str] = set()
        for target in self.products.values():
            if isinstance(target, str):
                targets.add(target)
            else:
                targets.add(target.default)
                targets.update(target.sub_products.values())

        undeclared = targets - set(self.categories)
        if undeclared:
            raise ValueError(f"Mapped to undeclared categories: {sorted(undeclared)}")

        unused = set(self.categories) - targets
        if unused:
            raise ValueError(f"Categories with no products: {sorted(unused)}")

        both = set(self.products) & set(self.excluded)
        if both:
            raise ValueError(f"Products both mapped and excluded: {sorted(both)}")

        return self

    def category_for(self, product: str, sub_product: str | None = None) -> str | None:
        """Return the category, or None for an excluded product."""
        if product in self.excluded:
            return None

        try:
            target = self.products[product]
        except KeyError:
            raise UnknownProductError(f"Unmapped CFPB product: {product!r}") from None

        if isinstance(target, str):
            return target
        return target.sub_products.get(sub_product or "", target.default)

    def map_frame(
        self,
        df: pd.DataFrame,
        product_col: str = "Product",
        sub_product_col: str = "Sub-product",
    ) -> pd.Series:
        products = df[product_col]
        sub_products = df[sub_product_col].fillna("")

        lookup = {
            (product, sub_product): self.category_for(product, sub_product)
            for product, sub_product in set(zip(products, sub_products, strict=True))
        }

        return pd.Series(
            [lookup[key] for key in zip(products, sub_products, strict=True)],
            index=df.index,
            name="category",
            dtype="object",
        )


def load_label_map(path: Path = LABEL_MAP_PATH) -> LabelMap:
    with path.open(encoding="utf-8") as f:
        return LabelMap.model_validate(yaml.safe_load(f))
