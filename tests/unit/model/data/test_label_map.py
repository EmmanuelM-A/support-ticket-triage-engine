import pandas as pd
import pytest
from pydantic import ValidationError

from src.model.data.label_map import LabelMap, UnknownProductError, load_label_map

# Every `Product` value in the CFPB export downloaded on 2026-11-08.
CFPB_PRODUCTS = [
    "Bank account or service",
    "Checking or savings account",
    "Consumer Loan",
    "Credit card",
    "Credit card or prepaid card",
    "Credit reporting",
    "Credit reporting or other personal consumer reports",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Debt collection",
    "Debt or credit management",
    "Money transfer, virtual currency, or money service",
    "Money transfers",
    "Mortgage",
    "Other financial service",
    "Payday loan",
    "Payday loan, title loan, or personal loan",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Student loan",
    "Vehicle loan or lease",
    "Virtual currency",
]


@pytest.fixture(scope="module")
def label_map() -> LabelMap:
    return load_label_map()


def test_every_known_product_is_mapped_or_excluded(label_map: LabelMap) -> None:
    for product in CFPB_PRODUCTS:
        category = label_map.category_for(product)

        assert category is None or category in label_map.categories


def test_only_other_financial_service_is_excluded(label_map: LabelMap) -> None:
    excluded = [p for p in CFPB_PRODUCTS if label_map.category_for(p) is None]

    assert excluded == ["Other financial service"]


def test_renamed_products_merge_into_one_category(label_map: LabelMap) -> None:
    credit_reporting_names = [
        p for p in CFPB_PRODUCTS if p.startswith("Credit reporting")
    ]

    assert {label_map.category_for(p) for p in credit_reporting_names} == {
        "credit_reporting"
    }


def test_unknown_product_raises(label_map: LabelMap) -> None:
    with pytest.raises(UnknownProductError):
        label_map.category_for("Crypto staking")


@pytest.mark.parametrize(
    ("product", "sub_product", "expected"),
    [
        ("Credit card or prepaid card", "Gift card", "prepaid_card"),
        ("Credit card or prepaid card", "Store credit card", "credit_card"),
        ("Credit card or prepaid card", "Some new card type", "credit_card"),
        ("Credit card or prepaid card", None, "credit_card"),
        ("Consumer Loan", "Vehicle lease", "vehicle_loan"),
        ("Consumer Loan", "Installment loan", "personal_loan"),
        ("Mortgage", "FHA mortgage", "mortgage"),
    ],
)
def test_sub_product_overrides(
    label_map: LabelMap, product: str, sub_product: str | None, expected: str
) -> None:
    assert label_map.category_for(product, sub_product) == expected


def test_map_frame(label_map: LabelMap) -> None:
    df = pd.DataFrame(
        {
            "Product": ["Credit reporting", "Consumer Loan", "Other financial service"],
            "Sub-product": [None, "Vehicle loan", "Check cashing"],
        },
        index=[10, 20, 30],
    )

    categories = label_map.map_frame(df)

    assert categories.to_dict() == {
        10: "credit_reporting",
        20: "vehicle_loan",
        30: None,
    }


def test_map_frame_raises_on_unknown_product(label_map: LabelMap) -> None:
    df = pd.DataFrame(
        {"Product": ["Mortgage", "Crypto staking"], "Sub-product": ["", ""]}
    )

    with pytest.raises(UnknownProductError):
        label_map.map_frame(df)


def test_mapping_to_undeclared_category_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LabelMap(version=1, categories=["mortgage"], products={"Mortgage": "home_loan"})


def test_category_without_products_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LabelMap(
            version=1,
            categories=["mortgage", "student_loan"],
            products={"Mortgage": "mortgage"},
        )


def test_product_both_mapped_and_excluded_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LabelMap(
            version=1,
            categories=["mortgage"],
            products={"Mortgage": "mortgage"},
            excluded=["Mortgage"],
        )
