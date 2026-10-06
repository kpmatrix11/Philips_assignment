import pandas as pd

from step_1_2_oralb_scrapper import (
    OralBProduct,
    extract_claims_and_brush_mode_count,
    save_flat_csv,
)


def test_extract_claims_and_brush_mode_count_from_long_description():
    description = (
        "Removes 100% more plaque than a manual brush. "
        "Protect your gums with the smart pressure sensor. "
        "Choose from five different smart brushing modes."
    )

    plaque_claim, gum_claims, brush_mode_count = extract_claims_and_brush_mode_count(description)

    assert plaque_claim == "Removes 100% more plaque than a manual brush."
    assert gum_claims == "Protect your gums with the smart pressure sensor."
    assert brush_mode_count == 5


def test_flat_csv_includes_long_description_extractions(tmp_path, monkeypatch):
    monkeypatch.setattr("src.oralb_scrapper.OUTPUT_DIR", tmp_path)
    product = OralBProduct(
        long_description="100% more plaque removal. Healthier gums. Includes 7 brushing modes."
    )

    save_flat_csv([product])
    output = pd.read_csv(tmp_path / "oralb_products_flat.csv")

    assert output.loc[0, "plaque_claims"] == "100% more plaque removal."
    assert output.loc[0, "gum_claims"] == "Healthier gums."
    assert output.loc[0, "brush_modes_count"] == 7