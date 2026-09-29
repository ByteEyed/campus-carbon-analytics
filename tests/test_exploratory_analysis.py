"""
Tests for Exploratory Carbon Analysis and Notebook
==================================================
Verifies existence, integrity, execution, and coverage of:
- notebooks/exploratory_analysis.ipynb
- reports/figures/*.png
- Required analytical areas (monthly, yearly, category, building, electricity, travel, waste, per-student)
- Epistemic requirement: distinguishing observations from assumptions and non-causal language
"""

import json
import os
from pathlib import Path

import pytest


def test_notebook_exists_and_valid_json():
    """Verify notebooks/exploratory_analysis.ipynb exists and is valid Jupyter format v4."""
    nb_path = Path("notebooks/exploratory_analysis.ipynb")
    assert nb_path.exists(), "Notebook file notebooks/exploratory_analysis.ipynb was not found."

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data.get("nbformat") == 4
    assert len(nb_data["cells"]) >= 15


def test_notebook_coverage_of_all_required_analyses():
    """
    Verify notebook includes all requested analytical themes:
    - monthly emissions
    - yearly emissions
    - category contribution
    - building contribution
    - electricity trend
    - travel trend
    - waste trend
    - per-student emissions
    """
    nb_path = Path("notebooks/exploratory_analysis.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_text = " ".join("".join(c["source"]) for c in nb_data["cells"]).lower()

    required_keywords = [
        "monthly",
        "yearly",
        "category",
        "building",
        "electricity",
        "travel",
        "waste",
        "per-student",
    ]
    for kw in required_keywords:
        assert kw in all_text, f"Notebook is missing coverage of required theme: '{kw}'"


def test_notebook_epistemic_distinctions():
    """Verify notebook explicitly distinguishes empirical observations from assumptions."""
    nb_path = Path("notebooks/exploratory_analysis.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_text = " ".join("".join(c["source"]) for c in nb_data["cells"])

    assert "Observations vs. Assumptions" in all_text
    assert "Empirical Observations" in all_text
    assert "Underlying Parameterized Assumptions" in all_text
    assert "Non-Causal" in all_text


def test_publication_charts_exist_and_non_empty():
    """Verify all 6 publication-quality figures exist and have valid PNG byte headers."""
    expected_figures = [
        "01_monthly_emissions_trend.png",
        "02_yearly_emissions_comparison.png",
        "03_category_contribution.png",
        "04_building_contribution.png",
        "05_activity_trends_seasonality.png",
        "06_per_student_emissions.png",
    ]
    figures_dir = Path("reports/figures")
    assert figures_dir.exists()

    png_header = b"\x89PNG\r\n\x1a\n"

    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Figure {fig_name} missing from reports/figures/"
        assert fig_path.stat().st_size > 5000, f"Figure {fig_name} is unexpectedly small or empty."
        with open(fig_path, "rb") as f:
            header = f.read(8)
            assert header == png_header, f"Figure {fig_name} is not a valid PNG image."


def test_notebook_code_cells_execution():
    """Verify all code cells in the notebook execute cleanly without runtime errors."""
    nb_path = Path("notebooks/exploratory_analysis.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Execute with working directory set to notebooks/
    original_cwd = os.getcwd()
    try:
        os.chdir(nb_path.parent)
        global_env = {}
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code_text = "".join(cell["source"])
                exec(code_text, global_env)
    finally:
        os.chdir(original_cwd)
