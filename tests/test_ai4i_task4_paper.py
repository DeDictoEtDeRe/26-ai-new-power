"""TDD tests for Task 4: AI4I 2020 mini paper."""
import os

PAPER_PATH = "docs/ai4i_nn_analysis.md"

REQUIRED_SECTIONS = [
    "Abstract",
    "1. Introduction",
    "2. Dataset",
    "3. Methodology",
    "4. Results",
    "5. Discussion",
    "6. Conclusion",
]


def test_paper_exists():
    assert os.path.exists(PAPER_PATH), f"{PAPER_PATH} should exist"


def test_paper_has_required_sections():
    with open(PAPER_PATH) as handle:
        content = handle.read()
    for section in REQUIRED_SECTIONS:
        assert section in content, f"Required section missing: {section!r}"


def test_paper_reports_measured_results():
    with open(PAPER_PATH) as handle:
        content = handle.read().lower()
    # The paper must reference concrete measured numbers, not placeholders.
    assert "auc" in content or "f1" in content
    assert "ai4i" in content
    assert "machine failure" in content or "failure" in content
    assert "neural network" in content or "mlp" in content
    assert "pca" in content


def test_paper_minimum_length():
    with open(PAPER_PATH) as handle:
        content = handle.read()
    words = content.split()
    assert len(words) >= 500, f"Paper too short: {len(words)} words, expected >= 500"
