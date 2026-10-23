"""Test file for Task 4: Mini Paper Generation"""
import os
import pytest

def test_mini_paper_exists():
    """Test that the mini paper Markdown file exists."""
    assert os.path.exists('docs/iris_nn_analysis.md'), "docs/iris_nn_analysis.md should exist"

def test_mini_paper_has_sections():
    """Test that the mini paper has all required sections."""
    with open('docs/iris_nn_analysis.md', 'r') as f:
        content = f.read()
    
    required_sections = [
        'Abstract',
        '1. Introduction',
        '2. Dataset Description',
        '3. Methodology',
        '4. Results',
        '5. Discussion',
        '6. Conclusion'
    ]
    
    for section in required_sections:
        assert section in content, f"Required section '{section}' missing from mini paper"

def test_mini_paper_has_key_results():
    """Test that the mini paper contains key experimental results."""
    with open('docs/iris_nn_analysis.md', 'r') as f:
        content = f.read()
    
    # Check for accuracy mention
    assert 'accuracy' in content.lower() or '83' in content, "Mini paper should mention accuracy results"
    
    # Check for PCA mention
    assert 'PCA' in content or 'principal' in content.lower(), "Mini paper should mention PCA visualization"
    
    # Check for neural network mention
    assert 'neural' in content.lower() or 'network' in content.lower(), "Mini paper should mention neural network"

def test_mini_paper_minimum_length():
    """Test that the mini paper has minimum content length."""
    with open('docs/iris_nn_analysis.md', 'r') as f:
        content = f.read()
    
    # Word count check (minimum 500 words for a proper mini paper)
    words = content.split()
    assert len(words) > 500, f"Mini paper too short: {len(words)} words, expected > 500"