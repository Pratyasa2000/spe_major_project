import pytest
import numpy as np
from pwapp import handle_input, generate_probability_distribution_graph, generate_distraction_graph

# Test handle_input function
def test_handle_input():
    # Test with various inputs and ensure no errors occur
    assert handle_input("I am feeling happy") is None
    assert handle_input("I am sad") is None

# Test generate_probability_distribution_graph function
def test_generate_probability_distribution_graph():
    # Test that the function runs without errors
    assert generate_probability_distribution_graph() is None

# Test generate_distraction_graph function
def test_generate_distraction_graph():
    # Test that the function runs without errors
    assert generate_distraction_graph() is None
