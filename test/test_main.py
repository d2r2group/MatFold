"""Tests for the MatFold package's main functionality.

This module contains tests for the core functionality of MatFold, including:
- CIF file conversion
- Split statistics calculation
- Cross-validation split creation
- Leave-one-out split creation
"""

import json
import os

import pandas as pd
import pytest

from MatFold import MatFold, cifs_to_dict

TEST_DIR = "test/"


def test_cifs_to_dict():
    """Test the cifs_to_dict function by checking if it returns a dictionary with the expected number of entries."""
    bulk_dict = cifs_to_dict(TEST_DIR)
    assert isinstance(bulk_dict, dict)
    assert len(bulk_dict.keys()) == 2


@pytest.fixture
def load_test_data():
    """Load test data from JSON and CSV files and set up test output directory."""
    with open(TEST_DIR + "test.json", "r") as fp:
        cifs = json.load(fp)

    # Check if output directory exists and delete it
    output_dir = TEST_DIR + "output/"
    if os.path.exists(output_dir):
        # Remove all files and subdirectories
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(output_dir)

    os.mkdir(output_dir)
    return output_dir, cifs, pd.read_csv(TEST_DIR + "test.csv", header=None)


def test_statistics(load_test_data):
    """Test the split_statistics method by verifying the statistics sum to 1.0 and have the expected number of crystal systems."""
    _, cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    stats = mfc.split_statistics("crystalsys")
    print(stats)
    assert len(stats.keys()) == 7
    assert pytest.approx(sum(stats.values()), 0.01) == 1.0


def test_create_splits(load_test_data):
    """Test the create_splits method by creating splits with specific parameters."""
    output_dir, cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    mfc.create_splits(
        "crystalsys",
        n_outer_splits=0,
        n_inner_splits=0,
        fraction_upper_limit=0.8,
        keep_n_elements_in_train=2,
        min_train_test_factor=None,
        output_dir=output_dir,
        verbose=True,
    )


def test_create_loo_split(load_test_data):
    """Test the create_loo_split method by creating a leave-one-out split for the 'Fe' element."""
    output_dir, cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    mfc.create_loo_split(
        "elements",
        "Fe",
        keep_n_elements_in_train=None,
        output_dir=output_dir,
        verbose=True,
    )
