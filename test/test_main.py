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

# Check if output directory exists and delete it
OUTPUT_DIR = TEST_DIR + "output/"
if os.path.exists(OUTPUT_DIR):
    # Remove all files and subdirectories
    for root, dirs, files in os.walk(OUTPUT_DIR, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(OUTPUT_DIR)

os.mkdir(OUTPUT_DIR)

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
    return cifs, pd.read_csv(TEST_DIR + "test.csv", header=None)


def test_statistics(load_test_data):
    """Test the split_statistics method by verifying the statistics sum to 1.0 and have the expected number of crystal systems."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    stats = mfc.split_statistics("crystalsys")
    print(stats)
    assert len(stats.keys()) == 7
    assert pytest.approx(sum(stats.values()), 0.01) == 1.0

def test_split_synonyms(load_test_data):
    """Test the split_statistics method by verifying the statistics sum to 1.0 and have the expected number of crystal systems."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    stats = mfc.split_statistics("crystalsystem")
    stats2 = mfc.split_statistics("spacegroup_number")
    assert len(stats2.keys()) == 47
    assert len(stats.keys()) == 7
    with pytest.raises(ValueError):
        stats = mfc.split_statistics("non_existent_split")

def test_train_test_splits_index(load_test_data):
    """Test the create_train_test_splits method by creating splits with specific parameters."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    tdf, tdf, _, _, _ = mfc._create_train_test_splits(
        None,
        "index",
        0.7,
        0.3,
        0.05,
        [2],
        [],
        []
    )


def test_train_test_splits_structureid(load_test_data):
    """Test the create_train_test_splits method by creating splits with specific parameters."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    tdf, tdf, _, _, _ = mfc._create_train_test_splits(
        None,
        "structureid",
        0.7,
        0.3,
        0.05,
        [2],
        [],
        [],
    )


def test_train_test_splits_chemsys(load_test_data):
    """Test the create_train_test_splits method by creating splits with specific parameters."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    stats = mfc.split_statistics("chemsys")
    print(stats, flush=True)
    tdf, tdf, _, _, _ = mfc._create_train_test_splits(
        None,
        "chemsys",
        0.7,
        0.3,
        fraction_tolerance=0.05,
        keep_n_elements_in_train=[],
        default_train=["Mg-O-Ti", "Ce-Fe-O"],
        default_test=["O-Ti-Y", "Fe-O-Ti"],
    )


def test_train_test_splits_crystalsys(load_test_data):
    """Test the create_train_test_splits method by creating splits with specific parameters."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=1.0, always_include_n_elements=None)
    stats = mfc.split_statistics("crystalsys")
    print(stats, flush=True)
    tdf, tdf, _, _, _ = mfc._create_train_test_splits(
        None,
        "crystalsys",
        0.7,
        0.3,
        fraction_tolerance=0.05,
        keep_n_elements_in_train=[],
        default_train=[],
        default_test=["tetragonal"],
    )

def test_train_validation_test_splits(load_test_data):
    """Test the create_train_validation_test_splits method by creating splits with specific parameters."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=1.0, always_include_n_elements=None)
    tdf, vdf, tdf = mfc.create_train_validation_test_splits(
        "structureid",
        "crystalsys",
        0.6,
        0.2,
        0.2,
        fraction_tolerance=0.05,
        keep_n_elements_in_train=[],
        default_train=[],
        default_validation=[],
        default_test=["tetragonal"],
        output_dir=OUTPUT_DIR,
    )

def test_train_validation_test_splits_noval(load_test_data):
    """Test the create_train_validation_test_splits method by creating splits with specific parameters."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=1.0, always_include_n_elements=None)
    tdf, vdf, tdf = mfc.create_train_validation_test_splits(
        "structureid",
        "crystalsys",
        0.8,
        0.0,
        0.2,
        fraction_tolerance=0.05,
        keep_n_elements_in_train=[],
        default_train=[],
        default_validation=[],
        default_test=["tetragonal"],
        output_dir=OUTPUT_DIR,
    )


def test_create_nested_splits(load_test_data):
    """Test the create_nested_splits method by creating splits with specific parameters."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    mfc.create_nested_splits(
        "crystalsys",
        n_outer_splits=0,
        n_inner_splits=0,
        fraction_upper_limit=0.8,
        keep_n_elements_in_train=2,
        min_train_test_factor=None,
        output_dir=OUTPUT_DIR,
        verbose=True,
    )


def test_create_loo_split(load_test_data):
    """Test the create_loo_split method by creating a leave-one-out split for the 'Fe' element."""
    cifs, data = load_test_data
    mfc = MatFold(data, cifs, return_frac=0.5, always_include_n_elements=None)
    mfc.create_loo_split(
        "elements",
        "Fe",
        keep_n_elements_in_train=None,
        output_dir=OUTPUT_DIR,
        verbose=True,
    )
