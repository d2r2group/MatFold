"""Test utility functions for MatFold package."""

import json
import os

import pandas as pd
import pytest

from MatFold import MatFold
from MatFold.utils import KFold, _check_split_dfs, _create_and_save_split_dfs, _validate_train_validation_test_fractions, \
    _validate_inputs, _check_split_indices_passed, collapse, VALID_SPLIT_TYPES

TEST_DIR = "test/"

# Check if output directory exists and delete it
OUTPUT_DIR = TEST_DIR + "output_utils/"
if os.path.exists(OUTPUT_DIR):
    # Remove all files and subdirectories
    for root, dirs, files in os.walk(OUTPUT_DIR, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(OUTPUT_DIR)

os.mkdir(OUTPUT_DIR)

@pytest.fixture
def load_sgnum_test_data():
    """Load sgnum test data from JSON and CSV files and set up test output directory."""
    with open(TEST_DIR + "test_sgnum.json", "r") as fp:
        cifs = json.load(fp)
    data = pd.read_csv(TEST_DIR + "test_sgnum.csv", header=0)
    return cifs, data

def test_collapse():
    """Test the collapse function by checking if it flattens a nested list correctly."""
    nested_list = [[1, 2], [3, 4], [5]]
    flattened_list = collapse(nested_list)
    assert list(flattened_list) == [1, 2, 3, 4, 5]

def test_validate_inputs(load_sgnum_test_data):
    """Test the _validate_inputs function by checking if it raises ValueError for invalid inputs."""
    cifs, data = load_sgnum_test_data
    mfc = MatFold(data, cifs, return_frac=1, keep_splitlabel_cols=False, always_include_n_elements=None)
    with pytest.raises(ValueError):
        _validate_inputs(None, None, None, "spgnum", None)
    for limit in [0.0, 1.0, -0.5, 1.5]:
        with pytest.raises(ValueError):
            _validate_inputs(mfc.df, limit, 0.5, "spgnum", [])
        with pytest.raises(ValueError):
            _validate_inputs(mfc.df, 0.5, limit, "spgnum", [])
    for split_type in VALID_SPLIT_TYPES.keys():
        _validate_inputs(mfc.df, 0.5, 0.5, split_type, None)
    _validate_inputs(mfc.df, 0.5, 0.5, "sgnum", [1])
    with pytest.raises(ValueError):
            _validate_inputs(mfc.df, 0.5, 0.5, "spgnum", [4])

def test_check_split_indices_passed():
    """Test the _check_split_indices_passed function."""
    with pytest.raises(ValueError):
        _check_split_indices_passed([0, 1, 2], [2, 3, 4], None)
    check = _check_split_indices_passed([], [2, 3, 4], None)
    assert not check
    check = _check_split_indices_passed([0, 1, 2], [], None)
    assert not check
    check = _check_split_indices_passed([], [], None)
    assert not check
    check = _check_split_indices_passed([0, 1, 2, 3], [4], 0.3)
    assert check
    check = _check_split_indices_passed([0], [1, 2, 3, 4], 0.3)
    assert not check

def test_validate_train_validation_test_fractions():
    """Test the _validate_train_validation_test_fractions function."""
    with pytest.raises(ValueError):
        _validate_train_validation_test_fractions(1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        _validate_train_validation_test_fractions(0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        _validate_train_validation_test_fractions(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        _validate_train_validation_test_fractions(0.5, 0.5, None)
    with pytest.raises(ValueError):
        _validate_train_validation_test_fractions(0.5, 0.3, 0.19)
    with pytest.raises(AssertionError):
        _validate_train_validation_test_fractions(None, 0.5, 0.1)
    for fraction in [0.0, 1.0, -0.5, 1.5, 0.5]:
        with pytest.raises(ValueError):
            _validate_train_validation_test_fractions(fraction, fraction, fraction)
        with pytest.raises(ValueError):
            _validate_train_validation_test_fractions(fraction, fraction, 0.1)
        with pytest.raises(ValueError):
            _validate_train_validation_test_fractions(fraction, 0.1, 0.1)
        with pytest.raises(ValueError):
            _validate_train_validation_test_fractions(0.1, fraction, 0.1)
        with pytest.raises(ValueError):
            _validate_train_validation_test_fractions(0.1, 0.1, fraction)
        with pytest.raises(ValueError):
            _validate_train_validation_test_fractions(0.1, fraction, fraction)
        with pytest.raises(ValueError):
            _validate_train_validation_test_fractions(fraction, 0.1, fraction)

def test_create_and_save_split_dfs(load_sgnum_test_data):
    """Test the _save_split_dfs function by checking if it saves DataFrames to CSV files."""
    cifs, data = load_sgnum_test_data
    mfc = MatFold(data, cifs, return_frac=1, keep_splitlabel_cols=False, always_include_n_elements=None)
    path = os.path.join(OUTPUT_DIR, "testdfs.csv")
    cols_to_keep = ["mpid", "sgnum_col"]
    traindf, testdf = _create_and_save_split_dfs(mfc.df, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [16, 17, 18], [2, 3], [14, 15], cols_to_keep, path)
    assert len(mfc.df.index) == len(traindf.index) + len(testdf.index)
    assert os.path.exists(str(path).replace(".csv", ".train.csv"))
    assert os.path.exists(str(path).replace(".csv", ".test.csv"))
    # Check saved files contain same information as the returned DataFrames (for cols_to_keep)
    traindf_file = pd.read_csv(str(path).replace(".csv", ".train.csv"), header=0)
    testdf_file = pd.read_csv(str(path).replace(".csv", ".test.csv"), header=0)
    assert traindf.loc[:, cols_to_keep].reset_index(drop=True).equals(traindf_file)
    assert testdf.loc[:, cols_to_keep].reset_index(drop=True).equals(testdf_file)

def test_check_split_dfs(load_sgnum_test_data):
    """Test the _check_split_dfs function by checking if it raises ValueError for invalid DataFrames."""
    cifs, data = load_sgnum_test_data
    mfc = MatFold(data, cifs, return_frac=1, keep_splitlabel_cols=False, always_include_n_elements=None)
    cols_to_keep = ["mpid", "sgnum_col"]
    # Check ValueError for case where the sum of the dfs is not equal to the original DataFrame
    with pytest.raises(ValueError):
        traindf, testdf = _create_and_save_split_dfs(mfc.df, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11], [16, 17, 18], [2, 3], [14, 15], cols_to_keep, None)
        _check_split_dfs(mfc.df, [traindf, testdf])
    # Check ValueError for case where the either DataFrame has overlapping indices in themselves
    with pytest.raises(ValueError):
        traindf, testdf = _create_and_save_split_dfs(mfc.df, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12], [16, 17, 18], [2, 3], [14, 15], cols_to_keep, None)
        _check_split_dfs(mfc.df, [traindf, testdf])
    with pytest.raises(ValueError):
        traindf, testdf = _create_and_save_split_dfs(mfc.df, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12], [16, 17, 18], [2, 3, 3], [14, 15], cols_to_keep, None)
        _check_split_dfs(mfc.df, [traindf, testdf])
    # Check that there is no overlapping splits in the subsets
    with pytest.raises(ValueError):
        traindf, testdf = _create_and_save_split_dfs(mfc.df, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16], [16, 17, 18], [2, 3], [14, 15], cols_to_keep, None)
        _check_split_dfs(mfc.df, [traindf, testdf])
    # Check valid case
    traindf, testdf = _create_and_save_split_dfs(mfc.df, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [16, 17, 18], [2, 3], [14, 15], cols_to_keep, None)
    _check_split_dfs(mfc.df, [traindf, testdf])

def test_kfold():
    """Test the KFold class by checking if it creates the correct number of splits with no overlap."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    data = pd.DataFrame({"A": range(20), "B": range(20)})
    splits = list(kf.split(data))
    assert len(splits) == 5
    for train_index, test_index in splits:
        assert len(train_index) + len(test_index) == len(data)
        assert set(train_index).isdisjoint(set(test_index))
    
