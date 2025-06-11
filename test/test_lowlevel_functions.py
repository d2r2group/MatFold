"""Test cases for low-level functions in MatFold package."""
import json
import os

import pandas as pd
import pytest

from MatFold import MatFold
from MatFold.main import _get_unique_split_possibilities, _get_split_set_indices, _get_train_test_indices

TEST_DIR = "test/"

@pytest.fixture
def load_sgnum_test_data():
    """Load sgnum test data from JSON and CSV files and set up test output directory."""
    with open(TEST_DIR + "test_sgnum.json", "r") as fp:
        cifs = json.load(fp)
    data = pd.read_csv(TEST_DIR + "test_sgnum.csv", header=0)
    # mpids = set(data["mpid"].tolist())
    # cifs_selected = {key: value for key, value in cifs.items() if key in mpids}
    # with open(TEST_DIR + "test_sgnum.json", "w") as fp:
    #     json.dump(cifs_selected, fp)
    return cifs, data

@pytest.fixture
def load_elements_test_data():
    """Load elements test data from JSON and CSV files and set up test output directory."""
    with open(TEST_DIR + "test_elements.json", "r") as fp:
        cifs = json.load(fp)
    data = pd.read_csv(TEST_DIR + "test_elements.csv", header=0)
    # mpids = set(data["mpid"].tolist())
    # cifs_selected = {key: value for key, value in cifs.items() if key in mpids}
    # with open(TEST_DIR + "test_elements.json", "w") as fp:
    #     json.dump(cifs_selected, fp)
    return cifs, data

def test_sgnum_split_indices(load_sgnum_test_data):
    """Test the _get_split_set_indices function for space group numbers test data."""
    cifs, data = load_sgnum_test_data
    mfc = MatFold(data, cifs, return_frac=1, keep_splitlabel_cols=False, always_include_n_elements=None)
    sp = _get_unique_split_possibilities(mfc.df, [], "sgnum")
    assert sp == sorted(["P-62m", "P6_3/mmc", "I4/mmm", "P-1", "P6/mmm", "Immm"])
    si = _get_split_set_indices(mfc.df, ["P-62m"], [], "sgnum")
    assert si == [5]
    si = _get_split_set_indices(mfc.df, ["P6_3/mmc"], [], "sgnum")
    assert si == [0, 1, 2, 3, 4]
    si = _get_split_set_indices(mfc.df, ["I4/mmm"], [], "sgnum")
    assert si == [6, 7, 8, 9, 10]
    si = _get_split_set_indices(mfc.df, ["P-1"], [], "sgnum")
    assert si == [12]
    si = _get_split_set_indices(mfc.df, ["P6/mmm"], [], "sgnum")
    assert si == [13, 14, 15, 16, 17, 18]
    si = _get_split_set_indices(mfc.df, ["Immm"], [], "sgnum")
    assert si == [11]
    si = _get_split_set_indices(mfc.df, ["Immm"], [11], "sgnum")
    assert si == []
    si = _get_split_set_indices(mfc.df, ["Immm", "P-1"], [], "sgnum")
    assert si == [11, 12]
    si = _get_split_set_indices(mfc.df, ["Immm", "P-1"], [12], "sgnum")
    assert si == [11]
    strain, stest = _get_train_test_indices(mfc.df, ["P-62m", "P6_3/mmc", "I4/mmm", "P-1"], ["P6/mmm", "Immm"], [], [], "sgnum")
    assert strain == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    assert stest == [11, 13, 14, 15, 16, 17, 18]
    strain, stest = _get_train_test_indices(mfc.df, ["P-62m", "P6_3/mmc", "I4/mmm", "Immm"], ["P6/mmm", "P-1"], [12], [11], "sgnum")
    assert strain == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert stest == [13, 14, 15, 16, 17, 18]
    # Check AssertionError for case where train and test sets are overlapping
    with pytest.raises(AssertionError):
        _get_train_test_indices(mfc.df, ["P-62m", "P6_3/mmc", "I4/mmm", "Immm"], ["P6/mmm", "Immm"], [], [], "sgnum")

def test_elements_split_indices(load_elements_test_data):
    """Test the _get_split_set_indices function for elements test data."""
    cifs, data = load_elements_test_data
    mfc = MatFold(data, cifs, return_frac=1, keep_splitlabel_cols=False, always_include_n_elements=None)
    sp = _get_unique_split_possibilities(mfc.df, [], "elements")
    assert sp == sorted(["Cd", "Hg", "Li", "Pt", "Sc", "Sn", "Yb", "Zn"])
    si = _get_split_set_indices(mfc.df, ["Hg"], [], "elements")
    assert si == [4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18]
    si = _get_split_set_indices(mfc.df, ["Pt"], [25], "elements")
    assert si == [19, 20, 21]
    si = _get_split_set_indices(mfc.df, ["Zn", "Pt"], [], "elements", all_must_be_in_split_set=False)
    assert si == [7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25]
    si = _get_split_set_indices(mfc.df, ["Zn", "Pt"], [], "elements", all_must_be_in_split_set=True)
    assert si == [7, 8, 9, 10, 19, 20, 21]
    si = _get_split_set_indices(mfc.df, ["Zn", "Pt"], [7], "elements", all_must_be_in_split_set=False)
    assert si == [8, 9, 10, 19, 20, 21, 22, 23, 24, 25]
    si = _get_split_set_indices(mfc.df, ["Zn", "Pt"], [7], "elements", all_must_be_in_split_set=True)
    assert si == [8, 9, 10, 19, 20, 21]
    strain, stest = _get_train_test_indices(mfc.df, ["Cd", "Hg", "Li", "Pt", "Sc", "Sn"], ["Yb", "Zn"], [], [], "elements")
    assert strain == [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18]
    assert stest == [7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25]
    strain, stest = _get_train_test_indices(mfc.df, ["Hg", "Li", "Pt", "Sc", "Sn", "Yb"], ["Zn", "Cd"], [22, 23, 24], [0, 1, 2, 3, 25], "elements")
    assert strain == [4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18]
    assert stest == [7, 8, 9, 10, 19, 20, 21]
    # Check AssertionError for case where train and test sets are overlapping
    with pytest.raises(AssertionError):
        _get_train_test_indices(mfc.df, ["Hg", "Li", "Pt", "Sc", "Sn", "Zn"], ["Zn", "Cd"], [], [], "elements")
