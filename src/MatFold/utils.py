"""Utility functions for MatFold class that verify input parameters and split indices.

Further contains a minimal reproduction of the `KFold` class from scikit-learn to 
remove package dependency.

BSD 3-Clause License

Copyright (c) 2007-2024 The scikit-learn developers.
All rights reserved.

"""

import numbers
import collections
import os

import numpy as np
import pandas as pd


VALID_SPLIT_TYPES = {
    "index": "index",
    "random": "index",
    "structureid": "structureid",
    "structure": "structureid",
    "composition": "composition",
    "comp": "composition",
    "chemsys": "chemsys",
    "chemicalsystem": "chemsys",
    "sgnum": "sgnum",
    "spacegroup": "sgnum",
    "spacegroupnumber": "sgnum",
    "pointgroup": "pointgroup",
    "pg": "pointgroup",
    "pointgroupnumber": "pointgroup",
    "pointgroupsymbol": "pointgroup",
    "pgsymbol": "pointgroup",
    "crystalsys": "crystalsys",
    "crystalsystem": "crystalsys",
    "elements": "elements",
    "elems": "elements",
    "periodictablerows": "periodictablerows",
    "ptrows": "periodictablerows",
    "periodictablegroups": "periodictablegroups",
    "ptgroups": "periodictablegroups",
}


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not isinstance(x, pd.DataFrame) and hasattr(x, "__dataframe__"):
        return x.__dataframe__().num_rows()

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Args:
        *arrays: List of arrays to check.

    Raises:
        ValueError: If arrays have inconsistent lengths.

    """
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(length) for length in lengths]
        )


def _make_indexable(iterable):
    if hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)


def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Args:
        *iterables: List of objects to make indexable.

    Returns:
        List of indexable arrays.

    Raises:
        ValueError: If arrays have inconsistent lengths.

    """
    result = [_make_indexable(X) for X in iterables]
    check_consistent_length(*result)
    return result


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Args:
        seed: None, int, or RandomState instance.

    Returns:
        RandomState instance.

    Raises:
        ValueError: If seed cannot be converted to RandomState.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(int(seed))
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


class KFold:
    """K-Fold cross-validation iterator.

    Provides train/test indices to split data in k folds. Each fold is then used once as a validation
    while the k - 1 remaining folds form the training set.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        """Initialize K-Fold cross-validation.

        Args:
            n_splits: Number of folds. Must be at least 2.
            shuffle: Whether to shuffle the data before splitting.
            random_state: Controls the randomness of the fold generation.

        """
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                (
                    "Setting a random_state has no effect since shuffle is "
                    "False. You should leave "
                    "random_state to its default (None), or set shuffle=True."
                ),
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets.

        Args:
            X: Array-like of shape (n_samples, n_features).
            y: Array-like of shape (n_samples,).
            groups: Array-like of shape (n_samples,).

        Yields:
            train: The training set indices for that split.
            test: The testing set indices for that split.

        Raises:
            ValueError: If n_splits > n_samples.

        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, X=None, y=None, groups=None):
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

def _check_split_dfs(
    df: pd.DataFrame,
    df_list: list[pd.DataFrame],
    verbose: bool = True,
) -> None:
    """Check that there are no duplicates in dfs in `df_list` and that the number of indices in `df` is the same
    as in the combined dfs in `df_list`

    :param df: DataFrame containing the dataset.
    :param df_list: List of sub dataframes.
    :param verbose: Whether to print out the lengths of the `df_list` dfs.
    :return: None.
    """
    indices_list = [list(sub_df.index) for sub_df in df_list]
    sizes_list = [len(lst) for lst in indices_list]

    if verbose:
        print(
            f"Individual lengths of indices lists = {'+'.join([str(s) for s in sizes_list])} = {sum(sizes_list)}."
            f" Original total length of dataframe indices = {len(df)}",
        )

    duplicates = [
        len([item for item, count in collections.Counter(lst).items() if count > 1])
        for lst in indices_list
    ]
    if np.sum(duplicates) != 0:
        raise Exception(
            "Error: Duplicate indices detected within individual dfs: ",
            duplicates,
        )

    if sum(sizes_list) != len(df):
        raise Exception(
            f"Error: Non-equal num of indices in splits {sum(sizes_list)} vs. original {len(df)}.",
        )


def _save_split_dfs(
    df: pd.DataFrame,
    train_indices: list | np.typing.NDArray,
    test_indices: list | np.typing.NDArray,
    default_train_indices: list | np.typing.NDArray,
    default_test_indices: list | np.typing.NDArray,
    cols_to_keep: list,
    path: str | os.PathLike | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train and test dfs and saves them as csv files (if path is specified).

    Note that the path needs to end with `.csv` and the output file endings will be changed to
    `<>.train.csv` and `<>.test.csv`

    :param df: DataFrame containing the dataset.
    :param train_indices: List of train indices.
    :param test_indices: List of test indices.
    :param default_train_indices: List of indices that are part of the training set by default.
    :param default_test_indices: List of indices that are part of the test set by default.
    :param cols_to_keep: List of columns to keep in the splits.
    :param path: Path for the output files. Must end with `.csv`. If no path is specified then no files will be saved.
    :return: Tuple of train and test dataframes.
    """
    test_df = df.loc[test_indices + default_test_indices, :].copy()
    train_df = df.loc[train_indices + default_train_indices, :].copy()
    if path is not None:
        train_df.loc[:, cols_to_keep].to_csv(
            str(path).replace(".csv", ".train.csv"),
            header=True,
            index=False,
        )
        test_df.loc[:, cols_to_keep].to_csv(
            str(path).replace(".csv", ".test.csv"),
            header=True,
            index=False,
        )
    return train_df, test_df

def _validate_train_validation_test_fractions(
    train_fraction: float | None,
    validation_fraction: float | None,
    test_fraction: float | None,
) -> tuple[float, float, float]:
    """Validate train and test fractions and complete them.

    :param train_fraction: Fraction of data for training.
    :param validation_fraction: Fraction of data for validation.
    :param test_fraction: Fraction of data for testing.
    :return: Tuple of validated/completed train and test fractions.
    """
    if test_fraction is None:
        raise ValueError(
            "Error: `test_fraction` needs to be defined."
        )
    if validation_fraction is None and train_fraction is None:
        validation_fraction = 0.0
        train_fraction = 1.0 - test_fraction - validation_fraction
    elif validation_fraction is None and train_fraction is not None:
        validation_fraction = 1.0 - test_fraction - train_fraction
    assert train_fraction is not None and validation_fraction is not None and test_fraction is not None
    if (
        train_fraction <= 0.0
        or train_fraction >= 1.0
        or validation_fraction < 0.0
        or validation_fraction > 1.0
        or test_fraction <= 0.0
        or test_fraction >= 1.0
    ):
        raise ValueError(
            "Error: `train_fraction` and `test_fraction` need to be "
            "greater than 0.0 and less than 1.0. `validation_fraction` needs to be "
            "greater than or equal to 0.0 and less than or equal to 1.0.",
        )
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError(
            "Error: `train_fraction` plus `test_fraction` need to be equal to 1.0."
        )
    return train_fraction, validation_fraction, test_fraction


def _validate_inputs(
    df: pd.DataFrame | None,
    fraction_upper_limit: float | None,
    fraction_lower_limit: float | None,
    split_type: str,
    keep_n_elements_in_train: list[int] | None,
) -> None:
    """Validate input parameters for nested and LOO splitting.

    :param df: DataFrame containing the dataset.
    :param fraction_upper_limit: If a split possiblity is represented in the dataset with a fraction above
    this limit then the corresponding indices will be forced to be in the training set by default.
    :param fraction_lower_limit: If a split possiblity is represented in the dataset with a fraction below
    this limit then the corresponding indices will be forced to be in the training set by default.
    :param split_type: Defines the type of splitting, must be either "index", "structureid", "composition",
    "chemsys", "sgnum", "pointgroup", "crystalsys", "elements", "periodictablerows", or "periodictablegroups"
    :param n_inner_splits: Number of inner splits (for nested k-fold); if set to 0, then `n_inner_splits` is set
    equal to the number of inner test possiblities (i.e., each inner test set holds one possibility out
    for all possible options)
    :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
    in the test set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
    
    :return: None.
    """
    if fraction_upper_limit is not None and fraction_lower_limit is not None:
        if (
            fraction_upper_limit < 0.0
            or fraction_upper_limit > 1.0
            or fraction_lower_limit < 0.0
            or fraction_lower_limit > 1.0
        ):
            raise ValueError(
                "Error: `fraction_upper_limit` and `fraction_lower_limit` need to be "
                "greater or equal to 0.0 and less or equal to 1.0",
            )

    if split_type.replace("_", "") not in VALID_SPLIT_TYPES.keys():
        raise ValueError(
            f'Error: `split_type` must be one of the following: {VALID_SPLIT_TYPES.keys()}.',
        )

    if df is not None and keep_n_elements_in_train is not None:
        for n in keep_n_elements_in_train:
            if n not in df["nelements"].tolist():
                raise ValueError(
                    f"Error: No structure exists in the dataset that contain {n} elements. "
                    f"Adjust `keep_n_elements_in_train` accordingly.",
                )


def _check_split_indices_passed(
    train_indices: list[int],
    test_indices: list[int],
    min_train_test_factor: float | None,
) -> bool:
    """Check if train and test indices are valid.

    :param train_indices: List of train indices.
    :param test_indices: List of test indices.
    :param min_train_test_factor: Minimum factor for train/test size ratio.
    :return: True if valid, False otherwise.
    """
    if len(set(train_indices).intersection(set(test_indices))) != 0:
        raise Exception(
            f"Error: Training and test indices are not mutually exclusive "
            f"({len(set(train_indices).intersection(set()))} indices in common).",
        )
    if min_train_test_factor is not None:
        if len(train_indices) < len(test_indices) * min_train_test_factor:
            print(
                f"Warning! Train set size ({len(train_indices)}) is smaller than "
                f"test set size times min_train_test_factor ({len(test_indices)} * {min_train_test_factor} = "
                f"{int(round(len(test_indices) * min_train_test_factor, 0))}).",
                flush=True,
            )
            return False
    if len(train_indices) == 0 or len(test_indices) == 0:
        print(
            f"Warning! Either train (len={len(train_indices)}) or test "
            f"(len={len(test_indices)}) set is empty and split cannot be created.",
            flush=True,
        )
        return False
    return True
