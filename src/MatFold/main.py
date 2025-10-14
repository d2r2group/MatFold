"""MatFold: A package for systematic insights into materials discovery models' performance.

This module provides functionality for creating standardized chemical cross-validation protocols
and analyzing model performance across different materials categories.
"""

from __future__ import annotations
import hashlib
import inspect
import json
import os
import random
from types import FrameType
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ._version import __version__
from .utils import KFold, _check_split_dfs, _create_and_save_split_dfs, _validate_train_validation_test_fractions, \
    _validate_inputs, _check_split_indices_passed, collapse, VALID_SPLIT_TYPES

MAX_OUTLIER_FRACTION_FOR_NAIVE_SPLIT = 0.7


def cifs_to_dict(directory: str | os.PathLike) -> dict[str, dict[str, Any]]:
    """Convert a directory of cif files into a dictionary.

    With keys '<filename>' (of `<filename>.cif`) and values 'pymatgen dictionary'
    (parsed from `<filename>.cif`).

    :param directory: Directory where cif files are stored
    :return: Dictionary of cif files with keys '<filename>' (of `<filename>.cif`).
    Can be used as input `bulk_df` to `MatFold` class
    """
    output_dict = {}
    for x in os.listdir(directory):
        if x.endswith(".cif"):
            output_dict[x.split(".cif")[0]] = Structure.from_file(
                os.path.join(directory, x),
            ).as_dict()
    return output_dict


class MatFold:
    """A class for handling materials data and creating cross-validation splits.

    This class provides functionality for processing materials data and creating
    various types of cross-validation splits based on the following criteria 
    (strings may also contain underscores at any point, e.g. `chemical_system`):
    - `index` (or `random`)
    - `structureid` (or `structure`)
    - `composition` (or `comp`)
    - `chemsys` (or `chemicalsystem`)
    - `sgnum` (or `spacegroup`, `spacegroupnumber`)
    - `pointgroup` (or `pg`, `pointgroupsymbol`, `pgsymbol`)
    - `crystalsys` (or `crystalsystem`)
    - `elements` (or `elems`)
    - `periodictablerows` (or `ptrows`)
    - `periodictablegroups` (or `ptgroups`)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        bulk_dict: dict[str, dict[str, Any]],
        return_frac: float = 1.0,
        always_include_n_elements: list[int] | int | None = None,
        cols_to_keep: list | None = None,
        keep_splitlabel_cols: bool = False,
        write_data_checksums: bool = True,
        seed: int = 0,
    ) -> None:
        """MatFold class constructor.

        :param df: Pandas dataframe with the first column containing strings of either form `<structureid>` or
        `<structureid>:<structuretag>` (where <structureid> refers to a bulk ID and <structuretag> refers to
        an identifier of a derivative structure). All other columns are optional and may be retained specifying the
        `cols_to_keep` parameter described below.
        :param bulk_dict: Dictionary containing <structureid> as keys and the corresponding bulk pymatgen
        dictionary as values.
        :param return_frac: The fraction of the df dataset that is utilized during splitting.
        Must be larger than 0.0 and equal/less than 1.0 (=100%).
        :param always_include_n_elements: A list of number of elements for which the corresponding materials are
        always to be included in the dataset (for cases where `return_frac` < 1.0).
        :param cols_to_keep: List of columns to keep in the splits. If left `None`, then all columns of the
        original df are kept.
        :param keep_splitlabel_cols: Whether to keep the split label columns in the splits.
        :param write_data_checksums: Whether to write the checksums of the data in `df` and `bulk_dict` to json.
        :param seed: Seed for selecting random subset of data and splits.
        """
        frame = inspect.currentframe()
        if frame is None:
            raise RuntimeError("Could not get current frame")
        keys, _, _, values = inspect.getargvalues(frame)
        self.serialized: dict = {"MatFold": f"v{__version__}"}
        bdh = hashlib.md5()
        for b in sorted(bulk_dict.keys()):
            bdh.update(b.encode("utf-8"))
        if write_data_checksums:
            self.serialized["data_checksums"] = [
                int(
                    hashlib.md5(
                        np.array(
                            pd.util.hash_pandas_object(df, index=True).values
                        ).tobytes(),
                    ).hexdigest(),
                    16,
                ),
                int(bdh.hexdigest(), 16),
            ]
        serialized_init: dict = {}
        for key in keys:
            if key not in ["self", "df", "bulk_dict"]:
                serialized_init[key] = values[key]
        self.serialized["__init__"] = serialized_init

        self.return_frac = return_frac
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        if return_frac <= 0.0 or return_frac > 1.0:
            raise ValueError(
                "Error: `return_frac` needs to be greater than 0.0 and less or equal to 1.0",
            )

        if always_include_n_elements is None:
            always_include_n_elements = []
        elif isinstance(always_include_n_elements, int):
            always_include_n_elements = [always_include_n_elements]

        if cols_to_keep is None:
            self.cols_to_keep = list(df.columns)
        else:
            self.cols_to_keep = cols_to_keep

        self.df = df.copy()
        if len(str(self.df.iloc[0, 0]).split(":")) <= 2:
            self.df["structureid"] = [val.split(":")[0] for val in self.df.iloc[:, 0]]
        else:
            raise ValueError(
                "Error: Materials tags should either be of form "
                "`<structureid>` or `<structureid>:<structuretag>`.",
            )

        unique_structures = set(self.df["structureid"])

        for us in unique_structures:
            if us not in bulk_dict:
                raise ValueError(f"Error: Structure {us} not in `bulk_df` data.")

        structures = {
            id_: Structure.from_dict(bulk_dict[id_]) for id_ in unique_structures
        }

        space_groups = {
            id_: SpacegroupAnalyzer(structures[id_], 0.1) for id_ in unique_structures
        }

        self.df["composition"] = [
            str(structures[id_].composition.reduced_formula)
            for id_ in self.df["structureid"]
        ]

        self.df["chemsys"] = [
            str(structures[id_].composition.chemical_system)
            for id_ in self.df["structureid"]
        ]

        self.df["sgnum"] = [
            str(space_groups[id_].get_space_group_symbol())
            for id_ in self.df["structureid"]
        ]

        self.df["pointgroup"] = [
            str(space_groups[id_].get_point_group_symbol())
            for id_ in self.df["structureid"]
        ]

        self.df["crystalsys"] = [
            str(space_groups[id_].get_crystal_system())
            for id_ in self.df["structureid"]
        ]

        self.df["elements"] = [
            [str(element) for element in structures[id_].composition.get_el_amt_dict().keys()]
            for id_ in self.df["structureid"]
        ]

        self.df["nelements"] = [
            len(structures[id_].composition.get_el_amt_dict().keys())
            for id_ in self.df["structureid"]
        ]

        self.df["periodictablerows"] = [
            sorted({f"row{el.row}" for el in structures[id_].composition.elements})
            for id_ in self.df["structureid"]
        ]

        self.df["periodictablegroups"] = [
            sorted({f"group{el.group}" for el in structures[id_].composition.elements})
            for id_ in self.df["structureid"]
        ]

        if keep_splitlabel_cols:
            self.cols_to_keep.extend(["structureid", "composition", "chemsys", "sgnum",
                                      "pointgroup", "crystalsys", "elements", "nelements",
                                      "periodictablerows", "periodictablegroups"])

        if return_frac < 1.0:
            if len(always_include_n_elements) > 0:
                unique_nelements = set(
                    self.df[self.df["nelements"].isin(always_include_n_elements)][
                        "structureid"
                    ],
                )
                keep_possibilities = sorted(
                    set(
                        self.df[~self.df["nelements"].isin(always_include_n_elements)][
                            "structureid"
                        ],
                    ),
                )
                n_element_fraction = len(unique_nelements) / (
                    len(keep_possibilities) + len(unique_nelements)
                )
                if n_element_fraction < return_frac:
                    end_keep_index = int(
                        np.round(
                            (return_frac - n_element_fraction)
                            * len(keep_possibilities),
                            0,
                        ),
                    )
                    np.random.shuffle(keep_possibilities)
                    selection = list(keep_possibilities[:end_keep_index]) + list(
                        unique_nelements,
                    )
                else:
                    raise ValueError(
                        "Error: Fraction of `always_include_n_elements` portion of the dataset is "
                        "larger than `return_frac`. Either increase `return_frac` or reduce number "
                        "of n elements to be included.",
                    )
            else:
                keep_possibilities = sorted(unique_structures)
                end_keep_index = int(np.round(return_frac * len(keep_possibilities), 0))
                np.random.shuffle(keep_possibilities)
                selection = keep_possibilities[:end_keep_index]

            self.df = self.df[self.df["structureid"].isin(selection)]

        self.df["index"] = self.df.index

    @classmethod
    def from_json(
        cls,
        df: pd.DataFrame,
        bulk_dict: dict[str, dict[str, Any]],
        json_file: str | os.PathLike,
        create_splits: bool = True,
        enforce_checksums: bool = True,
        write_base_str: str | None = None,
        output_dir: str | os.PathLike | None = None,
        verbose: bool | None = None
    ) -> MatFold:
        """Reconstruct a `MatFold` class instance, along with its associated splits.

        Previously generated by the `create_nested_splits`, `create_train_validation_test_splits`, 
        or `create_loo_split` methods. The same `df` and `bulk_dict` used during the original 
        split creation must be provided to guarantee that the exact splits are regenerated.

        :param df: Pandas dataframe with the first column containing strings of either form
        `<structureid>` or `<structureid>:<structuretag>` (where <structureid> refers to a
        bulk ID and <structuretag> refers to an identifier of a derivative structure). All
        other columns are optional and may be retained specifying the `cols_to_keep` parameter
        described below.
        :param bulk_dict: Dictionary containing <structureid> as keys and the corresponding
        bulk pymatgen dictionary as values.
        :param json_file: Location of JSON file that is created when MatFold is used to
        generate splits.
        :param create_splits: Whether to create splits with the same json settings
        :param enforce_checksums: If `True`, checksums of the provided `df` and `bulk_dict` are
        compared to the checksums stored in the json_file. If they do not match, a error is raised.
        :param write_base_str: Base string for writing split files. 
        If not `None`, overwrites the value stored in the json_file.
        :param output_dir: Directory where split files are written to. 
        If not `None`, overwrites the value stored in the json_file.
        :param verbose: If `True`, prints additional information during split creation.
        If not `None`, overwrites the value stored in the json_file.
        :return: MatFold class instance
        """
        with open(json_file) as f:
            serialized_dict = json.load(f)

        serialized_dict["__init__"]["write_data_checksums"] = enforce_checksums
        mf = MatFold(df, bulk_dict, **serialized_dict["__init__"])
        if enforce_checksums:
            bdh = hashlib.md5()
            for b in sorted(bulk_dict.keys()):
                bdh.update(b.encode("utf-8"))
            df_checksum = int(
                hashlib.md5(
                    np.array(pd.util.hash_pandas_object(df, index=True).values).tobytes()
                ).hexdigest(),
                16,
            )
            bd_checksum = int(bdh.hexdigest(), 16)
            if df_checksum != serialized_dict["data_checksums"][0]:
                raise ValueError(
                    f"Error. Data in `df` might be different to the data used during "
                    f"JSON file generation.\nChecksums original/current: "
                    f"{serialized_dict['data_checksums'][0]}/{df_checksum}",
                )
            if bd_checksum != serialized_dict["data_checksums"][1]:
                raise ValueError(
                    "Error. Data in `bulk_df` might be different to the data used during "
                    "JSON file generation (only keys are part of checksum, not the "
                    "structures themselves). "
                    f"Error can be ignored (set `enforce_checksums` to False) if new data "
                    f"was added without changing the old data.\n"
                    f"Checksums original/current: {serialized_dict['data_checksums'][1]}/{bd_checksum}",
                )

        if not create_splits:
            return mf

        if "split_function_called" in serialized_dict:
            if write_base_str is not None:
                serialized_dict["split_parameters"]["write_base_str"] = write_base_str
            if output_dir is not None:
                serialized_dict["split_parameters"]["output_dir"] = output_dir
            if verbose is not None:
                serialized_dict["split_parameters"]["verbose"] = verbose
            if serialized_dict["split_function_called"] == "create_nested_splits":
                mf.create_nested_splits(**serialized_dict["split_parameters"])
            elif serialized_dict["split_function_called"] == "create_loo_split":
                mf.create_loo_split(**serialized_dict["split_parameters"])
            elif serialized_dict["split_function_called"] == "create_train_validation_test_splits":
                mf.create_train_validation_test_splits(**serialized_dict["split_parameters"])
            else:
                raise ValueError(
                    f"Function {serialized_dict['split_function_called']} does not exist in MatFold.",
                )
        return mf

    def split_statistics(self, split_type: str) -> dict[str, float]:
        """Analyze statistics for splits of `split_type`.

        :param split_type: String specifying the splitting type.
        :return: Dictionary with keys of unique split labels and the corresponding fraction of
        this key being represented in the entire dataset.
        """
        _validate_inputs(None, None, None, split_type, None)
        split_type = VALID_SPLIT_TYPES[split_type.replace("_", "")]
        statistics = dict.fromkeys((str(key) for key in set(collapse(self.df[split_type]))), 0.0)
        for uk in statistics.keys():
            n = 0
            for s in self.df[split_type]:
                if split_type in [
                    "elements",
                    "periodictablerows",
                    "periodictablegroups",
                ]:
                    for e in s:
                        if e == uk:
                            n += 1
                            break
                elif s == uk:
                    n += 1
            statistics[uk] = n / len(self.df[split_type])
        return statistics

    def create_train_validation_test_splits(
        self,
        split_type_validation: str,
        split_type_test: str,
        train_fraction: float | None = None,
        validation_fraction: float | None = None,
        test_fraction: float | None = None,
        fraction_tolerance: float = 0.05,
        keep_n_elements_in_train: list[int] | int | None = None,
        default_train: list[str] | None = None,
        default_validation: list[str] | None = None,
        default_test: list[str] | None = None,
        n_validation_min: int = 1,
        n_test_min: int = 1,
        write_base_str: str = "mf",
        output_dir: str | os.PathLike | None = None,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
        """Create train, validation, and test splits based on two specified split types.

        :param split_type_validation: String specifying the splitting type for the validation set.
        :param split_type_test: String specifying the splitting type for the test set.
        :param train_fraction: Fraction of the dataset to be used for training.
        :param validation_fraction: Fraction of the dataset to be used for validation.
        :param test_fraction: Fraction of the dataset to be used for testing.
        :param fraction_tolerance: Tolerance for the fraction of data in each split.
        :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
        in the train set by default. For example, '2' will keep all binaries in the training set.
        :param default_train: List of split labels which are put in the train set by default.
        :param default_validation: List of split labels which are put in the validation set by default.
        :param default_test: List of split labels which are put in the test set by default.
        :param n_validation_min: Minimum number of labels in the validation set.
        :param n_test_min: Minimum number of labels in the test set.
        :param write_base_str: Beginning string of generated file names of the written splits.
        :param output_dir: Directory where the files are written to.
        :param verbose: Whether to print detailed information during execution.
        :return: Tuple containing train, validation, and test DataFrames. 
        Validation DataFrame may be None if `validation_fraction` is 0.
        """
        if output_dir is None:
            output_dir = os.getcwd()
        if default_train is None:
            default_train = []
        if default_validation is None:
            default_validation = []
        if default_test is None:
            default_test = []
        
        if keep_n_elements_in_train is None:
            keep_n_elements_in_train = []
        elif isinstance(keep_n_elements_in_train, int):
            keep_n_elements_in_train = [keep_n_elements_in_train]

        for st in [split_type_validation, split_type_test]:
            _validate_inputs(
                self.df,
                None,
                None,
                st,
                keep_n_elements_in_train,
            )
        split_type_validation = VALID_SPLIT_TYPES[split_type_validation.replace("_", "")]
        split_type_test = VALID_SPLIT_TYPES[split_type_test.replace("_", "")]
        
        trainf, valf, testf = _validate_train_validation_test_fractions(
            train_fraction, validation_fraction, test_fraction
        )

        path = os.path.join(output_dir, write_base_str + 
                            f".{split_type_validation}-val_{split_type_test}-test_{trainf}-{valf}-{testf}.csv")
        frame: FrameType | None = inspect.currentframe()
        self._save_serialized(
            frame, str(path).replace(".csv", ".json")
        )

        trainval_df, test_df, trainval_labels, test_labels, naive_random_split_trainval_test = \
            self._create_train_test_splits(
                None,
                split_type_test,
                trainf + valf,
                testf,
                fraction_tolerance=fraction_tolerance,
                keep_n_elements_in_train=keep_n_elements_in_train,
                default_train=default_train + default_validation,
                default_test=default_test,
                n_test_min=n_test_min,
                verbose=verbose,
            )

        if np.isclose(valf, 0.0):
            _check_split_dfs(self.df, [trainval_df, test_df], verbose=verbose)
            train_df = trainval_df.copy()
            val_df = None
            naive_random_split_train_val = None
            val_labels = []
        else:
            train_df, val_df, train_labels, val_labels, naive_random_split_train_val = \
                self._create_train_test_splits(
                    trainval_df,
                    split_type_validation,
                    trainf / (trainf + valf),
                    valf / (trainf + valf),
                    fraction_tolerance=fraction_tolerance,
                    keep_n_elements_in_train=keep_n_elements_in_train,
                    default_train=default_train,
                    default_test=default_validation,
                    n_test_min=n_validation_min,
                    verbose=verbose,
                )

            _check_split_dfs(self.df, [train_df, val_df, test_df], verbose=verbose)
        summary = {
            "create_train_validation_test_splits": {
                "split_type_validation": split_type_validation,
                "split_type_test": split_type_test,
                "train_fraction": trainf,
                "validation_fraction": valf,
                "test_fraction": testf,
                "fraction_tolerance": fraction_tolerance,
                "keep_n_elements_in_train": keep_n_elements_in_train,
                "default_train": default_train,
                "default_validation": default_validation,
                "default_test": default_test,
                "n_validation_min": n_validation_min,
                "n_test_min": n_test_min,
            },
            "train_fraction_actual": len(train_df) / len(self.df),
            "validation_fraction_actual": len(val_df) / len(self.df) if val_df is not None else 0.0,
            "test_fraction_actual": len(test_df) / len(self.df),
            "n_train": len(train_df),
            "n_validation": len(val_df) if val_df is not None else 0,
            "n_test": len(test_df),
            "naive_random_split_trainval_test": naive_random_split_trainval_test,
            "naive_random_split_train_val": naive_random_split_train_val,
            "n_train_set_labels": len(train_labels),
            "n_validation_set_labels": len(val_labels),
            "n_trainvalidation_set_labels": len(trainval_labels),
            "n_test_set_labels": len(test_labels),
            "train_set_labels": train_labels,
            "validation_set_labels": val_labels,
            "trainvalidation_set_labels": trainval_labels,
            "test_set_labels": test_labels,
        }

        with open(str(path).replace(".csv", ".summary.json"), "w") as summary_file:
            json.dump(summary, summary_file, indent=4)

        train_df = train_df.loc[:, self.cols_to_keep]
        train_df.to_csv(
            str(path).replace(".csv", ".train.csv"),
            header=True,
            index=False,
        )
        if val_df is not None:
            val_df = val_df.loc[:, self.cols_to_keep]
            val_df.to_csv(
                str(path).replace(".csv", ".validation.csv"),
                header=True,
                index=False,
            )
        test_df = test_df.loc[:, self.cols_to_keep]
        test_df.to_csv(
            str(path).replace(".csv", ".test.csv"),
            header=True,
            index=False,
        )
        return train_df, val_df, test_df
        
    def _create_train_test_splits(
        self,
        df: pd.DataFrame | None,
        split_type: str,
        train_fraction: float,
        test_fraction: float,
        fraction_tolerance: float,
        keep_n_elements_in_train: list[int],
        default_train: list[str],
        default_test: list[str],
        n_test_min: int = 1,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], bool]:
        """Create train and test splits based on the specified split type.
        
        :param df: DataFrame to split. If None, then self.df is used.
        :param split_type: String specifying the splitting type.
        :param train_fraction: Fraction of the dataset to use for training.
        :param test_fraction: Fraction of the dataset to use for testing.
        :param fraction_tolerance: Tolerance for the fraction of data in each split.
        :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
        in the train set by default. For example, '2' will keep all binaries in the training set.
        :param default_train: List of split labels which are put in the train set by default.
        :param default_test: List of split labels which are put in the test set by default.
        :param n_test_min: Minimum number of labels in the test set.
        :param verbose: Whether to print detailed information during execution.
        :return: Tuple containing train and test DataFrames, bool containing whether naive random split 
        was utilized, train and test set labels.
        """
        if df is None:
            df = self.df.copy()

        default_train_indices_nelements = (
            list(df[df["nelements"].isin(keep_n_elements_in_train)].index)
            if len(keep_n_elements_in_train) > 0
            else []
        )

        default_train_indices_splits = _get_split_set_indices(df, default_train, [], split_type, all_must_be_in_split_set=True)
        default_train_indices = list(
            set(default_train_indices_nelements + default_train_indices_splits)
        )

        default_test_indices = _get_split_set_indices(df, default_test, [], split_type, all_must_be_in_split_set=False)

        split_possibilities_all = _get_unique_split_possibilities(
            df,
            keep_n_elements_in_train,
            split_type,
        )

        split_possibilities = [
            sp
            for sp in split_possibilities_all
            if sp not in default_train + default_test
        ]

        default_train_fraction = len(default_train_indices) / len(df.index)
        default_test_fraction = len(default_test_indices) / len(df.index)

        if default_train_fraction > train_fraction:
            raise ValueError(
                f"Error. Default training set fraction ({default_train_fraction}) based on specified"
                f"`keep_n_elements_in_train` and `default_train` is larger than requested training set size ({train_fraction})."
            )

        if default_test_fraction > test_fraction:
            raise ValueError(
                f"Error. Default test set fraction ({default_test_fraction}) based on specified"
                f"`default_test` is larger than requested training set size ({test_fraction})."
            )

        naive_random_split = False
        if split_type == "index":
            naive_random_split = True
        else:
            stats = self.split_statistics(split_type)
            average_fraction = sum([stats[sp] for sp in split_possibilities]) / len(split_possibilities)
            outliers = []
            if verbose:
                print(f"{average_fraction=}", flush=True)
            # Split possibilities that have a larger fraction than the average (considering `fraction_tolerance`) are considered "outliers" 
            # to determine whether to split naively (randomly, assuming every split label contributes equally to the test/training sets) or not. 
            # The outliers are added to the training set by default.
            for sp in split_possibilities:
                frac = stats[sp]
                if abs(frac - average_fraction) > fraction_tolerance:
                    outliers.append(sp)
            outlier_indices = _get_split_set_indices(df, outliers, [], split_type)
            outlier_total_fraction = len(set(default_train_indices + outlier_indices)) / len(df.index)
            if verbose:
                print(f"{outliers=}\n {outlier_total_fraction=}", flush=True)
            # If the fraction of outliers and default training is less than `MAX_OUTLIER_FRACTION_FOR_NAIVE_SPLIT` times `train_fraction`, 
            # then do naive random splitting. This is a somewhat arbitrary threshold.
            if outlier_total_fraction < MAX_OUTLIER_FRACTION_FOR_NAIVE_SPLIT * train_fraction:
                default_train_indices = list(set(default_train_indices + outlier_indices))
                default_train_fraction = len(default_train_indices) / len(df.index)
                split_possibilities = [
                    sp for sp in split_possibilities.copy() if sp not in outliers
                ]
                naive_random_split = True

        if naive_random_split:
            test_size = int(
                len(split_possibilities)
                * (test_fraction - default_test_fraction)
                / (train_fraction - default_train_fraction + test_fraction - default_test_fraction)
            )
            test_set = random.sample(split_possibilities, test_size)
            train_set = [item for item in split_possibilities if item not in test_set]
        else:
            found = False
            # Go through all combinations of split possibilities to find a combination that matches the requested train/test split fractions
            # within the tolerance. For large number of split possibilities, this can be slow. This will return the first combination 
            # that matches the requested split fractions and will not search for the most optimal one (instead, user can decrease `fraction_tolerance`).
            random.shuffle(split_possibilities)  # to ensure that seed affects the search
            for r in range(n_test_min, len(split_possibilities) + 1):
                for indices in combinations(range(len(split_possibilities)), r):
                    test_set = [split_possibilities[i] for i in indices]
                    sp_test_indices = _get_split_set_indices(df, test_set, [], split_type)
                    current_test_fraction = len(set(sp_test_indices + default_test_indices)) / len(df.index)
                    if abs(current_test_fraction - test_fraction) <= fraction_tolerance:
                        train_set = [
                            item for item in split_possibilities if item not in test_set
                        ]
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(
                    f"No combination of split_possibilities could be found such that the final training set fraction "
                    f"of {train_fraction} is reached within tolerance {fraction_tolerance} "
                    f"and minimum test set label number {n_test_min}."
                )

        train_indices, test_indices = _get_train_test_indices(
            df,
            train_set,
            test_set,
            default_train_indices,
            default_test_indices,
            split_type,
        )

        if verbose:
            print(
                f"_create_train_test_splits: "
                f"{split_type=}, {train_fraction=}, {test_fraction=}, {naive_random_split=}\n"
                f"{len(default_train_indices)=} ({len(default_train_indices)/len(df)=:.3f}),\n"
                f"{len(default_test_indices)=} ({len(default_test_indices)/len(df)=:.3f}),\n"
                f"{len(train_indices)=} ({len(train_indices)/len(df)=:.3f}),\n"
                f"{len(test_indices)=} ({len(test_indices)/len(df)=:.3f})",
                flush=True,
            )
        if (
            len(train_indices + default_train_indices) == 0
            or len(test_indices + default_test_indices) == 0
        ):
            raise Exception(
                f"Error. Either train (len={len(train_indices + default_train_indices)}) or test "
                f"(len={len(test_indices + default_test_indices)}) set is empty and split cannot be created.",
            )

        train_df, test_df = _create_and_save_split_dfs(
            df,
            train_indices,
            test_indices,
            default_train_indices,
            default_test_indices,
            self.cols_to_keep,
            None,
        )
        _check_split_dfs(df, [train_df, test_df], verbose=verbose)
        if verbose:
            print(f"{len(train_df)/len(df)=:.3f}, {len(test_df)/len(df)=:.3f}")
        return train_df, test_df, train_set, test_set, naive_random_split

    def create_nested_splits(
        self,
        split_type: str,
        n_inner_splits: int = 10,
        n_outer_splits: int = 10,
        fraction_upper_limit: float = 1.0,
        fraction_lower_limit: float = 0.0,
        keep_n_elements_in_train: list[int] | int | None = None,
        min_train_test_factor: float | None = None,
        inner_equals_outer_split_strategy: bool = True,
        write_base_str: str = "mf",
        output_dir: str | os.PathLike | None = None,
        verbose: bool = False,
    ) -> None:
        """Create splits based on `split_type`, `n_inner_splits`, `n_outer_splits` among other specifications.

        The splits are saved in `output_dir` as csv files named
        `<write_base_str>.<split_type>.k<i>_outer.<train/test>.csv` and
        `<write_base_str>.<split_type>.k<i>_outer.l<j>_inner.<train/test>.csv` for all outer (index `<i>`) and inner
        splits (index `<j>`), respectively. Additionally, a summary of the created splits is saved as
        `<write_base_str>.<split_type>.summary.k<n_outer_splits>.l<n_inner_splits>.<self.return_frac>.csv`.
        Lastly, a JSON file is saved that stores all relevant class and function variables to recreate the splits
        utilizing the class function `from_json` and is named `<write_base_str>.<split_type>.json`.

        :param split_type: String specifying the splitting type.
        :param n_inner_splits: Number of inner splits (for nested k-fold); if set to 0, then `n_inner_splits` is set
        equal to the number of inner test possiblities (i.e., each inner test set holds one possibility out
        for all possible options).
        :param n_outer_splits: Number of outer splits (k-fold); if set to 0, then `n_outer_splits` is set equal to the
        number of test possiblities (i.e., each outer test set holds one possibility out for all possible options).
        :param fraction_upper_limit: If a split possiblity is represented in the dataset with a fraction above
        this limit then the corresponding indices will be forced to be in the training set by default.
        :param fraction_lower_limit: If a split possiblity is represented in the dataset with a fraction below
        this limit then the corresponding indices will be forced to be in the training set by default.
        :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
        in the train set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
        :param min_train_test_factor: Minimum factor that the training set needs to be
        larger (for factors greater than 1.0) than the test set.
        :param inner_equals_outer_split_strategy: If true, then the inner splitting strategy used is equal to
        the outer splitting strategy, if false, then inner splitting strategy is random (by index).
        :param write_base_str: Beginning string of csv file names of the written splits
        :param output_dir: Directory where the splits are written to.
        :param verbose: Whether to print out details during code execution.
        :return: None
        """
        if output_dir is None:
            output_dir = os.getcwd()

        if keep_n_elements_in_train is None:
            keep_n_elements_in_train = []
        elif isinstance(keep_n_elements_in_train, int):
            keep_n_elements_in_train = [keep_n_elements_in_train]

        _validate_inputs(
            self.df,
            fraction_upper_limit,
            fraction_lower_limit,
            split_type,
            keep_n_elements_in_train,
        )
        split_type = VALID_SPLIT_TYPES[split_type.replace("_", "")]

        frame: FrameType | None = inspect.currentframe()
        self._save_serialized(
            frame, os.path.join(output_dir, write_base_str + f".{split_type}.json")
        )

        default_train_indices = (
            list(self.df[self.df["nelements"].isin(keep_n_elements_in_train)].index)
            if len(keep_n_elements_in_train) > 0
            else []
        )
        split_possibilities = _get_unique_split_possibilities(
            self.df,
            keep_n_elements_in_train,
            split_type,
        )

        # Remove splits from test set that have larger fractions than `max_fraction_testset`
        # then add their indices to `default_train_indices`
        remove_from_test_dict = {
            k: round(v, 3)
            for k, v in self.split_statistics(split_type).items()
            if v > fraction_upper_limit or v < fraction_lower_limit
        }
        remove_from_test = list(remove_from_test_dict.keys())
        if verbose:
            print(
                f"The following instances will be removed from possible test sets, as their fraction in the dataset "
                f"was higher than {fraction_upper_limit} or "
                f"lower than {fraction_lower_limit}: {remove_from_test_dict}.",
            )
        add_train_indices = []
        for r in set(remove_from_test):
            split_possibilities.remove(r)
            add_train_indices.extend(list(self.df[self.df[split_type] == r].index))
        if split_type in ["elements", "periodictablerows", "periodictablegroups"]:
            default_train_elements = remove_from_test.copy()
        else:
            default_train_indices.extend(add_train_indices)
            default_train_indices = list(set(default_train_indices))
            default_train_elements = []

        if n_outer_splits == 0:
            n_outer_splits = len(split_possibilities)

        if len(split_possibilities) < n_outer_splits:
            raise ValueError(
                f"Error: `n_outer_splits`, {n_outer_splits}, is larger than available "
                f"`split_possibilities`, {len(split_possibilities)} "
                f"for splitting strategy {split_type} and `fraction_lower_limit` / `fraction_upper_limit` "
                f"of {fraction_lower_limit} / {fraction_upper_limit}.",
            )

        if verbose:
            if split_type in ["elements", "periodictablerows", "periodictablegroups"]:
                print(
                    f"Default train {split_type} ({len(default_train_elements)}): {default_train_elements}",
                )
            print(
                f"Default train indices ({len(default_train_indices)}): ",
                default_train_indices,
            )
            print(f"Possible test examples: {split_possibilities}")
        if n_outer_splits > 1:
            kf_outer = KFold(
                n_splits=n_outer_splits,
                random_state=self.seed,
                shuffle=True,
            )
        else:
            raise ValueError("Error: `n_outer_splits` needs to be greater than 1.")

        summary_outer_splits = pd.DataFrame(
            columns=["n", "l", "train", "test", "ntrain", "ntest", "comment"],
        )

        # Splits for outer loop
        for i, (
            outer_train_possibility_indices,
            outer_test_possibility_indices,
        ) in enumerate(kf_outer.split(split_possibilities)):
            # Outer train structure ids
            outer_train_set = sorted(
                set(
                    np.take(
                        split_possibilities,
                        outer_train_possibility_indices,
                    ).tolist()
                    + default_train_elements,
                ),
            )

            # Outer test structure ids
            outer_test_set = list(
                np.take(split_possibilities, outer_test_possibility_indices),
            )

            if verbose:
                print(
                    f"\nSplitting: k{i}, {'-'.join(sorted(outer_test_set)) if split_type == 'elements' else ''}",
                )
                print(outer_train_set)
                print(outer_test_set)

            outer_train_indices, outer_test_indices = _get_train_test_indices(
                self.df,
                outer_train_set,
                outer_test_set,
                default_train_indices,
                [],
                split_type,
            )

            if not _check_split_indices_passed(
                outer_train_indices + default_train_indices,
                outer_test_indices,
                min_train_test_factor,
            ):
                summary_outer_splits.loc[len(summary_outer_splits.index) + 1, :] = [
                    i,
                    -1,
                    outer_train_set,
                    outer_test_set,
                    len(outer_train_indices) + len(default_train_indices),
                    len(outer_test_indices),
                    f"split check failed - factor = {min_train_test_factor}",
                ]
                continue

            path_outer = os.path.join(
                output_dir,
                write_base_str + f".{split_type}.k{i}_outer.csv",
            )

            outer_train_df, outer_test_df = _create_and_save_split_dfs(
                self.df,
                outer_train_indices,
                outer_test_indices,
                default_train_indices,
                [],
                self.cols_to_keep,
                path_outer,
            )
            _check_split_dfs(self.df, [outer_train_df, outer_test_df], verbose=verbose)

            summary_outer_splits.loc[len(summary_outer_splits.index) + 1, :] = [
                i,
                -1,
                outer_train_set,
                outer_test_set,
                len(outer_train_df),
                len(outer_test_df),
                "split successful",
            ]

            # Splits for inner loop
            inner_split_possibilities = [
                split
                for split in outer_train_set
                if split not in default_train_elements
            ]
            if n_inner_splits == 0:
                n_inner_splits = len(inner_split_possibilities)
            if n_inner_splits > 1:
                kf_inner = KFold(
                    n_splits=n_inner_splits,
                    random_state=self.seed,
                    shuffle=True,
                )
            else:
                kf_inner = None

            if kf_inner is not None and inner_equals_outer_split_strategy:
                if len(inner_split_possibilities) < n_inner_splits:
                    raise ValueError(
                        f"Error: `n_inner_splits`, {n_inner_splits}, is larger than available "
                        f"`split_possibilities` of the inner train set, {len(inner_split_possibilities)} "
                        f"for splitting strategy {split_type}, `max_fraction_testset` "
                        f"cutoff of {fraction_upper_limit}, and `n_outer_splits` of {n_outer_splits}.",
                    )
                for j, (
                    inner_train_possibility_indices,
                    inner_test_possibility_indices,
                ) in enumerate(kf_inner.split(list(inner_split_possibilities))):
                    # Inner train structure ids
                    inner_train_set = sorted(
                        set(
                            np.take(
                                list(inner_split_possibilities),
                                inner_train_possibility_indices,
                            ).tolist()
                            + default_train_elements,
                        ),
                    )

                    # Inner test structure ids
                    inner_test_set = list(
                        np.take(
                            list(inner_split_possibilities),
                            inner_test_possibility_indices,
                        ),
                    )

                    if verbose:
                        print(
                            f"\nSplitting: k{i}, i{j}, "
                            f"{'-'.join(sorted(inner_test_set)) if split_type == 'elements' else ''}",
                        )
                        print(inner_train_set)
                        print(inner_test_set)

                    inner_train_indices, inner_test_indices = _get_train_test_indices(
                        self.df,
                        inner_train_set,
                        inner_test_set,
                        default_train_indices,
                        [],
                        split_type,
                    )
                    # Ensure that no outer_test_indices are present in inner_test_indices
                    inner_test_indices = [
                        ind
                        for ind in inner_test_indices
                        if ind not in outer_test_indices
                    ]

                    if not _check_split_indices_passed(
                        inner_train_indices + default_train_indices,
                        inner_test_indices,
                        min_train_test_factor,
                    ):
                        summary_outer_splits.loc[
                            len(summary_outer_splits.index) + 1,
                            :,
                        ] = [
                            i,
                            j,
                            inner_train_set,
                            inner_test_set,
                            len(inner_train_indices) + len(default_train_indices),
                            len(inner_test_indices),
                            f"split check failed - factor = {min_train_test_factor}",
                        ]
                        continue

                    path_inner = os.path.join(
                        output_dir,
                        write_base_str + f".{split_type}.k{i}_outer.l{j}_inner.csv",
                    )

                    inner_train_df, inner_test_df = _create_and_save_split_dfs(
                        self.df,
                        inner_train_indices,
                        inner_test_indices,
                        default_train_indices,
                        [],
                        self.cols_to_keep,
                        path_inner,
                    )
                    _check_split_dfs(
                        self.df,
                        [outer_test_df, inner_train_df, inner_test_df],
                        verbose=verbose,
                    )
                    summary_outer_splits.loc[len(summary_outer_splits.index) + 1, :] = [
                        i,
                        j,
                        inner_train_set,
                        inner_test_set,
                        len(inner_train_indices) + len(default_train_indices),
                        len(inner_test_indices),
                        "split successful",
                    ]

            elif kf_inner is not None:
                for j, (train_inner_index_index, test_inner_index_index) in enumerate(
                    kf_inner.split(outer_train_indices),
                ):
                    if verbose:
                        print(f"Splitting inner {j!s}")
                    train_inner_index = np.take(
                        outer_train_indices,
                        train_inner_index_index,
                    )
                    test_inner_index = np.take(
                        outer_train_indices,
                        test_inner_index_index,
                    )

                    final_inner_train_indices = default_train_indices + list(
                        train_inner_index,
                    )
                    final_inner_test_indices = test_inner_index.copy()

                    path_inner = os.path.join(
                        output_dir,
                        write_base_str + f".{split_type}.k{i}_outer.l{j}_inner.csv",
                    )

                    inner_train_df, inner_test_df = _create_and_save_split_dfs(
                        self.df,
                        final_inner_train_indices,
                        final_inner_test_indices,
                        [],
                        [],
                        self.cols_to_keep,
                        path_inner,
                    )
                    _check_split_dfs(
                        self.df,
                        [outer_test_df, inner_train_df, inner_test_df],
                        verbose=verbose,
                    )

        summary_outer_splits.to_csv(
            os.path.join(
                output_dir,
                write_base_str
                + f".{split_type}.summary.k{n_outer_splits}.l{n_inner_splits}."
                f"{self.return_frac}.csv",
            ),
        )

    def create_splits(
        self,
        split_type: str,
        n_inner_splits: int = 10,
        n_outer_splits: int = 10,
        fraction_upper_limit: float = 1.0,
        fraction_lower_limit: float = 0.0,
        keep_n_elements_in_train: list | int | None = None,
        min_train_test_factor: float | None = None,
        inner_equals_outer_split_strategy: bool = True,
        write_base_str: str = "mf",
        output_dir: str | os.PathLike | None = None,
        verbose: bool = False,
    ) -> None:
        """Deprecated. Use `create_nested_splits` instead."""
        import warnings

        warnings.warn(
            "`create_splits` is deprecated and will be removed in a future version. "
            "Use `create_nested_splits` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_nested_splits(
            split_type=split_type,
            n_inner_splits=n_inner_splits,
            n_outer_splits=n_outer_splits,
            fraction_upper_limit=fraction_upper_limit,
            fraction_lower_limit=fraction_lower_limit,
            keep_n_elements_in_train=keep_n_elements_in_train,
            min_train_test_factor=min_train_test_factor,
            inner_equals_outer_split_strategy=inner_equals_outer_split_strategy,
            write_base_str=write_base_str,
            output_dir=output_dir,
            verbose=verbose,
        )

    def create_loo_split(
        self,
        split_type: str,
        loo_label: str,
        keep_n_elements_in_train: list[int] | int | None = None,
        write_base_str: str = "mf",
        output_dir: str | os.PathLike | None = None,
        verbose: bool = False,
    ) -> None:
        """Create leave-one-out split based on `split_type`, specified `loo_label` and `keep_n_elements_in_train`.

        The splits are saved in `output_dir` as csv files named
        `<write_base_str>.<split_type>.loo.<loo_label>.<train/test>.csv`. Additionally, a summary of the created split
        is saved as `<write_base_str>.<split_type>.summary.loo.<loo_label>.<self.return_frac>.csv`.
        Lastly, a JSON file is saved that stores all relevant class and function variables to recreate the splits
        utilizing the class function `from_json` and is named `<write_base_str>.<split_type>.loo.<loo_label>.json`.

        :param split_type: String specifying the splitting type.
        :param loo_label: Label specifying which single option is to be left out (i.e., constitute the test set).
        This label must be a valid option for the specified `split_type`.
        :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
        in the test set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
        :param write_base_str: Beginning string of csv file names of the written splits
        :param output_dir: Directory where the splits are written to.
        :param verbose: Whether to print out details during code execution.
        :return: None
        """
        if output_dir is None:
            output_dir = os.getcwd()

        if keep_n_elements_in_train is None:
            keep_n_elements_in_train = []
        elif isinstance(keep_n_elements_in_train, int):
            keep_n_elements_in_train = [keep_n_elements_in_train]

        _validate_inputs(self.df, None, None, split_type, keep_n_elements_in_train)
        split_type = VALID_SPLIT_TYPES[split_type.replace("_", "")]

        frame: FrameType | None = inspect.currentframe()
        self._save_serialized(
            frame,
            os.path.join(
                output_dir, write_base_str + f".{split_type}.loo.{loo_label}.json"
            ),
        )

        default_train_indices = (
            list(self.df[self.df["nelements"].isin(keep_n_elements_in_train)].index)
            if len(keep_n_elements_in_train) > 0
            else []
        )
        split_possibilities = _get_unique_split_possibilities(
            self.df,
            keep_n_elements_in_train,
            split_type,
        )

        if loo_label not in split_possibilities:
            raise ValueError(
                f"Error. LOO label ({loo_label}) is not in `split_possibilities` of type {split_type}.",
            )

        summary_loo_split = pd.DataFrame(
            columns=["n", "l", "train", "test", "ntrain", "ntest", "comment"],
        )

        train_set = [sp for sp in split_possibilities if sp != loo_label]
        test_set = [loo_label]

        train_indices, test_indices = _get_train_test_indices(
            self.df,
            train_set,
            test_set,
            default_train_indices,
            [],
            split_type,
        )

        if len(train_indices + default_train_indices) == 0 or len(test_indices) == 0:
            raise Exception(
                f"Error. Either train (len={len(train_indices + default_train_indices)}) or test "
                f"(len={len(test_indices)}) set is empty and split cannot be created.",
            )

        path_outer = os.path.join(
            output_dir,
            write_base_str + f".{split_type}.loo.{loo_label.replace('/', '_')}.csv",
        )
        train_df, test_df = _create_and_save_split_dfs(
            self.df,
            train_indices,
            test_indices,
            default_train_indices,
            [],
            self.cols_to_keep,
            path_outer,
        )
        _check_split_dfs(self.df, [train_df, test_df], verbose=verbose)

        summary_loo_split.loc[len(summary_loo_split.index) + 1, :] = [
            0,
            0,
            train_set,
            test_set,
            len(train_df),
            len(test_df),
            "loo split",
        ]

        summary_loo_split.to_csv(
            os.path.join(
                output_dir,
                write_base_str
                + f".{split_type}.summary.loo.{loo_label.replace('/', '_')}."
                f"{self.return_frac}.csv",
            ),
        )

    def _save_serialized(
        self, frame: FrameType | None, path_serialized: str | os.PathLike
    ) -> None:
        if frame is None:
            raise RuntimeError("Could not get current frame")
        keys, _, _, values = inspect.getargvalues(frame)
        function_name = frame.f_code.co_name
        create_splits_serialized: dict = {}
        for key in keys:
            if key != "self":
                create_splits_serialized[key] = values[key]
        self.serialized["split_function_called"] = function_name
        self.serialized["split_parameters"] = create_splits_serialized

        with open(path_serialized, "w") as f:
            json.dump(self.serialized, f, indent=4)

def _get_unique_split_possibilities(
    df: pd.DataFrame,
    keep_n_elements_in_train: list[int],
    split_type: str,
) -> list[str]:
    """Determines the list of possible unique split labels for the given `split_type`.

    If a specific split possibility contains *only* entries that are part of `keep_n_elements_in_train`,
    then this function will not return it as a split possibility.

    :param df: DataFrame containing the dataset.
    :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
    in the test set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
    :param split_type: String specifying the splitting type.
    :return: List of unique split labels.
    """
    if len(keep_n_elements_in_train) > 0:
        if split_type in ["elements", "periodictablerows", "periodictablegroups"]:
            split_possibilities = sorted(
                {
                    item
                    for sublist in df[~df["nelements"].isin(keep_n_elements_in_train)][split_type]
                    for item in sublist
                }
            )
        else:
            split_possibilities = sorted(
                set(
                    df[~df["nelements"].isin(keep_n_elements_in_train)][split_type]
                )
            )
    elif split_type in ["elements", "periodictablerows", "periodictablegroups"]:
        split_possibilities = sorted(
            {item for sublist in df[split_type] for item in sublist}
        )
    else:
        split_possibilities = sorted(set(df[split_type]))
    return split_possibilities

def _get_split_set_indices(
    df: pd.DataFrame,
    split_set: list[str] | set[str],
    default_indices: list[int],
    split_type: str,
    all_must_be_in_split_set: bool = False,
) -> list[int]:
    """Get the indices for `df` entries that contain label(s) specified in `split_set` (minus the `default_indices`).

    :param df: DataFrame containing the dataset.
    :param split_set: List of split set labels (e.g., spacegroups).
    :param default_indices: List of indices that are part of the training set by default.
    :param split_type: String specifying the splitting type.
    :param all_must_be_in_split_set: If True, only return indices where all entries in `split_set` are present. 
    True used for training sets of elemental-type splits, False for test sets of the same.
    :return: List of indices for the specified split set.
    """
    if len(split_set) == 0:
        return []
    if split_type not in ["elements", "periodictablerows", "periodictablegroups"]:
        return sorted(
            set(df[df[split_type].isin(split_set)].index) - set(default_indices)
        )
    else:
        if all_must_be_in_split_set:
            return sorted(
                set(
                    df[
                        df.apply(
                            lambda x: all(e in split_set for e in x[split_type]),
                            axis=1,
                        )
                    ].index,
                )
                - set(default_indices),
            )
        else:
            return sorted(
                set(
                    df[
                        df.apply(
                            lambda x: any(e in split_set for e in x[split_type]),
                            axis=1,
                        )
                    ].index,
                )
                - set(default_indices),
            )

def _get_train_test_indices(
    df: pd.DataFrame,
    train_set: list[str] | set[str],
    test_set: list[str] | set[str],
    default_train_indices: list[int] | None,
    default_test_indices: list[int] | None,
    split_type: str,
) -> tuple[list[int], list[int]]:
    """Determine the split indices based in the specified `split_type` and train and test sets.

    The returned split indices do not include `default_train_indices`
    (they are added manually later in `_save_split_dfs`).

    :param df: DataFrame containing the dataset.
    :param train_set: Training set options (e.g., list of spacegroups in the training set).
    :param test_set: Test set options.
    :param default_train_indices: List of indices that are part of the training set by default.
    :param default_test_indices: List of indices that are part of the test set by default.
    :param split_type: String specifying the splitting type.
    :return: Tuple of train and test indices lists.
    """
    if default_train_indices is None:
        default_train_indices = []
    if default_test_indices is None:
        default_test_indices = []
    assert len(set(test_set) & set(train_set)) == 0, (
            "Error. The train and test sets have overlapping split labels."
        )
    train_indices = _get_split_set_indices(
        df,
        train_set,
        default_train_indices + default_test_indices,
        split_type,
        all_must_be_in_split_set=True,
    )
    test_indices = _get_split_set_indices(
        df,
        test_set,
        default_train_indices + default_test_indices,
        split_type,
        all_must_be_in_split_set=False,
    )
    assert len(set(train_indices) & set(test_indices)) == 0, (
            "Error. The train and test sets have overlapping indices."
        )
    return train_indices, test_indices


if __name__ == "__main__":
    pass
