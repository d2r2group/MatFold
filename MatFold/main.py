import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import itertools
import os
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def cifs_to_dict(directory: str | os.PathLike) -> dict:
    """
    Converts a directory of cif files into a dictionary with keys '<filename>' (of `<filename>.cif`)
    and values 'pymatgen dictionary' (parsed from `<filename>.cif`)
    :param directory: Directory where cif files are stored
    :return: Dictionary of cif files with keys '<filename>' (of `<filename>.cif`).
    Can be used as input `bulk_df` to `MatFold` class
    """
    output_dict = {}
    for x in os.listdir(directory):
        if x.endswith(".cif"):
            output_dict[x.split(".cif")[0]] = Structure.from_file(os.path.join(directory, x)).as_dict()
    return output_dict


class MatFold:

    def __init__(self, df: pd.DataFrame, bulk_df: dict,
                 return_frac: float = 1.0, always_include_n_elements: list | int | None = None,
                 cols_to_keep: list | None = None, seed: int = 0) -> None:
        """
        MatFold class constructor
        :param df: Pandas dataframe with the first column containing strings of either form `<structureid>` or
        `<structureid>:<structuretag>` (where <structureid> refers to a bulk ID and <structuretag> refers to
        an identifier of a derivative structure). All other columns are optional and may be retained specifying the
        `cols_to_keep` parameter described below.
        :param bulk_df: Dictionary containing <structureid> as keys and the corresponding bulk pymatgen
        dictionary as values.
        :param return_frac: The fraction of the df dataset that is utilized during splitting.
        Must be larger than 0.0 and equal/less than 1.0 (=100%).
        :param always_include_n_elements: A list of number of elements for which the corresponding materials are
        always to be included in the dataset (for cases where `return_frac` < 1.0).
        :param cols_to_keep: List of columns to keep in the splits. If left `None`, then all columns of the
        original df are kept.
        :param seed: Seed for selecting random subset of data and splits.
        """
        self.return_frac = return_frac
        self.seed = seed
        if return_frac <= 0.0 or return_frac > 1.0:
            raise ValueError("Error: `return_frac` needs to be greater than 0.0 and less or equal to 1.0")

        if always_include_n_elements is None:
            always_include_n_elements = []
        elif isinstance(always_include_n_elements, int):
            always_include_n_elements = [always_include_n_elements]

        if cols_to_keep is None:
            self.cols_to_keep = list(df.columns)
        else:
            self.cols_to_keep = cols_to_keep

        self.df = df.copy()
        if len(self.df.iloc[0, 0].split(':')) <= 2:
            self.df['structureid'] = [val.split(':')[0] for val in self.df.iloc[:, 0]]
        else:
            raise ValueError("Error: Materials tags should either be of form "
                             "`<structureid>` or `<structureid>:<structuretag>`.")

        unique_structures = set(self.df['structureid'])

        for us in unique_structures:
            if us not in bulk_df:
                raise ValueError(f"Error: Structure {us} not in `bulk_df` data.")

        structures = dict(
            [(id_, Structure.from_dict(bulk_df[id_])) for id_ in unique_structures]
        )

        space_groups = dict(
            [(id_, SpacegroupAnalyzer(structures[id_], 0.1))
             for id_ in unique_structures]
        )

        self.df['composition'] = [structures[id_].composition.reduced_formula for id_ in self.df['structureid']]

        self.df['chemsys'] = [structures[id_].composition.chemical_system for id_ in self.df['structureid']]

        self.df['sgnum'] = [str(space_groups[id_].get_space_group_symbol()) for id_ in self.df['structureid']]

        self.df['crystalsys'] = [str(space_groups[id_].get_crystal_system()) for id_ in self.df['structureid']]

        self.df['elements'] = [structures[id_].composition.get_el_amt_dict().keys()
                               for id_ in self.df['structureid']]

        self.df['nelements'] = [len(structures[id_].composition.get_el_amt_dict().keys())
                                for id_ in self.df['structureid']]

        self.df['periodictablerows'] = [sorted(list(set([f'row{el.row}'
                                                         for el in structures[id_].composition.elements])))
                                        for id_ in self.df['structureid']]

        self.df['periodictablegroups'] = [sorted(list(set([f'group{el.group}'
                                                           for el in structures[id_].composition.elements])))
                                          for id_ in self.df['structureid']]

        if return_frac < 1.0:
            np.random.seed(self.seed)
            if len(always_include_n_elements) > 0:
                unique_nelements = set(self.df[self.df['nelements'].isin(always_include_n_elements)]['structureid'])
                keep_possibilities = sorted(list(set(self.df[~self.df['nelements'].isin(
                    always_include_n_elements)]['structureid'])))
                n_element_fraction = len(unique_nelements) / (len(keep_possibilities) + len(unique_nelements))
                if n_element_fraction < return_frac:
                    end_keep_index = int(np.round((return_frac - n_element_fraction) * len(keep_possibilities), 0))
                    np.random.shuffle(keep_possibilities)
                    selection = list(keep_possibilities[:end_keep_index]) + list(unique_nelements)
                else:
                    raise ValueError("Error: Fraction of `always_include_n_elements` portion of the dataset is "
                                     "larger than `return_frac`. Either increase `return_frac` or reduce number "
                                     "of n elements to be included.")
            else:
                keep_possibilities = sorted(list(unique_structures))
                end_keep_index = int(np.round(return_frac * len(keep_possibilities), 0))
                np.random.shuffle(keep_possibilities)
                selection = keep_possibilities[:end_keep_index]

            self.df = self.df[self.df['structureid'].isin(selection)]

        self.df['index'] = self.df.index

    def split_statistics(self, split_type: str) -> dict:
        """
        Analyzes the statistics of the sgnum, crystalsys, chemsys, composition, elements,
        periodictablerows, and periodictablegroups splits.
        :param split_type: String specifying the splitting type
        :return: Dictionary with keys of unique split values and the corresponding fraction of this key being
        represented in the entire dataset.
        """
        if split_type not in ["chemsys", "composition", "sgnum", "crystalsys",
                              "elements", "periodictablerows", "periodictablegroups"]:
            return {}
        if split_type in ["elements", "periodictablerows", "periodictablegroups"]:
            statistics = {key: 0. for key in list(set(itertools.chain.from_iterable(self.df[split_type])))}
        else:
            statistics = {key: 0. for key in list(set(self.df[split_type]))}
        for uk in statistics.keys():
            n = 0
            for s in self.df[split_type]:
                if split_type in ["elements", "periodictablerows", "periodictablegroups"]:
                    for e in s:
                        if e == uk:
                            n += 1
                            break
                else:
                    if s == uk:
                        n += 1
            statistics[uk] = n / len(self.df[split_type])
        return statistics

    def create_splits(self, split_type: str, n_inner_splits: int = 10, n_outer_splits: int = 10,
                      default_train_cutoff_fraction: float = 1.0, keep_n_elements_in_train: list | int | None = None,
                      min_train_test_factor: float | None = None, inner_equals_outer_split_strategy: bool = True,
                      write_base_str: str = 'mf', output_dir: str | os.PathLike | None = None,
                      verbose: bool = False) -> None:
        """
        Creates splits based on split_type.
        :param split_type: Defines the type of splitting, must be either "index", "structureid",
        "composition", "chemsys", "sgnum", "crystalsys", "elements", "periodictablerows", or "periodictablegroups"
        :param n_inner_splits: Number of inner splits (for nested k-fold)
        :param n_outer_splits: Number of outer splits (k-fold)
        :param default_train_cutoff_fraction: If a split possiblity exceeds this fraction in the entire dataset
        then the corresponding indices will be forced to be in the training set by default.
        :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
        in the test set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
        :param min_train_test_factor: Minimum factor that the training set needs to be
        larger (for factors greater than 1.0) than the test set.
        :param inner_equals_outer_split_strategy: If true, then the inner splitting strategy used is equal to
        the outer splitting strategy, if false, then inner splitting strategy is random (by index).
        :param write_base_str: Beginning string of csv file names of the written splits
        :param output_dir: Directory where the splits are written to
        :param verbose: Whether to print out details during code execution.
        :return: None
        """
        if output_dir is None:
            output_dir = os.getcwd()

        if split_type not in ["index", "structureid", "composition", "chemsys", "sgnum", "crystalsys",
                              "elements", "periodictablerows", "periodictablegroups"]:
            raise ValueError('Error: `split_type` must be either "index", "structureid", '
                             '"composition", "chemsys", "sgnum", "crystalsys", '
                             '"elements", "periodictablerows", or "periodictablegroups"')

        if keep_n_elements_in_train is None:
            keep_n_elements_in_train = []
        elif isinstance(keep_n_elements_in_train, int):
            keep_n_elements_in_train = [keep_n_elements_in_train]

        default_train_indices = list(self.df[self.df['nelements'].isin(keep_n_elements_in_train)].index
                                     ) if len(keep_n_elements_in_train) > 0 else []
        split_possibilities = self._get_unique_split_possibilities(keep_n_elements_in_train, split_type)

        # Remove splits from test set that have larger fractions than `max_fraction_testset`
        # then add their indices to `default_train_indices`
        remove_from_test_dict = {k: round(v, 3) for k, v in self.split_statistics(split_type).items()
                                 if v > default_train_cutoff_fraction}
        remove_from_test = list(remove_from_test_dict.keys())
        if verbose:
            print(f"The following instances will be removed from possible test sets, as their fraction in the dataset "
                  f"was higher than {default_train_cutoff_fraction}: {remove_from_test_dict}.")
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

        if len(split_possibilities) < n_outer_splits:
            raise ValueError(f'Error: `n_outer_splits`, {n_outer_splits}, is larger than available '
                             f'`split_possibilities`, {len(split_possibilities)} '
                             f'for splitting strategy {split_type} and `max_fraction_testset` '
                             f'cutoff of {default_train_cutoff_fraction}.')

        if verbose:
            if split_type in ["elements", "periodictablerows", "periodictablegroups"]:
                print(f'Default train {split_type} ({len(default_train_elements)}): {default_train_elements}')
            print(f'Default train indices ({len(default_train_indices)}): ', default_train_indices)
            print(f'Possible test examples: {split_possibilities}')
        if n_inner_splits > 1:
            kf_inner = KFold(n_splits=n_inner_splits, random_state=self.seed, shuffle=True)
        else:
            kf_inner = None
        if n_outer_splits > 1:
            kf_outer = KFold(n_splits=n_outer_splits, random_state=self.seed, shuffle=True)
        else:
            raise ValueError("Error: `n_outer_splits` needs to be greater than 1.")

        summary_outer_splits = pd.DataFrame(columns=['n', 'l', 'train', 'test', 'ntrain', 'ntest', 'comment'])

        # Splits for outer loop
        for i, (outer_train_possibility_indices, outer_test_possibility_indices) \
                in enumerate(kf_outer.split(split_possibilities)):
            # Outer train structure ids
            outer_train_set = sorted(list(set(np.take(split_possibilities, outer_train_possibility_indices).tolist() +
                                              default_train_elements)))

            # Outer test structure ids
            outer_test_set = list(np.take(split_possibilities, outer_test_possibility_indices))

            if verbose:
                print(f"\nSplitting: k{i}, {'-'.join(sorted(outer_test_set)) if split_type == 'elements' else ''}")
                print(outer_train_set)
                print(outer_test_set)

            outer_train_indices, outer_test_indices = self._get_split_indices(outer_train_set, outer_test_set,
                                                                              default_train_indices, split_type)

            if not self._check_split_indices_passed(outer_train_indices + default_train_indices, outer_test_indices,
                                                    min_train_test_factor):
                summary_outer_splits.loc[len(summary_outer_splits.index) + 1, :] = \
                    [i, -1, outer_train_set, outer_test_set, len(outer_train_indices) + len(default_train_indices),
                     len(outer_test_indices), f'split check failed - factor = {min_train_test_factor}']
                continue

            path_outer = os.path.join(output_dir, write_base_str + f'.{split_type}.k{i}_outer.csv')

            outer_train_df, outer_test_df = self._save_split_dfs(outer_train_indices, outer_test_indices,
                                                                 default_train_indices, path_outer)
            self._check_split_dfs([outer_train_df, outer_test_df], verbose=verbose)

            summary_outer_splits.loc[len(summary_outer_splits.index) + 1, :] = \
                [i, -1, outer_train_set, outer_test_set, len(outer_train_df), len(outer_test_df), 'split successful']

            if kf_inner is not None and inner_equals_outer_split_strategy:
                inner_split_possibilities = [split for split in outer_train_set if split not in default_train_elements]
                if len(inner_split_possibilities) < n_inner_splits:
                    raise ValueError(f'Error: `n_inner_splits`, {n_inner_splits}, is larger than available '
                                     f'`split_possibilities` of the inner train set, {len(inner_split_possibilities)} '
                                     f'for splitting strategy {split_type}, `max_fraction_testset` '
                                     f'cutoff of {default_train_cutoff_fraction}, and `n_outer_splits` of {n_outer_splits}.')
                for j, (inner_train_possibility_indices, inner_test_possibility_indices) in (
                        enumerate(kf_inner.split(list(inner_split_possibilities)))):
                    # Inner train structure ids
                    inner_train_set = sorted(list(set(
                        np.take(list(inner_split_possibilities), inner_train_possibility_indices).tolist() +
                        default_train_elements)))

                    # Inner test structure ids
                    inner_test_set = list(np.take(list(inner_split_possibilities), inner_test_possibility_indices))

                    if verbose:
                        print(f"\nSplitting: k{i}, i{j}, "
                              f"{'-'.join(sorted(inner_test_set)) if split_type == 'elements' else ''}")
                        print(inner_train_set)
                        print(inner_test_set)

                    inner_train_indices, inner_test_indices = self._get_split_indices(inner_train_set, inner_test_set,
                                                                                      default_train_indices, split_type)
                    # Ensure that no outer_test_indices are present in inner_test_indices
                    inner_test_indices = [ind for ind in inner_test_indices if ind not in outer_test_indices]

                    if not self._check_split_indices_passed(inner_train_indices + default_train_indices,
                                                            inner_test_indices, min_train_test_factor):
                        summary_outer_splits.loc[len(summary_outer_splits.index) + 1, :] = \
                            [i, j, inner_train_set, inner_test_set,
                             len(inner_train_indices) + len(default_train_indices),
                             len(inner_test_indices), f'split check failed - factor = {min_train_test_factor}']
                        continue

                    path_inner = os.path.join(output_dir, write_base_str + f'.{split_type}.k{i}_outer.l{j}_inner.csv')

                    inner_train_df, inner_test_df = self._save_split_dfs(inner_train_indices, inner_test_indices,
                                                                         default_train_indices, path_inner)
                    self._check_split_dfs([outer_test_df, inner_train_df, inner_test_df], verbose=verbose)
                    summary_outer_splits.loc[len(summary_outer_splits.index) + 1, :] = \
                        [i, j, inner_train_set, inner_test_set, len(inner_train_indices) + len(default_train_indices),
                         len(inner_test_indices), 'split successful']

            elif kf_inner is not None:
                for j, (train_inner_index_index, test_inner_index_index) in (
                        enumerate(kf_inner.split(outer_train_indices))):
                    if verbose:
                        print(f'Splitting inner {str(j)}')
                    train_inner_index = np.take(outer_train_indices, train_inner_index_index)
                    test_inner_index = np.take(outer_train_indices, test_inner_index_index)

                    final_inner_train_indices = default_train_indices + list(train_inner_index)
                    final_inner_test_indices = test_inner_index.copy()

                    path_inner = os.path.join(output_dir, write_base_str + f'.{split_type}.k{i}_outer.l{j}_inner.csv')

                    inner_train_df, inner_test_df = self._save_split_dfs(final_inner_train_indices,
                                                                         final_inner_test_indices, [], path_inner)
                    self._check_split_dfs([outer_test_df, inner_train_df, inner_test_df], verbose=verbose)

        summary_outer_splits.to_csv(os.path.join(output_dir, write_base_str +
                                                 f'.{split_type}.summary.k{n_outer_splits}.l{n_inner_splits}.'
                                                 f'{self.return_frac}.csv'))

    def create_loo_split(self, split_type: str, loo_label: str, keep_n_elements_in_train: list | int | None = None,
                         write_base_str: str = 'mf', output_dir: str | os.PathLike | None = None,
                         verbose: bool = False) -> None:
        """
        Creates leave-one-out split by `split_type` and specified `loo_label`.
        :param split_type: Defines the type of splitting, must be either "structureid", "composition", "chemsys",
        "sgnum", "crystalsys", "elements", "periodictablerows", or "periodictablegroups".
        :param loo_label: Label specifying which single option is to be left out (i.e., constitute the test set).
        This label must be a valid option for the specified `split_type`.
        :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
        in the test set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
        :param write_base_str: Beginning string of csv file names of the written splits
        :param output_dir: Directory where the splits are written to
        :param verbose: Whether to print out details during code execution.
        :return: None
        """
        if output_dir is None:
            output_dir = os.getcwd()

        if split_type not in ["structureid", "composition", "chemsys", "sgnum", "crystalsys",
                              "elements", "periodictablerows", "periodictablegroups"]:
            raise ValueError('Error: `split_type` must be either "structureid", '
                             '"composition", "chemsys", "sgnum", "crystalsys", '
                             '"elements", "periodictablerows", or "periodictablegroups"')

        if keep_n_elements_in_train is None:
            keep_n_elements_in_train = []
        elif isinstance(keep_n_elements_in_train, int):
            keep_n_elements_in_train = [keep_n_elements_in_train]

        default_train_indices = list(self.df[self.df['nelements'].isin(keep_n_elements_in_train)].index
                                     ) if len(keep_n_elements_in_train) > 0 else []
        split_possibilities = self._get_unique_split_possibilities(keep_n_elements_in_train, split_type)

        if loo_label not in split_possibilities:
            raise ValueError(f'Error. LOO label ({loo_label}) is not in `split_possibilities` of type {split_type}.')

        summary_loo_split = pd.DataFrame(columns=['n', 'l', 'train', 'test', 'ntrain', 'ntest', 'comment'])

        train_set = [sp for sp in split_possibilities if sp != loo_label]
        test_set = [loo_label]

        train_indices, test_indices = self._get_split_indices(train_set, test_set, default_train_indices, split_type)

        if len(train_indices + default_train_indices) == 0 or len(test_indices) == 0:
            raise Exception(f"Error. Either train (len={len(train_indices + default_train_indices)}) or test "
                            f"(len={len(test_indices)}) set is empty and split cannot be created.")

        path_outer = os.path.join(output_dir, write_base_str + f'.{split_type}.loo.{loo_label.replace("/", "_")}.csv')
        train_df, test_df = self._save_split_dfs(train_indices, test_indices, default_train_indices, path_outer)
        self._check_split_dfs([train_df, test_df], verbose=verbose)

        summary_loo_split.loc[len(summary_loo_split.index) + 1, :] = \
            [0, 0, train_set, test_set, len(train_df), len(test_df), 'loo split']

        summary_loo_split.to_csv(os.path.join(output_dir, write_base_str +
                                              f'.{split_type}.summary.loo.{loo_label.replace("/", "_")}.'
                                              f'{self.return_frac}.csv'))

    @staticmethod
    def _check_split_indices_passed(train_indices: list[int], test_indices: list[int],
                                    min_train_test_factor: float | None) -> bool:
        """
        Checks if the lists of train and test indices have no intersection and are non-zero in length.
        :param train_indices: List of train indices.
        :param test_indices: List of test indices.
        :param min_train_test_factor: Minimum factor that the training set needs to be
        larger (for factors greater than 1.0) than the test set.
        :return: Returns false if any of the lists is of length zero, true otherwise.
        """
        if len(set(train_indices).intersection(set(test_indices))) != 0:
            raise Exception(f'Error: Training and test indices are not mutually exclusive '
                            f'({len(set(train_indices).intersection(set()))} indices in common).')
        if min_train_test_factor is not None:
            if len(train_indices) < len(test_indices) * min_train_test_factor:
                print(f"Warning! Train set size ({len(train_indices)}) is smaller than "
                      f"test set size times min_train_test_factor ({len(test_indices)} * {min_train_test_factor} = "
                      f"{int(round(len(test_indices) * min_train_test_factor, 0))}).", flush=True)
                return False
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Warning! Either train (len={len(train_indices)}) or test "
                  f"(len={len(test_indices)}) set is empty and split cannot be created.", flush=True)
            return False
        return True

    def _check_split_dfs(self, df_list: list[pd.DataFrame], verbose: bool = True) -> None:
        """
        Checks that there are no duplicates in dfs in `df_list` and that the number of indices in `self.df` is the same
        as in the combined dfs in `df_list`
        :param df_list: List of sub dataframes
        :param verbose: Whether to print out the lengths of the `df_list` dfs
        :return: None
        """
        indices_list = [list(df.index) for df in df_list]
        sizes_list = [len(lst) for lst in indices_list]

        if verbose:
            print(f"Individual lengths of indices lists = {'+'.join([str(s) for s in sizes_list])} = {sum(sizes_list)}."
                  f" Original total length of dataframe indices = {len(self.df)}")

        # counts the number of duplicate indices for each individual df
        duplicates = [len([item for item, count in collections.Counter(lst).items() if count > 1])
                      for lst in indices_list]
        if np.sum(duplicates) != 0:
            raise Exception('Error: Duplicate indices detected within individual dfs: ', duplicates)

        if sum(sizes_list) != len(self.df):
            raise Exception(f'Error: Non-equal num of indices in splits {sum(sizes_list)} vs. original {len(self.df)}.')

    def _save_split_dfs(self, train_indices: list, test_indices: list, default_train_indices: list,
                        path: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates train and test dfs and saves them as csv files. Note that the path needs to end with `.csv` and
        the output file endings will be changed to `<>.train.csv` and `<>.test.csv`
        :param train_indices: List of train indices
        :param test_indices: List of test indices
        :param default_train_indices: List of indices that are part of the training set by default
        :param path: Path for the output files. Must end with `.csv`
        :return: Tuple of train and test dataframes
        """
        test_df = self.df.loc[test_indices, :].copy()
        train_df = self.df.loc[train_indices + default_train_indices, :].copy()
        train_df.loc[:, self.cols_to_keep].to_csv(path.replace('.csv', '.train.csv'),
                                                  header=True, index=False)
        test_df.loc[:, self.cols_to_keep].to_csv(path.replace('.csv', '.test.csv'),
                                                 header=True, index=False)
        return train_df, test_df

    def _get_split_indices(self, train_set: list | set, test_set: list | set,
                           default_train_indices: list | None, split_type: str) -> tuple[list, list]:
        """
        Determines the split indices based in the specified `split_type` and train and test sets. The returned split
        indices do not include `default_train_indices` (they are added manually later in `_save_split_dfs`).
        :param train_set: Training set options (e.g., list of spacegroups in the training set)
        :param test_set: Test set options
        :param default_train_indices: List of indices that are part of the training set by default
        :param split_type: String specifying the splitting type
        :return: Tuple of train and test indices lists
        """
        if default_train_indices is None:
            default_train_indices = []
        if split_type not in ["elements", "periodictablerows", "periodictablegroups"]:
            # indices of all examples for outer train fold (less any specified by default_train_indices)
            train_indices = list(
                set(self.df[self.df[split_type].isin(train_set)].index) - set(default_train_indices)
            )
            # indices of all examples for outer test fold (less any specified by default_train_indices)
            test_indices = list(
                set(self.df[self.df[split_type].isin(test_set)].index) - set(default_train_indices)
            )
        else:
            # Indices of all examples whose structures contain only train elements
            train_indices = list(
                set(
                    self.df[
                        self.df.apply(
                            lambda x: all([e in train_set for e in x[split_type]]),
                            # check that all elements are in the training set options, if any is not,
                            # then it won't be part of final training set indices
                            axis=1
                        )
                    ].index
                ) - set(default_train_indices)
            )

            # indices of all examples whose structures contain a test element
            test_indices = list(
                set(
                    self.df[
                        self.df.apply(
                            lambda x: any([e in test_set for e in x[split_type]]),
                            axis=1
                        )
                    ].index
                ) - set(default_train_indices)
            )
        return train_indices, test_indices

    def _get_unique_split_possibilities(self, keep_n_elements_in_train: list, split_type: str) -> list:
        """
        Determines the list of possible unique split labels for the given `split_type`.
        :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
        in the test set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
        :param split_type: String specifying the splitting type
        :return: List of unique split labels
        """
        if len(keep_n_elements_in_train) > 0:
            if split_type in ["elements", "periodictablerows", "periodictablegroups"]:
                split_possibilities = list(set(itertools.chain.from_iterable(
                    self.df[~self.df['nelements'].isin(keep_n_elements_in_train)][split_type])))
            else:
                split_possibilities = list(
                    set(self.df[~self.df['nelements'].isin(keep_n_elements_in_train)][split_type]))
        else:
            if split_type in ["elements", "periodictablerows", "periodictablegroups"]:
                split_possibilities = list(set(itertools.chain.from_iterable(self.df[split_type])))
            else:
                split_possibilities = list(set(self.df[split_type]))
        return sorted(split_possibilities)


if __name__ == "__main__":
    import json
    # cifs = cifs_to_dict('./test/')
    # # print(cifs.keys())
    # with open('test.json', 'w') as fp:
    #     json.dump(cifs, fp)
    with open('test.json', 'r') as fp:
        cifs = json.load(fp)
    mf = MatFold(pd.read_csv('./test.csv', header=None), cifs,
                 return_frac=0.5, always_include_n_elements=None)
    stats = mf.split_statistics('crystalsys')
    print(stats)
    mf.create_splits("periodictablegroups", n_outer_splits=3, n_inner_splits=2,
                     default_train_cutoff_fraction=0.8, keep_n_elements_in_train=None, min_train_test_factor=None,
                     output_dir='./output/', verbose=True)
    mf.create_loo_split("elements", 'Fe', keep_n_elements_in_train=None,
                        output_dir='./output/', verbose=True)
