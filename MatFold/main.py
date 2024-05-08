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
    return_seed: int = 0

    def __init__(self, df: pd.DataFrame, bulk_df: dict,
                 return_frac: float = 1.0, always_return_binary: bool = True,
                 cols_to_keep: list | None = None) -> None:
        """
        MatFold class constructor
        :param df: Pandas dataframe with the first column containing strings of either form `<structure_id>` or
        `<structure_id>:<structure_tag>` (where <structure_id> refers to a bulk ID and <structure_tag> refers to
        an identifier of a derivative structure). All other columns are optional and may be retained specifying the
        `cols_to_keep` parameter described below.
        :param bulk_df: Dictionary containing <structure_id> as keys and the corresponding bulk pymatgen
        dictionary as values.
        :param return_frac: The fraction of the df dataset that is utilized during splitting.
        Must be larger than 0.0 and equal/less than 1.0 (=100%).
        :param always_return_binary: Whether to always return binaries.
        :param cols_to_keep: List of columns to keep in the splits. If left `None`, then all columns of the
        original df are kept.
        """
        if return_frac <= 0.0 or return_frac > 1.0:
            raise ValueError("Error: `return_frac` needs to be greater than 0.0 and less or equal to 1.0")

        if cols_to_keep is None:
            self.cols_to_keep = list(np.arange(df.shape[1]))
        else:
            self.cols_to_keep = cols_to_keep

        df_split = df.copy()
        if len(df_split[0][0].split(':')) == 2:
            df_split['structure_id'] = [val.split(':')[0] for val in df_split[0]]
            df_split['structure_tag'] = [val.split(':')[1] for val in df_split[0]]
        elif len(df_split[0][0].split(':')) == 1:
            df_split['structure_id'] = [val.split(':')[0] for val in df_split[0]]
        else:
            raise ValueError("Error: Materials tags should either be of form "
                             "`<structure_id>` or `<structure_id>:<structure_tag>`.")

        unique_structures = set(df_split['structure_id'])
        
        for us in unique_structures:
            if us not in bulk_df:
                raise ValueError(f"Error: Structure {us} not in `bulk_df` data.")

        structures = dict(
            [(id_, Structure.from_dict(bulk_df[id_])) for id_ in unique_structures]
        )

        unique_elements = set(list(itertools.chain(
            *[structures[id_].composition.get_el_amt_dict().keys() for id_ in unique_structures]
        )))

        space_groups = dict(
            [(id_, SpacegroupAnalyzer(structures[id_], 0.1))
             for id_ in unique_structures]
        )

        df_split['composition'] = [structures[id_].composition.reduced_formula for id_ in df_split['structure_id']]

        df_split['chemsys'] = [structures[id_].composition.chemical_system for id_ in df_split['structure_id']]

        df_split['sgnum'] = [str(space_groups[id_].get_space_group_symbol()) for id_ in df_split['structure_id']]

        df_split['crystalsys'] = [str(space_groups[id_].get_crystal_system()) for id_ in df_split['structure_id']]

        df_split['elements'] = [structures[id_].composition.get_el_amt_dict().keys()
                                for id_ in df_split['structure_id']]

        df_split['isbinary'] = [True if len(structures[id_].composition.get_el_amt_dict().keys()) == 2 else False
                                for id_ in df_split['structure_id']]

        for elem in unique_elements:
            df_split[f'contains_{elem}'] = df_split.apply(lambda x: elem in x['elements'], axis=1)

        if return_frac < 1.0:
            np.random.seed(self.return_seed)
            if always_return_binary:
                unique_binaries = set(df_split[df_split['isbinary']])
                keep_possibilities = sorted(list(set(df_split[not df_split['isbinary']]['structure_id'])))
                end_keep_index = int(np.round(return_frac * len(keep_possibilities), 0))
                np.random.shuffle(keep_possibilities)
                selection = list(keep_possibilities[:end_keep_index]) + list(unique_binaries)
            else:
                keep_possibilities = sorted(list(unique_structures))
                end_keep_index = int(np.round(return_frac * len(keep_possibilities), 0))
                np.random.shuffle(keep_possibilities)
                selection = keep_possibilities[:end_keep_index]

            df_split = df_split[df_split['structure_id'].isin(selection)]

        self.df_decorated = df_split.copy()

    @staticmethod
    def check_split(df_full: pd.DataFrame, df_list: list[pd.DataFrame], verbose: bool = True) -> None:
        """
        Checks that there are no duplicates in `df_full` and that the number of indices in `df_full` is the same
        as in the combined df's in `df_list`
        :param df_full: Full dataframe
        :param df_list: List of sub dataframes
        :param verbose: Whether to print out the lengths of the `df_list` df's
        :return: None
        """
        indices_list = [list(df.index) for df in df_list]
        sizes_list = [len(lst) for lst in indices_list]
        # combined_indices_list = list(itertools.chain(*indices_list))
        if verbose:
            print(sizes_list, sum(sizes_list), len(df_full))

        # counts the number of duplicate indices for each individual df
        duplicates = [len([item for item, count in collections.Counter(lst).items() if count > 1])
                      for lst in indices_list]
        try:
            assert np.sum(duplicates) == 0
        except Exception as e:
            print('Duplicate indices detected within individual dfs: ', duplicates)
            raise Exception(e)

        try:
            assert sum(sizes_list) == len(df_full)
        except Exception as e:
            print(f'Non-equal num of indices in splits {sum(sizes_list)} vs. original {len(df_full)}.')
            raise Exception(e)

    def split_statistics(self, split_type: str) -> dict:
        """
        Analyzes the statistics of the sgnum, crystalsys, chemsys, composition, or elements splits.
        :param split_type: String specifying the splitting type
        :return: Dictionary with keys of unique split values and the corresponding fraction of this key being
        represented in the entire dataset.
        """
        if split_type not in ["chemsys", "composition", "sgnum", "crystalsys", "elements"]:
            return {}
        if split_type == "elements":
            statistics = {key: 0. for key in list(set(itertools.chain.from_iterable(self.df_decorated[split_type])))}
        else:
            statistics = {key: 0. for key in list(set(self.df_decorated[split_type]))}
        for uk in statistics.keys():
            n = 0
            for s in self.df_decorated[split_type]:
                if split_type == "elements":
                    for e in s:
                        if e == uk:
                            n += 1
                            break
                else:
                    if s == uk:
                        n += 1
            statistics[uk] = n / len(self.df_decorated[split_type])
        return statistics

    def create_splits(self, split_type: str, n_inner_splits: int = 10, n_outer_splits: int = 10,
                      max_fraction_testset: float = 1.0, keep_binaries_in_train: bool = True,
                      write_base_str: str = 'mf', output_dir: str | os.PathLike | None = None,
                      verbose: bool = False) -> None:
        """
        Creates splits based on split_type.
        :param split_type: Defines the type of splitting, must be either "index", "structure_id",
        "composition", "chemsys", "sgnum", "crystalsys", or "elements"
        :param n_inner_splits: Number of inner splits (for nested k-fold)
        :param n_outer_splits: Number of outer splits (k-fold)
        :param max_fraction_testset: The maximum fraction a key can be represented in the entire dataset to still be
        considered a part of the test set during k-fold process.
        :param keep_binaries_in_train: Whether to always keep binaries in training set
        :param write_base_str: Beginning string of csv file names of the written splits
        :param output_dir: Directory where the splits are written to
        :param verbose: Whether to print out details during code execution.
        :return: None
        """

        if output_dir is None:
            output_dir = os.getcwd()

        if split_type not in ["index", "structure_id", "composition", "chemsys", "sgnum", "crystalsys", "elements"]:
            raise ValueError('Error: `split_type` must be either "index", "structure_id", '
                             '"composition", "chemsys", "sgnum", "crystalsys", or "elements"')
        if split_type == "index":
            split_type = 0

        out_df = self.df_decorated.copy()

        if keep_binaries_in_train:
            # default_train_indices = list(out_df[out_df['n_elements'] == 2].index)
            default_train_indices = list(out_df[out_df['isbinary']].index)
            if split_type == "elements":
                test_possibilities = list(set(itertools.chain.from_iterable(out_df[~out_df['isbinary']]['elements'])))
            else:
                test_possibilities = list(set(out_df[~out_df['isbinary']][split_type]))
        else:
            default_train_indices = []
            if split_type == "elements":
                test_possibilities = list(set(itertools.chain.from_iterable(out_df['elements'])))
            else:
                test_possibilities = list(set(out_df[split_type]))

        # Remove splits from test set that have larger fractions than `max_fraction_testset`
        # then add their indices to `default_train_indices`
        remove_from_test = [k for k, v in self.split_statistics(split_type).items() if v > max_fraction_testset]
        add_train_indices = []
        for r in set(remove_from_test):
            test_possibilities.remove(r)
            add_train_indices.extend(list(out_df[out_df[split_type] == r].index))
        default_train_indices.extend(add_train_indices)
        default_train_indices = list(set(default_train_indices))

        if len(test_possibilities) < n_outer_splits:
            raise ValueError(f'Error: `n_outer_splits`, {n_outer_splits}, is larger than available '
                             f'`test_possibilities`, {len(test_possibilities)} '
                             f'for splitting strategy {split_type} and `max_fraction_testset` '
                             f'cutoff of {max_fraction_testset}.')

        if verbose:
            print(f'Default train indices (hard-coded for binaries right now) ({len(default_train_indices)}): ',
                  default_train_indices)
            print('Possible test examples: ', set(test_possibilities))
        if n_inner_splits > 1:
            kf_inner = KFold(n_splits=n_inner_splits, random_state=self.return_seed, shuffle=True)
        else:
            kf_inner = None
        if n_outer_splits > 1:
            kf_outer = KFold(n_splits=n_outer_splits, random_state=self.return_seed, shuffle=True)
        else:
            raise ValueError("Error: `n_outer_splits` needs to be greater than 1.")

        # Splits for outer loop
        for i, (train_outer_set_index, test_outer_set_index) in enumerate(kf_outer.split(test_possibilities)):
            # train structure ids
            outer_train_set = set(np.take(test_possibilities, train_outer_set_index))

            # test structure ids
            outer_test_set = set(np.take(test_possibilities, test_outer_set_index))

            outersplit_string = '-'.join(sorted(outer_test_set)) if split_type == "elements" else f"k{str(i)}"
            if verbose:
                print('Splitting outer: ', outersplit_string)
                print(outer_train_set)
                print(outer_test_set)

            # ensure no overlap of outer train / test splitting criteria
            assert len(outer_train_set.intersection(outer_test_set)) == 0

            if split_type != "elements":
                # indices of all examples for outer train fold (less any specified by default_train_indices)
                outer_train_indices = list(
                    set(out_df[out_df[split_type].isin(outer_train_set)].index) - set(default_train_indices)
                )
                # indices of all examples for outer test fold (less any specified by default_train_indices)
                outer_test_indices = list(
                    set(out_df[out_df[split_type].isin(outer_test_set)].index) - set(default_train_indices)
                )
            else:
                # Indices of all examples whose structures contain only train elements
                outer_train_indices = list(
                    set(
                        out_df[
                            out_df.apply(
                                lambda x: len(x['elements'] & outer_train_set) == len(x['elements']),
                                axis=1
                            )
                        ].index
                    ) - set(default_train_indices)
                )
                # indices of all examples whose structures contain a test element
                outer_test_indices = list(
                    set(
                        out_df[
                            out_df.apply(
                                lambda x: len(x['elements'] & outer_test_set) > 0,
                                axis=1
                            )
                        ].index
                    ) - set(default_train_indices)
                )

            if len(outer_train_indices) == 0:
                # every single structure has the test element(s), so no training is possible
                continue

            outer_test_df = out_df.loc[outer_test_indices, :].copy()
            outer_train_df = out_df.loc[outer_train_indices + default_train_indices, :].copy()

            self.check_split(out_df, [outer_train_df, outer_test_df], verbose=verbose)

            outer_train_df.loc[:, self.cols_to_keep].to_csv(
                os.path.join(output_dir, write_base_str + f'.{split_type}.{outersplit_string}_outer.train.csv'),
                header=False, index=False
            )

            outer_test_df.loc[:, self.cols_to_keep].to_csv(
                os.path.join(output_dir, write_base_str + f'.{split_type}.{outersplit_string}_outer.test.csv'),
                header=False, index=False
            )

            if kf_inner is not None:
                for j, (train_inner_index_index, test_inner_index_index) in (
                        enumerate(kf_inner.split(outer_train_indices))):
                    if verbose:
                        print(f'Splitting inner {str(j)}')
                    train_inner_index = np.take(outer_train_indices, train_inner_index_index)
                    test_inner_index = np.take(outer_train_indices, test_inner_index_index)

                    final_inner_train_indices = default_train_indices + list(train_inner_index)
                    final_inner_test_indices = test_inner_index.copy()

                    inner_train_df = out_df.loc[final_inner_train_indices, :].copy()
                    inner_test_df = out_df.loc[final_inner_test_indices, :].copy()

                    inner_train_df.loc[:, self.cols_to_keep].to_csv(
                        os.path.join(output_dir,
                                     write_base_str + f'.{split_type}.{outersplit_string}_outer.l{j}_inner.train.csv'),
                        header=False, index=False
                    )
                    inner_test_df.loc[:, self.cols_to_keep].to_csv(
                        os.path.join(output_dir,
                                     write_base_str + f'.{split_type}.k{outersplit_string}_outer.l{j}_inner.test.csv'),
                        header=False, index=False
                    )

                    self.check_split(out_df, [outer_test_df, inner_train_df, inner_test_df], verbose=verbose)


if __name__ == "__main__":
    import json
    # cifs = cifs_to_dict('./test/')
    # # print(cifs.keys())
    # with open('test.json', 'w') as fp:
    #     json.dump(cifs, fp)
    with open('test.json', 'r') as fp:
        cifs = json.load(fp)
    mf = MatFold(pd.read_csv('./test.csv', header=None), cifs)
    stats = mf.split_statistics('crystalsys')
    print(stats)
    mf.create_splits("elements", n_outer_splits=5, n_inner_splits=1, max_fraction_testset=0.3,
                     output_dir='./output/', verbose=True)
