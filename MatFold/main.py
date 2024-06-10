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
                 return_frac: float = 1.0, always_include_n_elements: list | int | None = None,
                 cols_to_keep: list | None = None) -> None:
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
        """
        self.return_frac = return_frac
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

        df_split = df.copy()
        if len(df_split.iloc[0, 0].split(':')) <= 2:
            df_split['structureid'] = [val.split(':')[0] for val in df_split.iloc[:, 0]]
        else:
            raise ValueError("Error: Materials tags should either be of form "
                             "`<structureid>` or `<structureid>:<structuretag>`.")

        unique_structures = set(df_split['structureid'])
        
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

        df_split['composition'] = [structures[id_].composition.reduced_formula for id_ in df_split['structureid']]

        df_split['chemsys'] = [structures[id_].composition.chemical_system for id_ in df_split['structureid']]

        df_split['sgnum'] = [str(space_groups[id_].get_space_group_symbol()) for id_ in df_split['structureid']]

        df_split['crystalsys'] = [str(space_groups[id_].get_crystal_system()) for id_ in df_split['structureid']]

        df_split['elements'] = [structures[id_].composition.get_el_amt_dict().keys()
                                for id_ in df_split['structureid']]

        df_split['nelements'] = [len(structures[id_].composition.get_el_amt_dict().keys())
                                 for id_ in df_split['structureid']]

        if return_frac < 1.0:
            np.random.seed(self.return_seed)
            if len(always_include_n_elements) > 0:
                unique_nelements = set(df_split[df_split['nelements'].isin(always_include_n_elements)]['structureid'])
                keep_possibilities = sorted(list(set(df_split[~df_split['nelements'].isin(
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

            df_split = df_split[df_split['structureid'].isin(selection)]

        df_split['index'] = df_split.index
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
                      max_fraction_testset: float = 1.0, keep_n_elements_in_train: list | int | None = None,
                      write_base_str: str = 'mf', output_dir: str | os.PathLike | None = None,
                      verbose: bool = False) -> None:
        """
        Creates splits based on split_type.
        :param split_type: Defines the type of splitting, must be either "index", "structureid",
        "composition", "chemsys", "sgnum", "crystalsys", or "elements"
        :param n_inner_splits: Number of inner splits (for nested k-fold)
        :param n_outer_splits: Number of outer splits (k-fold)
        :param max_fraction_testset: The maximum fraction a key can be represented in the entire dataset to still be
        considered a part of the test set during k-fold process.
        :param keep_n_elements_in_train: List of number of elements for which the corresponding materials are kept
        in the test set (i.e., not k-folded).
        :param write_base_str: Beginning string of csv file names of the written splits
        :param output_dir: Directory where the splits are written to
        :param verbose: Whether to print out details during code execution.
        :return: None
        """

        if output_dir is None:
            output_dir = os.getcwd()

        if split_type not in ["index", "structureid", "composition", "chemsys", "sgnum", "crystalsys", "elements"]:
            raise ValueError('Error: `split_type` must be either "index", "structureid", '
                             '"composition", "chemsys", "sgnum", "crystalsys", or "elements"')

        out_df = self.df_decorated.copy()

        if keep_n_elements_in_train is not None:
            if isinstance(keep_n_elements_in_train, int):
                keep_n_elements_in_train = [keep_n_elements_in_train]
            default_train_indices = list(out_df[out_df['nelements'].isin(keep_n_elements_in_train)].index)
            if split_type == "elements":
                test_possibilities = list(set(itertools.chain.from_iterable(
                    out_df[~out_df['nelements'].isin(keep_n_elements_in_train)]['elements'])))
            else:
                test_possibilities = list(set(out_df[~out_df['nelements'].isin(keep_n_elements_in_train)][split_type]))
        else:
            default_train_indices = []
            if split_type == "elements":
                test_possibilities = list(set(itertools.chain.from_iterable(out_df['elements'])))
            else:
                test_possibilities = list(set(out_df[split_type]))

        # Remove splits from test set that have larger fractions than `max_fraction_testset`
        # then add their indices to `default_train_indices`
        remove_from_test = [k for k, v in self.split_statistics(split_type).items() if v > max_fraction_testset]
        if verbose:
            print(f"The following instances will be removed from possible test sets, as their fraction in the dataset "
                  f"was higher than {max_fraction_testset}: {remove_from_test}.")
        add_train_indices = []
        for r in set(remove_from_test):
            test_possibilities.remove(r)
            add_train_indices.extend(list(out_df[out_df[split_type] == r].index))
        if split_type == "elements":
            default_train_elements = remove_from_test.copy()
        else:
            default_train_indices.extend(add_train_indices)
            default_train_indices = list(set(default_train_indices))
            default_train_elements = []

        if len(test_possibilities) < n_outer_splits:
            raise ValueError(f'Error: `n_outer_splits`, {n_outer_splits}, is larger than available '
                             f'`test_possibilities`, {len(test_possibilities)} '
                             f'for splitting strategy {split_type} and `max_fraction_testset` '
                             f'cutoff of {max_fraction_testset}.')

        if verbose:
            print(f'Default train indices ({len(default_train_indices)}): ',
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

        summary_outer_splits = pd.DataFrame(columns=['train', 'test', 'ntrain', 'ntest'])

        # Splits for outer loop
        for i, (train_outer_set_index, test_outer_set_index) in enumerate(kf_outer.split(test_possibilities)):
            # train structure ids
            outer_train_set = set(np.take(test_possibilities, train_outer_set_index).tolist() + default_train_elements)

            # test structure ids
            outer_test_set = set(np.take(test_possibilities, test_outer_set_index))

            if verbose:
                print(f"Splitting outer: k{i} {'-'.join(sorted(outer_test_set)) if split_type == 'elements' else ''}")
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
                                lambda x: all([e in outer_train_set for e in x['elements']]),
                                # check that all elements are in the outer training set options, if any is not,
                                # then it won't be part of final training set indices
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

            summary_outer_splits.loc[i, :] = [outer_train_set, outer_test_set,
                                              len(outer_train_indices), len(outer_test_indices)]

            if len(outer_train_indices) == 0:
                print(f"Warning! Every structure contains test elements ({outer_test_set}) and splits "
                      f"could not be created for outer fold k{i}.", flush=True)
                continue

            outer_test_df = out_df.loc[outer_test_indices, :].copy()
            outer_train_df = out_df.loc[outer_train_indices + default_train_indices, :].copy()

            self.check_split(out_df, [outer_train_df, outer_test_df], verbose=verbose)

            outer_train_df.loc[:, self.cols_to_keep].to_csv(
                os.path.join(output_dir, write_base_str + f'.{split_type}.k{i}_outer.train.csv'),
                header=True, index=False
            )

            outer_test_df.loc[:, self.cols_to_keep].to_csv(
                os.path.join(output_dir, write_base_str + f'.{split_type}.k{i}_outer.test.csv'),
                header=True, index=False
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
                                     write_base_str + f'.{split_type}.k{i}_outer.l{j}_inner.train.csv'),
                        header=True, index=False
                    )
                    inner_test_df.loc[:, self.cols_to_keep].to_csv(
                        os.path.join(output_dir,
                                     write_base_str + f'.{split_type}.k{i}_outer.l{j}_inner.test.csv'),
                        header=True, index=False
                    )

                    self.check_split(out_df, [outer_test_df, inner_train_df, inner_test_df], verbose=verbose)

        summary_outer_splits.index.name = 'n'
        summary_outer_splits.to_csv(os.path.join(output_dir, write_base_str +
                                                 f'.{split_type}.k{n_outer_splits}.l{n_inner_splits}.'
                                                 f'{self.return_frac}_summary.csv'))


if __name__ == "__main__":
    import json
    # cifs = cifs_to_dict('./test/')
    # # print(cifs.keys())
    # with open('test.json', 'w') as fp:
    #     json.dump(cifs, fp)
    with open('test.json', 'r') as fp:
        cifs = json.load(fp)
    mf = MatFold(pd.read_csv('./test.csv', header=None), cifs,
                 return_frac=0.5, always_include_n_elements=2)
    stats = mf.split_statistics('crystalsys')
    print(stats)
    mf.create_splits("composition", n_outer_splits=5, n_inner_splits=5,
                     max_fraction_testset=0.3, keep_n_elements_in_train=2,
                     output_dir='./output/', verbose=True)
