<div align="center">
  <img alt="MatFold Logo" src=logo.svg width="200"><br>
</div>

# `MatFold` – Cross-validation Protocols for Materials Science Data 

![Python - Version](https://img.shields.io/pypi/pyversions/MatFold)
[![PyPI - Version](https://img.shields.io/pypi/v/MatFold?color=blue)](https://pypi.org/project/MatFold)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13147391.svg)](https://doi.org/10.5281/zenodo.13147391)

This is a Python package for gaining systematic insights into materials discovery models’ 
performance through standardized, reproducible, and featurization-agnostic chemical and structural cross-validation protocols.

Please, cite the following paper if you use the model in your research:
> Matthew D. Witman and Peter Schindler, *MatFold: systematic insights into materials discovery models’ performance 
> through standardized cross-validation protocols*, ArXiv (2024)

## Installation

`MatFold` can be installed using pip by running `pip install MatFold`.
Alternatively, this repository can be downloaded/cloned and then installed by running `pip install .` inside the main folder.

## Usage

### Data Preparation and Loading

To utilize `MatFold`, the user has to provide their materials data as a Pandas dataframe and 
a dictionary for initialization: 
`df` and `bulk_dict`.

The dataframe `df` has to contain as a first column the strings of either form `<structureid>` or
`<structureid>:<structuretag>` (where `<structureid>` refers to a bulk ID and `<structuretag>` 
refers to an identifier of a derivative structure). All other columns are optional and are 
retained during the splitting process by default. 

The dictionary `bulk_dict` has to contain `<structureid>` as keys and the corresponding bulk pymatgen
dictionary as values. This dictionary can also be directly created from cif files 
using the convenience function `cifs_to_dict`. The user should ensure that all bulk structures that 
are referred to in the `df` labels are provided in `bulk_dict` (and each string 
specifying `structureid` should match).

During initialization of `MatFold` the user can also pick a random subset of the data by specifying the 
variable `return_frac`. When this value is set to less than 1.0, then the variable 
`always_include_n_elements` can be specified to ensure that materials with a certain number of unique elements 
is always included (*i.e.*, not affected by the `return_frac` sampling). 
For example, `always_include_n_elements=[1, 2]` would ensure that all elemental and binary compounds remain 
in the selected subset of the data.

### Creating Splits with Different Chemical and Structural Holdout Strategies

Once the `MatFold` class is initialized with the material data, the user can choose from various chemical and 
structural holdout strategies when creating their splits. The available splitting options are: 
 - *"index"* (naive random splitting)
 - *"structureid"* (split by parent bulk structure - this is identical to *"index"* for datasets where each entry corresponds to a unique bulk structure)
 - *"composition"*
 - *"chemsys"*
 - *"sgnum"* (Space group number)
 - *"pointgroup"*
 - *"crystalsys"*
 - *"elements"*
 - *"periodictablerows"*
 - *"periodictablegroups"*

Further, the user can analyze the distribution of unique split values and the corresponding 
fraction (prevalence) in the dataset by using the class function `split_statistics`. 
There are several optional variables that the user can specify (full list in the documentation below). 
Most, notably the number of inner and outer splits for nested folding are specified in 
`n_inner_splits` and `n_outer_splits`, respectively. If either of these two value is set to 0, 
then `MatFold` will set them equal to the number of possible split label option (*i.e.*, this corresponds 
to leave-one-out cross-validation).

The user can also create a single leave-one-out split (rather than all possible splits) by utilizing the class 
function `create_loo_split` while specifying a single label that is to be held out in `loo_label` for 
the specified `split_type`.

### Example Use

Below find an example of how running `MatFold` could look like:

```Python3
from MatFold import MatFold
import pandas as pd
import json

df = pd.read_csv('data.csv')  # Ensure that first column contains the correct label format
with open('bulk.json', 'r') as fp:  
    # Ensure all bulk pymatgen dictionaries are contained with the same key as specified in `df`
    bulk_dict = json.load(fp)

# Initialize MatFold and work with 50% of the dataset, but ensure to include all binary compounds
mf = MatFold(df, bulk_dict, return_frac=0.5, always_include_n_elements=[2])
stats = mf.split_statistics('crystalsys')
print(stats)  # print out statistics for the `crystalsys` split strategy
# Create all nested (and non-nested) splits utilizing `crystalsys` with the outer 
# split being leave-one-out and the inner splits being split into 5.
mf.create_splits('crystalsys', n_outer_splits=0, n_inner_splits=5, output_dir='./output/', verbose=True)
# Create a single leave-one-out split where Iron is held out from the dataset
mf.create_loo_split('elements', 'Fe', output_dir='./output/', verbose=True)
```

## Code Documentation

Below find a detailed documentation of all `MatFold` capabilities and description of variables.

### Function `cifs_to_dict`

```python
def cifs_to_dict(directory: str | os.PathLike) -> dict
```

Converts a directory of cif files into a dictionary with keys '<filename>' (of `<filename>.cif`) 
and values 'pymatgen dictionary' (parsed from `<filename>.cif`)

**Arguments**:

- `directory`: Directory where cif files are stored

**Returns**:

Dictionary of cif files with keys '<filename>' (of `<filename>.cif`).
Can be used as input `bulk_df` to `MatFold` class.

### Class `MatFold`

### \_\_init\_\_

```python
def __init__(df: pd.DataFrame,
             bulk_dict: dict,
             return_frac: float = 1.0,
             always_include_n_elements: list | int | None = None,
             cols_to_keep: list | None = None,
             seed: int = 0) -> None
```

MatFold class constructor

**Arguments**:

- `df`: Pandas dataframe with the first column containing strings of either form `<structureid>` or
`<structureid>:<structuretag>` (where <structureid> refers to a bulk ID and <structuretag> refers to
an identifier of a derivative structure). All other columns are optional and may be retained specifying the
`cols_to_keep` parameter described below.
- `bulk_dict`: Dictionary containing <structureid> as keys and the corresponding bulk pymatgen
dictionary as values.
- `return_frac`: The fraction of the df dataset that is utilized during splitting.
Must be larger than 0.0 and equal/less than 1.0 (=100%).
- `always_include_n_elements`: A list of number of elements for which the corresponding materials are
always to be included in the dataset (for cases where `return_frac` < 1.0).
- `cols_to_keep`: List of columns to keep in the splits. If left `None`, then all columns of the
original df are kept.
- `seed`: Seed for selecting random subset of data and splits.


#### from\_json

```python
@classmethod
def from_json(cls,
              df: pd.DataFrame,
              bulk_dict: dict,
              json_file: str | os.PathLike,
              create_splits: bool = True)
```

Reconstruct a `MatFold` class instance, along with its associated splits, from a JSON file previously generated 
by the `create_splits` or `create_loo_split` methods. The same `df` and `bulk_dict` used during
the original split creation must be provided to guarantee that the exact splits are regenerated.

**Arguments**:

- `df`: Pandas dataframe with the first column containing strings of either form `<structureid>` or
`<structureid>:<structuretag>` (where <structureid> refers to a bulk ID and <structuretag> refers to
an identifier of a derivative structure). All other columns are optional and may be retained specifying the
`cols_to_keep` parameter described below.
- `bulk_dict`: Dictionary containing <structureid> as keys and the corresponding bulk pymatgen
dictionary as values.
- `json_file`: Location of JSON file that is created when MatFold is used to generate splits.
- `create_splits`: Whether to create splits with the same json settings

**Returns**:

MatFold class instance


### split\_statistics

```python
def split_statistics(split_type: str) -> dict
```

Analyzes the statistics of the sgnum, pointgroup, crystalsys, chemsys, composition, elements,  periodictablerows, 
and periodictablegroups splits.

**Arguments**:

- `split_type`: String specifying the splitting type

**Returns**:

Dictionary with keys of unique split values and the corresponding fraction of this key being 
represented in the entire dataset.


### create\_splits

```python
def create_splits(split_type: str,
                  n_inner_splits: int = 10,
                  n_outer_splits: int = 10,
                  fraction_upper_limit: float = 1.0,
                  fraction_lower_limit: float = 0.0,
                  keep_n_elements_in_train: list | int | None = None,
                  min_train_test_factor: float | None = None,
                  inner_equals_outer_split_strategy: bool = True,
                  write_base_str: str = 'mf',
                  output_dir: str | os.PathLike | None = None,
                  verbose: bool = False) -> None
```

Creates splits based on `split_type`, `n_inner_splits`, `n_outer_splits` among other specifications 
(cf. full list of function variables). The splits are saved in `output_dir` as csv files named
`<write_base_str>.<split_type>.k<i>_outer.<train/test>.csv` and
`<write_base_str>.<split_type>.k<i>_outer.l<j>_inner.<train/test>.csv` for all outer (index `<i>`) and inner
splits (index `<j>`), respectively. Additionally, a summary of the created splits is saved as
`<write_base_str>.<split_type>.summary.k<n_outer_splits>.l<n_inner_splits>.<self.return_frac>.csv`.
Lastly, a JSON file is saved that stores all relevant class and function variables to recreate the splits
utilizing the class function `from_json` and is named `<write_base_str>.<split_type>.json`.

**Arguments**:

- `split_type`: Defines the type of splitting, must be either "index", "structureid", "composition",
"chemsys", "sgnum", "pointgroup", "crystalsys", "elements", "periodictablerows", or "periodictablegroups"
- `n_inner_splits`: Number of inner splits (for nested k-fold); if set to 0, then `n_inner_splits` is set
equal to the number of inner test possiblities (i.e., each inner test set holds one possibility out
for all possible options)
- `n_outer_splits`: Number of outer splits (k-fold); if set to 0, then `n_outer_splits` is set equal to the
number of test possiblities (i.e., each outer test set holds one possibility out for all possible options)
- `fraction_upper_limit`: If a split possiblity is represented in the dataset with a fraction above
this limit then the corresponding indices will be forced to be in the training set by default.
- `fraction_lower_limit`: If a split possiblity is represented in the dataset with a fraction below
this limit then the corresponding indices will be forced to be in the training set by default.
- `keep_n_elements_in_train`: List of number of elements for which the corresponding materials are kept
in the test set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
- `min_train_test_factor`: Minimum factor that the training set needs to be
larger (for factors greater than 1.0) than the test set.
- `inner_equals_outer_split_strategy`: If true, then the inner splitting strategy used is equal to
the outer splitting strategy, if false, then inner splitting strategy is random (by index).
- `write_base_str`: Beginning string of csv file names of the written splits
- `output_dir`: Directory where the splits are written to
- `verbose`: Whether to print out details during code execution.

**Returns**:

None

### create\_loo\_split

```python
def create_loo_split(split_type: str,
                     loo_label: str,
                     keep_n_elements_in_train: list | int | None = None,
                     write_base_str: str = 'mf',
                     output_dir: str | os.PathLike | None = None,
                     verbose: bool = False) -> None
```

Creates leave-one-out split based on `split_type`, specified `loo_label` and `keep_n_elements_in_train`.
The splits are saved in `output_dir` as csv files named
`<write_base_str>.<split_type>.loo.<loo_label>.<train/test>.csv`. Additionally, a summary of the created split
is saved as `<write_base_str>.<split_type>.summary.loo.<loo_label>.<self.return_frac>.csv`.
Lastly, a JSON file is saved that stores all relevant class and function variables to recreate the splits
utilizing the class function `from_json` and is named `<write_base_str>.<split_type>.loo.<loo_label>.json`.

**Arguments**:

- `split_type`: Defines the type of splitting, must be either "structureid", "composition", "chemsys",
"sgnum", "pointgroup", "crystalsys", "elements", "periodictablerows", or "periodictablegroups".
- `loo_label`: Label specifying which single option is to be left out (i.e., constitute the test set).
This label must be a valid option for the specified `split_type`.
- `keep_n_elements_in_train`: List of number of elements for which the corresponding materials are kept
in the test set by default (i.e., not k-folded). For example, '2' will keep all binaries in the training set.
- `write_base_str`: Beginning string of csv file names of the written splits
- `output_dir`: Directory where the splits are written to
- `verbose`: Whether to print out details during code execution.

**Returns**:

None

