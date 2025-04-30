from main import MatFold
import json

with open('test.json', 'r') as fp:
    cifs = json.load(fp)
# mfc = MatFold.from_json(pd.read_csv('./test.csv', header=None), cifs, './output/mf.elements.loo.Fe.json')

mfc = MatFold(pd.read_csv('./test.csv', header=None), cifs,
              return_frac=0.5, always_include_n_elements=None)
stats = mfc.split_statistics('crystalsys')
print(stats)
mfc.create_splits("crystalsys", n_outer_splits=0, n_inner_splits=0,
                  fraction_upper_limit=0.8, keep_n_elements_in_train=2, min_train_test_factor=None,
                  output_dir='./output/', verbose=True)
mfc.create_loo_split("elements", 'Fe', keep_n_elements_in_train=None,
                     output_dir='./output/', verbose=True)
