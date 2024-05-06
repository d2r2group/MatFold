from setuptools import setup, find_packages

setup(
    name='MatFold',
    version='0.1.0',
    url='https://github.com/d2r2group/MatFold',
    description='Package for systematic insights into materials discovery models’ performance through standardized '
                'chemical cross-validation protocols',
    author='M. D. Witman and P. Schindler',
    author_email='p.schindler@northeastern.edu',
    install_requires=['numpy', 'pandas', 'scikit-learn>=1.4', 'python>=3.10', 'pymatgen>=2024'],
    packages=find_packages(),
    package_data={'MatFold': ['*.json', '*.csv']},
    include_package_data=True
)
