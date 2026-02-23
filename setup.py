from setuptools import setup, find_packages

setup(
    name='grapes',
    version='0.1.0',
    description='Generalized Radial Profiles for gas in halos',
    packages=find_packages(),
    package_data={'grapes': ['data/*']},
    scripts=[],
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
    ],
    python_requires='>=3.6',
)
