from glob import glob
from pathlib import Path
from setuptools import setup, find_packages

setup(
    name='grapes',
    version='0.1.0',
    description='Generalized Radial Profiles for gas in halos',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Sunil Simha',
    url='https://github.com/SunilSimha/grapes',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    packages=find_packages(),
    package_data={'grapes': ['data/*']},
    scripts=glob('bin/*'),
    install_requires=[
        'numpy>=2.0.0',
        'scipy>=1.11.0',
        'astropy>=6.0.0',
        'matplotlib>=3.8.0',
    ],
    python_requires='>=3.8',
)
