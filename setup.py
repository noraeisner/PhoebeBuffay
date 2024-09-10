from setuptools import setup, find_packages, Extension
import numpy as np

setup(
    name='phoebebuffay',
    version='0.1.0',
    description='Emulators for the PHOEBE binary star modeling software',
    author='Nora Eisner, Cole Johnston, Valentina Tardugno',
    author_email='colej@mpa-garching.mpg.de',
    url='https://github.com/noraeisner/PhoebeBuffay',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.25.0',
        'torch>=2.1.0',
        'matplotlib>=3.7.0',
        'sklearn>=1.2.0',
        'pandas>=2.2.0',
    ],
    classifiers=[
        # Choose appropriate classifiers from:
        # https://pypi.org/classifiers/
    ],
    python_requires='>=3.7',
)