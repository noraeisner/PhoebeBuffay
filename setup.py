from setuptools import setup, find_packages
import numpy as np

packages = find_packages(where='src')

setup(
    name='buffay',
    version='0.1.0',
    description='Emulators for the PHOEBE binary star modeling software',
    author='Nora Eisner, Cole Johnston, Valentina Tardugno',
    author_email='colej@mpa-garching.mpg.de',
    url='https://github.com/noraeisner/PhoebeBuffay',
    packages=packages,
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.25.0',
        'torch>=2.1.0',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.2.0',
        'pandas>=2.2.0',
    ],
    classifiers=[
        # Choose appropriate classifiers from:
        # https://pypi.org/classifiers/
    ],
    python_requires='>=3.7',
)
