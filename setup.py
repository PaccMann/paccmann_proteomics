"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ''
if os.path.exists('README.md'):
    with open('README.md') as fp:
        LONG_DESCRIPTION = fp.read()

setup(
    name='paccmann_proteomics',
    version='0.0.1',
    description='PyTorch implementations for protein language modeling',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Modestas Filipavicius, Matteo Manica, Jannis Born, Joris Cadow',
    author_email=(
        'mfilipav@gmail.com, drugilsberg@gmail.com, dow@zurich.ibm.com, jab@zurich.ibm.com'
    ),
    url='https://github.com/PaccMann/paccmann_proteomics',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'torch>=1.3', 
        'torchvision', 
        'transformers==3.0.2', 
        'tokenizers==0.8.0',
        'scikit-learn==0.23.2',
        'seaborn',
        'tensorboardX',
        'loguru',
        'seqeval'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages('.')
)
