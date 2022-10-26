from setuptools import setup, find_packages

VERSION = '0.0.11'
DESCRIPTION = 'A package for interaction analysis between moving individuals'
LONG_DESCRIPTION = 'A package for interaction analysis between moving individuals'


# Setting up
setup(
    name="ortega_2022",
    version=VERSION,
    author="Rongxiang Su",
    author_email="<rongxiangsu@ucsb.edu>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['attrs>=22.1.0',
                      'matplotlib>=3.6.0',
                      'numpy>=1.23.3',
                      'shapely>=1.8.4',
                      'pandas>=1.5.0'],
    keywords=['python', 'package'],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)