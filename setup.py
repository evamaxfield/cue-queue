#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
    "wheel>=0.34.2",
]

cdp_converter = [
    "cdp-backend>=3.0.0.dev27",
]

test_requirements = [
    "autoflake>=1.4",
    "black>=21.9b0",
    "codecov>=2.1.12",
    "flake8>=3.9.2",
    "flake8-debugger>=3.2.1",
    "flake8-typing-imports>=1.10.1",
    "isort>=5.9.3",
    "mypy>=0.910",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    "tox>=3.15.2",
    "types-pytz>=2021.1.2",
    "types-requests~=0.1.11",
    *cdp_converter,
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "jupyterlab>=3.1",
    "m2r2>=0.2.7",
    "matplotlib~=3.4",
    "pytest-runner>=5.2",
    "seaborn~=0.11.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "numpy>=1.21",
    "scikit-learn>=0.24.2",
    "segeval>=2.0.11",
    "sentence-transformers>=2.0",
    "tqdm>=4.62.2",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="Jackson Maxfield Brown",
    author_email="jmaxfieldbrown@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    description=(
        "Transcript segmentation using average semantic encodings of "
        "delimiter (cue) sentences."
    ),
    entry_points={
        "console_scripts": [
            "cue-queue-train=cue_queue.bin.train:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="cue_queue",
    name="cue-queue",
    packages=find_packages(
        exclude=[
            "tests",
            "*.tests",
            "*.tests.*",
            "benchmarks",
            "*.benchmarks",
            "*.benchmarks.*",
        ]
    ),
    python_requires=">=3.9",
    setup_requires=setup_requirements,
    test_suite="cue_queue/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/JacksonMaxfield/cue-queue",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.0.0",
    zip_safe=False,
)
