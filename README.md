# Metabolomics Analysis

This repository contains the Jupyter notebook implementing the metabolomics analysis workflow

## Installation

To install all the required dependencies, you need to have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed.

If you don't have the `conda-forge` channel added to conda (run `conda config --show channels` to see your current list of channels), then run:

```sh
conda config --append channels conda-forge
```

Create and activate the `metabolomics-analysis` environment from the `environment.yml` file:

```sh
conda env create -f environment.yml

conda activate metabolomics-analysis
```

### Usage

Start the local Jupyter notebook server by running:

```sh
jupyter notebook
```

The `data_analysis/` directory in this repository contains its own Jupyter notebook along with the required input files.

### Testing

To execute the unit tests, run:

```sh
pytest
```
