# Metabolomics Analysis

This repository contains the Jupyter notebook implementing the metabolomics analysis workflow.

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

## Usage

Start the local Jupyter notebook server by running:

```sh
jupyter notebook
```

The `data_analysis/` and `extract_ms2_spectra/` directories in this repository contains Jupyter notebooks along with the required input files.

### MS2 Similarity Search

The `library_ms2_similarity_search/library_ms2_similarity_search.py` script compares 
the similarity between the ms2 spectra from biological metabolites of interest against user-provided spectral library.

To run this script, you'll need to create and activate a separate `metabolomics-analysis-matchms` environment from the `environment_matchms.yml` file:

```sh
conda env create -f environment_matchms.yml

conda activate metabolomics-analysis-matchms
```

Please refer to the script comments for an example command.

## Testing

To execute the unit tests, run:

```sh
pytest
```
