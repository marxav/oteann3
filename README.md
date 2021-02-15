# OTEANN 

This repository is the official implementation of [OTEANN: Estimating the Transparency of Orthographies with an Artificial Neural Network](https://arxiv.org/abs/1912.13321v2).

## Requirements

* Require Ubuntu 18.04, Python 3.6+.

* Download files on your machine
  * git clone https://github.com/marxav/oteann3.git

* Go to the oteann3 main directory
  * cd oteann3 

* Create a virtual environment
  * python3 -m venv oteann 

* Activate virtual environment
  * source oteann/bin/activate

* Load the python librairies needed for GIPFA (e.g. numpy, pandas, torch...) from the requirements file
  * python -m pip install -r requirements.txt
  
* Extract the subdatasets (required local free spacesize=334Mo)
  * python extract_subdatasets.py

## Evaluation and Training

In order to run this code, you need to:
* Run the [oteann.ipynb](oteann.ipynb) in order to create the ANN model, train it, test it and display the results of the paper.

## Results

OTEANN achieves the following performance:

### 

|-----------|------------|------------|
|orthography| write score| read score |
|-----------|------------|------------|
|    ent    | 98.6 ± 1.2 | 99.4 ± 0.3 |
|    eno    | 00.0 ± 0.0 | 00.0 ± 0.0 |
|-----------|------------|------------|
|    ar     | 79.4 ± 1.3 | 98.0 ± 0.4 |
|    br     | 72.6 ± 1.8 | 69.8 ± 1.8 |
|    de     | 57.9 ± 2.3 | 64.8 ± 2.1 |
|    en     | 27.0 ± 1.3 | 25.8 ± 1.5 |
|    eo     | 98.1 ± 0.7 | 99.1 ± 0.4 |
|    es     | 61.0 ± 1.4 | 82.8 ± 1.4 |
|    fi     | 96.0 ± 0.7 | 87.4 ± 0.8 |
|    fr     | 22.8 ± 1.6 | 71.2 ± 2.6 |
|    fro    | 97.7 ± 0.4 | 87.1 ± 0.9 |
|    it     | 92.2 ± 1.1 | 65.7 ± 1.3 |
|    ko     | 77.5 ± 1.0 | 94.0 ± 0.9 |
|    nl     | 61.9 ± 1.5 | 47.6 ± 2.0 |
|    pt     | 66.1 ± 1.4 | 72.2 ± 1.4 |
|    ru     | 33.1 ± 1.9 | 94.4 ± 0.4 |
|    sh     | 98.0 ± 0.6 | 44.8 ± 2.6 |
|    tr     | 93.8 ± 0.7 | 88.1 ± 2.9 |
|    zh     | 05.4 ± 1.0 | 52.5 ± 1.6 |
|-----------|------------|------------|

## License

OTEANN uses minGTPT (https://github.com/karpathy/minGPT) which is released under the MIT licence.

## Citation
This code was used for the following paper:
```bibtex
@misc{marjou2020oteann,
      title={OTEANN: Estimating the Transparency of Orthographies with an Artificial Neural Network}, 
      author={Xavier Marjou},
      year={2020},
      eprint={1912.13321},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
