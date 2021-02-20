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

* Load the python librairies needed for OTEANN (e.g. numpy, pandas, torch...) from the requirements file
  * python -m pip install -r requirements.txt
  
* Extract the subdatasets (required local free spacesize=334Mo)
  * python extract_subdatasets.py

## Evaluation and Training

In order to run this code, you need to:
* Run the [oteann.ipynb](oteann.ipynb) in order to create the ANN model, train it, test it and display the results of the paper.

## Results

OTEANN achieves the following performance:

|orthography| write score| read score |
|-----------|------------|------------|
|    ent    | 99.8 ± 0.1 | 99.8 ± 0.2 |
|    eno    | 00.0 ± 0.0 | 00.0 ± 0.0 |
|    ar     | 83.8 ± 1.4 | 99.4 ± 0.2 |
|    br     | 79.4 ± 1.7 | 77.1 ± 1.1 |
|    de     | 69.1 ± 1.7 | 76.9 ± 1.0 |
|    en     | 36.4 ± 1.3 | 30.6 ± 1.0 |
|    eo     | 99.5 ± 0.3 | 99.7 ± 0.2 |
|    es     | 67.2 ± 1.4 | 85.8 ± 0.7 |
|    fi     | 97.5 ± 0.5 | 92.9 ± 0.7 |
|    fr     | 28.1 ± 1.7 | 78.6 ± 0.8 |
|    fro    | 99.2 ± 0.2 | 89.0 ± 0.9 |
|    it     | 94.6 ± 1.0 | 72.0 ± 1.9 |
|    ko     | 81.9 ± 1.1 | 97.7 ± 0.4 |
|    nl     | 72.6 ± 1.5 | 56.6 ± 1.8 |
|    pt     | 75.2 ± 1.6 | 82.2 ± 1.2 |
|    ru     | 41.9 ± 1.3 | 97.4 ± 0.7 |
|    sh     | 99.1 ± 0.3 | 54.8 ± 3.3 |
|    tr     | 95.4 ± 0.6 | 95.6 ± 0.8 |
|    zh     | 20.4 ± 1.3 | 78.4 ± 1.1 |

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
