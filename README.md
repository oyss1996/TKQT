# TKQT
This is my Python implementation for the paper:

>A Robust Low-rank Tensor Decomposition and Quantization based Compression Method (TKQT)

that submit to ICDE2024.

## Environment Requirement
The code has been tested under Python 3.7.13. The required packages are as follows:
* numpy == 1.21.6
* tensorly == 0.8.1
* optuna == 3.4.0
* opencv-python = 4.8.0.74
* pywavelets == 1.3.0

The last two are needed for DCT and DWT. If you don't want to implement them, you can comment them out to avoid reporting errors.


## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in run.py (lines 9-13)).
* TKQT with bayesian optimization of MBD Dataset
```
python run.py --method=TK --dataset=MBD --bayes=True
```

* TT Decomposition of PSD Dataset
```
python run.py --method=TT --dataset=PSD
```

* TR Decomposition of Harvard Dataset
```
python run.py --method=TR --dataset=Harvard
```

* DCT and DWT of PlanetLab Dataset
```
python run.py --method=DCT&DWT --dataset=PlanetLab
```

## The searched ranks by Bayesian optimization
| PSD                                                                                                                                                                                                                                                                                                   | MBD                                                                         | PlanetLab                                   | Harvard                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------|
| [3, 13, 26, 23] <br/>[3, 10, 26, 29]<br/>[4, 11, 24, 29]<br/>[3, 13, 26, 25]<br/>[3, 13, 25, 23]<br/>[3, 11, 24, 31]<br/>[3, 13, 26, 23]<br/>[3, 9, 26, 25]<br/>[4, 13, 26, 21]<br/>[3, 19, 26, 17]<br/>[3, 10, 25, 37]<br/>[4, 9, 26, 25]<br/>[3, 9, 25, 27]<br/>[3, 10, 25, 37]<br/>[3, 13, 23, 23] | [19, 5, 15]<br/>[19, 8, 27]<br/>[20, 7, 27]<br/>[17, 8, 31]<br/>[20, 7, 29] | [6, 26, 21]<br/>[6, 26, 26]<br/>[2, 31, 36] | [7, 27, 21]<br/>[5, 29, 31]<br/>[10, 25, 25]<br/>[4, 37, 33]<br/>[9, 29, 27]<br/>[8, 43, 37] |