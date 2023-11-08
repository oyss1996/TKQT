# TKQT
This is my Python implementation for the paper:

>A Robust Low-rank Tensor Decomposition and Quantization based Compression Method (TKQT)

that submit to ICDE2024.

## Environment Requirement
The code has been tested under Python 3.7.13. The required packages are as follows:
* numpy == 1.21.6
* tensorly == 0.8.1
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