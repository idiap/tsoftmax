# _t_-softmax with tnet

The following code uses the concept of _tnet_. Instead of training the networks from scratch as proposed in (1), only the last layers of a neural network trained with softmax are re-trained. This makes it easier to test the effectiveness of t-softmax since only few epochs are needed for training.

 (1) [Niccolò Antonello, Philip N. Garner "A _t_-distribution based operator for enhancing out of distribution robustness of neural network classifiers," IEEE Signal Processing Letters, 2020](https://arxiv.org/abs/2006.05389)

The code is based on the [Pytorch machine learning library](https://github.com/pytorch/pytorch).
If you want to use _t_-softmax in your classifiers/neural networks you can find the modules in `src/tsoftmax.py`.

## Installation

We use [conda](https://docs.conda.io/en/latest/miniconda.html) to create a reproducible environment. 
Run:
```
conda env create -f conda_env.yml
```
to install the dependencies.

Activate the environment using `conda activate tsoftmax`

In `path.sh` change `PYPATH` to the bin directory of the environment you just built e.g.:
```bash
PYPATH="/path_to_conda/miniconda3/envs/tsoftmax/bin"
```
and to the [LSUN](http://lsun.cs.princeton.edu) dataset:
```bash
LSUNPATH="/path_to_lsun/"
```

## Experiments on Fashion-MNIST and k-MNIST

### Training

To train a fmnist classifier run:
```bash
bash run_bw.sh fmnist
```

To train a kmnist classifier run:
```bash
bash run_bw.sh kmnist
```

This will train different models and save them in the folder `models`.

### Getting confidences 

Run evaluation scripts by running:
```bash
bash run_eval_bw.sh
```

### Visualizing results 

Finally you can view the results by running the following script:
```bash
python src/plot_fom.py --arch convnet --data fmnist 
```
and 
```bash
python src/plot_fom.py --arch convnet --data kmnist 
```

## Experiments on CIFAR10

For the CIFAR10 experiments, the procedure is similar: 
  * _Training_: `run.sh` (Note: training each model can be time-demanding. You might want to comment some lines on this script and run it in parallel using different machines)
  * _Confidence measures_: `run_eval.sh`
  * _Visualizing results_: `python plot_fom.py --arch densenet --data cifar10`
