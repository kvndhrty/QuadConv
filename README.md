# QuadConv: Quadrature-Based Convolutions with Applications to Non-Uniform PDE Data Compression

### [arXiv](https://arxiv.org/abs/2211.05151)

Kevin Doherty*, [Cooper Simpson*](https://rs-coop.github.io/), Stephen Becker, Alireza Doostan

Submitted to [Computer Methods in Applied Mechanics and Engineering](https://www.sciencedirect.com/journal/computer-methods-in-applied-mechanics-and-engineering)

## Abstract
We present a new convolution layer for deep learning architectures which we call QuadConv -- an approximation to continuous convolution via quadrature. Our operator is developed explicitly for use on non-uniform, mesh-based data, and accomplishes this by learning a continuous kernel that can be sampled at arbitrary locations. Moreover, the construction of our operator admits an efficient implementation which we detail and construct. In the setting of compressing data arising from partial differential equation (PDE) simulations, we show that QuadConv can match the performance of standard discrete convolutions on uniform grid data by comparing a QuadConv autoencoder (QCAE) to a standard convolutional autoencoder (CAE). Further, we show that the QCAE can maintain this accuracy even on non-uniform data.

## License & Citation
All source code is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full text.

Our work can be cited using the entry in `CITATION`.

## Usage

### Repository Structure
- `core`: Model architectures, data loading, core operators, and utilities.
- `data`: Data folders
- `experiments`: Experiment configuration files
  - `template.yaml`: Detailed experiment template
- `job_scripts`: HPC job submission scripts
- `lightning_logs`: Experiment logs
- `notebooks`: Various Jupyter Notebooks
- `py_scripts`: Various python scripts
- `main.py`: Model training and testing script

### Environment Setup
The file `environment.yaml` contains a list of dependencies, and it can be used to generate an anaconda environment with the following command:
```console
conda create --file=environment.yaml
```
which will install all necessary packages in the conda environment `QuadConv`. It might be faster in some cases to just install the packages individually. We would recommend installing the PyTorch, Nvidia, and PYG packages together, followed by the default packages, and then the pip packages.

For local development, it is easiest to install `core` as a pip package in editable mode using the following command from within the top level of this repository:
```console
pip install -e .
```
The main experiment script can still be run without doing this, but the notebooks will be non-functional.

### Data Acquisition
To obtain the datasets used in our paper...

### Running Experiments
Use the following command to run an experiment:
```console
python main.py --experiment <path/to/YAML/file/in/experiments>
```
If `logger` is set to `True` in the YAML config file, then the results of this experiment will be saved to `lightning_logs/<path/to/YAML/file/in/experiments>`.

To visualize the logging results saved to `lightning_logs/` using tensorboard run the following command:
```console
tensorboard --logdir=lightning_logs/
```
