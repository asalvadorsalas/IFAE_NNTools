# IFAE_NNTools
Neural Network algorithms for analysis searches

Setup
======
For the time being, [jupyter.pic.es](https://jupyter.pic.es/) does not have a working setup by *default*. Here are some steps to setup a local environment and install a kernel.

The kernel is the python instance used to run inside a notebook. If you run over scripts, you don't need ipykernel nor or the kernel installation.

**PLEASE** note that you only need to create the environment and the kernel installation **ONCE**. The environment can be activated again if needed and the kernel will appear as an option in the notebook after a succesful install.

Install/use our centralised kernel
====
1. Setup conda
    
    Conda is the friendly python environment that takes care of installing properly the compatible packages for you.
    
    `source source /nfs/pic.es/user/s/salvador/conda_setup.sh`

2. Activate the centralised environment (you can deactivate it with `source deactivate`)

    While activated, you have all the packages needed to work! Avoid installing more packages, to do that check the next section.
    
    `conda activate /data/at3/scratch/salvador/condaenv_IFAE_NN`
    
3. Install kernel with the setup environment

    This has to be done ONCE.
    
    `python -m ipykernel install --user --name=IFAE_NN` (or any other name)
    
    You can check which kernels you have installed with `jupyter kernelspec list`


Create a kernel from scratch
====

1. Setup conda
    
    Conda is the friendly python environment that takes care of installing properly the compatible packages for you.
    
    `source source /nfs/pic.es/user/s/salvador/conda_setup.sh`

2. Create local conda environment
    
    `conda create --prefix env_NN` (`--prefix` is to create it locally, if you use `--name` the environment be eliminated at the end of the session)

3. Activate it (you can deactivate it with `source deactivate`)

    While activated, you can install packages locally and recover the setup activating the environment again!
    
    `conda activate env_NN`

3. Install packages in environment

    Packages needed are tensorflow-gpu to run the gpu, pandas and tables to open the input, scikit-learn for BDTs and other tools, matplotlib for plotting tools, ipykernel to install the kernel and feather to save and load pandas efficiently.
    
    This can be tricky as the conda version may change and the compatibilities are well set. It has happened that installing new version of tensorflow-gpu with conda does not include the necessary cuda packages to work with GPU's
    
    `conda install tensorflow-gpu=2.2.0` (requires confirmation)
    
    `conda install keras=2.4.3` (requires confirmation)
    
    `pip install ipykernel pandas tables joblib scikit-learn matplotlib feather-format`

4. Install kernel with the created environment

    This has to be done ONCE.
    
    `python -m ipykernel install --user --name=kernel_NN` (or any other name)
    
    You can check which kernels you have installed with `jupyter kernelspec list`

5. Refresh the browser and make sure you see the new kernel option inside the jupyter nootebook or when creating a new one

_if there is anything wrong with this setup tell me_
  
Download the branch
======

Go to your desired workspace and,

`git clone https://github.com/asalvadorsalas/IFAE_NNTools.git`

The notebooks in the jupyter folder are documented working examples for different analysis. Feel to copy the jupyter/Train_tqX/Tutorial.ipynb in your working directory. Feel free to execute all the cells but there is risk of using too much memory (keras has memory leaks when training) if you train and evaluate in the kernel instance. The "user" has to be changed in the first cell.