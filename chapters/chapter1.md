## Python environment set-up

This chapter will describe how to set up a computer in order to run provided scripts.

First of all, ensure that you have installed **conda** package manager. If you have not done so, please follows the steps described [here](https://conda.io/docs/user-guide/install/download.html).

> Note: Python version required to run the provided scripts is **3.6.2** and for TensorFlow it is **1.3.0**.

Further, clone all the scripts that are available in [this repository](https://bitbucket.org/tomasbernotas/machine-learning-using-tensorflow) and then using command line switch to the location where the cloned files are. Here you should see `environment.yml` file that specifies the name of the conda environment and packages that will be installed. To create the environment run the following command:

```bash
conda env create -f environment.yml
```

Additional information on _conda_ package manager and commands can be found [here](https://conda.io/docs/).

This tutorial does not require GPU and for that reason, only CPU version of TensorFlow is installed. If you wish to install the GPU version, first follow the steps regarding CUDA set up on [TensorFlow website](https://www.tensorflow.org/install/), then modify `environment.yml` by replacing `tensorflow` with `tensorflow-gpu` and then run the command shown above.

> On the date of writing the [website](https://www.tensorflow.org/install/) has a misleading version number for **cuDNN** that should be **v6** rather than **v5.1**.

After environment setup is complete, to activate it on _Unix_ systems run the following:

```bash
source activate tf_tutorial
```

or if you run _Windows_ use:

```bash
activate tf_tutorial
```

Here **tf\_tutorial** is a default environment name that is given in `environment.yml` file in the repository, thus if you have replaced it, replace it also in the commands above.

While in the environment, you can run any python script and it will use only packages that are available in it. This tutorial is using the following packages:

*   [python](https://www.python.org/)
*   [matplotlib](https://matplotlib.org/)
*   [scikit-learn](http://scikit-learn.org/stable/)
*   [numpy](http://www.numpy.org/)
*   [pandas](http://pandas.pydata.org/)
*   [pip](https://pip.pypa.io/en/stable/)
*   [tensorflow](https://www.tensorflow.org/)

> Note:  It is advisable to create a separate environment and `environment.yml` files for each project. For more information on _conda_ environment management see [here](https://conda.io/docs/commands.html#conda-environment-commands).

[Next chapter](/chapters/chapter2.md) is going give a very brief introduction into TensorFlow, if you wish to return to previous chapter press [here](../README.md).

### Code

*   [environment.yml](/scripts/environment.yml)
