# Machine Learning using TensorFlow

## Introduction
The purpose of this tutorial is to be a practical guide in building Neural Networks rather than an in-depth explanation of the ever-changing universe of the Deep Learning. This means that tutorial assumes that a reader is familiar with basic concepts and nomenclature of the Deep Learning. However, for those who did not read hundreds of pages of blogs and books dedicated to the Deep Learning and those who desire more theoretical/mathematical explanations, the tutorial will try to give one or more links to appropriate websites.

This tutorial will use [Python](https://www.python.org/) as the main language with, as the title suggests,  [TensorFlow](https://www.tensorflow.org/). Therefore, it is also assumed that the reader is familiar with at least basic  Python syntax.  At the time of writing the latest Python version is **3.6.2** and for TensorFlow it is **1.3.0**. Explanation how to set up the environment will be provided in the next chapter, so sit tight.

All code presented in this tutorial is available in the form of the scripts. Each script is self-contained and has comments that supposed to guide the reader through the code (if it does not, please, let me know). Scripts were written with an average user in mind thus the scripts omit many advanced features that might speed-up the code but at the same time make code harder to understand. This code also can be used as a template for whatever reader might want to do with it, so feel free to play around.

This tutorial consists of the following sections:
 + [*Python environment set-up*](environment.md) chapter is a guide to a user on how to install all necessary tools in order to run the provided scripts
(feel free to skip this if you know about **conda** and **yml** files).
 + [*Introduction to TensorFlow*](tensorflow_intro.md) - gives a description of basic building blocks and concepts that are utilised for the model building in TensorFlow.
 + [*Logistic Regression*](logistic_regression.md) - we use the *MNIST* data set and find which picture belongs to which number/category.
(Bonus, we build a classifier that tells if given picture is a cat or not.)
 + [*Linear Regression*](linear_regression.md) - we show how to fit a straight line to a data.
 + [*Nonlinear Regression*](nonlinear_regression.md) - we fit a nonlinear equation to a data
 + [*Brief introduction to Recurrent Neural Network*]() chapter gives a very brief explanation why we might need to use the Recurrent Neural Networks and what options are available in TensorFlow out of the box.
 + [*Static and Dynamic Recurrent Neural Network*]() - gives a high-level overview of differences between two types of networks and the ways how to use them.
 + [*PhasedLSTM vs. Vanilla LSTM*]() - describes what is a *PhasedLSTM* and how it can be used.
 + [*Deep Architecture Networks*]() - shows how multiple layers of different types can be combined together.  

So, if you still feel positive and want to continue, let's [continue](environment.md).