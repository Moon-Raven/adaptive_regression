# Adaptive regression

This project implements an algorithm that performs adaptive regression. The algorithm is suited for real-time embedded applications. Because of that, the code is written in a C-like way for easy re-implementation in C. Methods are based on modified "Potential Based Recursive Clustering" and "General Regression Neural Network" algorithms.

## Prerequisites

The project is developed using python 3. The following python libraries are required:

* numpy (http://www.numpy.org/)
* matplotlib (https://matplotlib.org/)

## Installing and running

pbrc.py and grnn.py are modules that provide core implementation of the algorithm. For examples of use, take a look at files simple_example.py and pid_experiment.py.