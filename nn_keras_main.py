# Andrew Graziano
# CSE-4309-001
from nn_keras import *


# When you test your code, you can change the directory and
# dataset values to reflect:
# - where the dataset directory is located on your machine.
# - which dataset to use for training and testing your model.

directory = "/Users/andrewgraziano/Desktop/hw4"
dataset = "pendigits_string"
# dataset = "yeast"
# dataset = "satellite_string"


# When you test your code, you can select the hyperparameters you want to use 
# by modifying the next lines
layers = 4
units_per_layer = 50
epochs = 20

# Here we call your function.
nn_keras(directory, dataset, layers, units_per_layer, epochs)
