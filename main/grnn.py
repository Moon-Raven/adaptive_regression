import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# GRNN meta-parameters
SIGMA = 0.8
DIMENSIONS = 5

# ********************* Configuration functions *********************
def set_sigma(new_sigma):
    global SIGMA
    SIGMA = new_sigma

# *** End of configuration functions ***



# ********************* Control *********************

# Global/static variables
nodes = {"x":np.empty((0,DIMENSIONS)), "y": np.empty(0)}

# Appends new node to the existing array of nodes
def add_node(new_x, new_y):
    global nodes
    nodes["x"] = np.append(nodes["x"], [new_x], 0)
    nodes["y"] = np.append(nodes["y"], new_y)
    return nodes

# Returns the regression for a sample x
def get_regression(x):
    global nodes

    # If network doesn't have any nodes yet
    if len(nodes) == 0:
        return 0

    arr_xi = nodes["x"]
    arr_yi = nodes["y"]

    distances = x - arr_xi

    arr_Di = np.sum(distances * distances, 1)
    arr_ex = np.exp(-arr_Di/(2*SIGMA**2))
    
    upper_sum = np.sum(arr_ex * arr_yi)
    lower_sum = np.sum(arr_ex)

    y = upper_sum / lower_sum

    return y

# *** End of control ***


# ********************* Information fetching functions *********************

def get_nodes():
    return nodes

# *** End of information fetching functions ***