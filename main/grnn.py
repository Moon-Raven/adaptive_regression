import numpy as np
import matplotlib.pyplot as plt
import math

from timeit import default_timer as timer

# GRNN meta-parameters
SIGMA = 0.8

# ********************* Configuration functions *********************
def set_sigma(new_sigma):
    global SIGMA
    SIGMA = new_sigma

def reset():
    global nodes, max_dim, last_regression
    nodes = {"x":[], "y": np.empty(0)}
    last_regression = None   

# *** End of configuration functions ***



# ********************* Control *********************

# Global/static variables
nodes_x = np.empty((0, 0))
nodes_y = np.empty(0)

last_regression = None

# Adds more dimensions
def add_dimensions(new_dimensions):
    global nodes_x

    old_dimensions = nodes_x.shape[1]

    # Check if the dimensions should be increased (not decreased)
    if new_dimensions <= old_dimensions:
        print("Error in add_dimensions: new dimensions <= old dimensions. Exiting...")
        exit()

    # Expand existing nodes with additional dimensions, fill missing data with zeros
    dimensions_to_add = new_dimensions - old_dimensions    
    nodes_x = np.concatenate((nodes_x, np.zeros([nodes_x.shape[0], dimensions_to_add])), axis=1)

# Appends new node to the existing array of nodes
def add_node(new_x, new_y):
    global nodes_x, nodes_y

    # Check if input vector has increased in number of dimensions
    dims = nodes_x.shape
    if len(new_x) > dims[1]:
        add_dimensions(len(new_x))

    # Add new data
    nodes_x = np.concatenate((nodes_x, [new_x]), 0)
    nodes_y = np.concatenate((nodes_y, [new_y]))

# Returns the regression for a sample x
def get_regression(x):
    global last_regression

    # Fix some weird errors
    if not hasattr(x, '__len__'):
        x = np.array([x])

    node_num = len(nodes_x)

    # If network doesn't have any nodes yet
    if node_num == 0:
        return 0

    # Check if input vector has increased in number of dimensions
    if len(x) > nodes_x.shape[1]:
        add_dimensions(len(x))

    # Calculate estimated output for given input x
    distances = x - nodes_x

    arr_Di = np.sum(distances * distances, 1)
    arr_ex = np.exp(-arr_Di/(2*SIGMA**2))
    
    upper_sum = np.sum(arr_ex * nodes_y)
    lower_sum = np.sum(arr_ex)

    y = upper_sum / lower_sum
    last_regression = y

    return y

# *** End of control ***


# ********************* Information fetching functions *********************

def get_nodes():
    return {"x":nodes_x,"y":nodes_y}

def get_nodes_x():
    return nodes_x

def get_nodes_y():
    return get_nodes_y

def get_last_regression():
    return last_regression

def get_node_num():
    return nodes_y.shape[0]

# *** End of information fetching functions ***