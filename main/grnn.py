import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

from timeit import default_timer as timer

# GRNN meta-parameters
SIGMA = 0.8

# ********************* Configuration functions *********************
def set_sigma(new_sigma):
    global SIGMA
    SIGMA = new_sigma

#def set_dimensions(new_dimensions):
    #global DIMENSIONS
    #DIMENSIONS = new_dimensions

# *** End of configuration functions ***



# ********************* Control *********************

# Global/static variables
nodes = {"x":[], "y": np.empty(0)}
max_dim = 0
last_regression = None

# Appends new node to the existing array of nodes
def add_node(new_x, new_y):
    global nodes, max_dim
    nodes["x"].append(new_x)
    nodes["y"] = np.append(nodes["y"], new_y)
    if len(new_x) > max_dim:
        max_dim = len(new_x)



# Returns the regression for a sample x
#@profile
def get_regression(x):
    t1 = timer()
    global nodes
    global last_regression

    if not hasattr(x, '__len__'):
        x = np.array([x])

    node_num = len(nodes["x"])

    # If network doesn't have any nodes yet
    if node_num == 0:
        return 0

    if len(x) < max_dim:
        print("GRNN: len(x) < max_dim; Exiting...")
        exit()

    
    arr_yi = nodes["y"]

    distances = np.zeros([node_num, max_dim])

    t2 = timer()
    for i in range(node_num):
        x_i = nodes['x'][i]
        
        #for j in range(len(x_i)):
        #    distances[i][j] = x[j] - x_i[j]

        ll = len(x_i)
        distances[i][0:ll] = x[0:ll] - x_i[0:ll]
    t3 = timer()

    #print(distances)
    arr_Di = np.sum(distances * distances, 1)
    arr_ex = np.exp(-arr_Di/(2*SIGMA**2))

    t4 = timer()

    upper_sum = np.sum(arr_ex * arr_yi)
    lower_sum = np.sum(arr_ex)

    t5 = timer()
    y = upper_sum / lower_sum
    last_regression = y
    t6 = timer()
    #print("1-2:{0}; 2-3:{1}; 3-4:{2}; 4-5:{3}; 5-6:{4}".format(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5))    
    return y

# *** End of control ***


# ********************* Information fetching functions *********************

def get_nodes():
    return nodes

def get_last_regression():
    return last_regression

def get_node_num():
    return nodes['y'].shape[0]

# *** End of information fetching functions ***