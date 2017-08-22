import numpy as np
import matplotlib.pyplot as plt
import math

# GRNN meta-parameters
SIGMA = 0.8
CLUSTER_RADIUS = 0.3

# ***** Configuration functions *****

def set_sigma(new_sigma):
    global SIGMA
    SIGMA = new_sigma

def set_cluster_radius(new_radius):
    global CLUSTER_RADIUS
    CLUSTER_RADIUS = new_radius

def reset():
    global cluster_centres, cumul_cluster_outputs, cluster_occurences
    global last_regression

    cluster_centres = np.empty((0, 0))
    cumul_cluster_outputs = np.empty(0)
    cluster_occurences = np.empty(0)

    last_regression = None   


# ***** Control functions*****

# Global/static variables
cluster_centres = np.empty((0, 0))
cumul_cluster_outputs = np.empty(0)
cluster_occurences = np.empty(0)

last_regression = None

# Adds more dimensions
def add_dimensions(new_dimensions):
    global cluster_centres

    old_dimensions = cluster_centres.shape[1]

    # Check if the dimensions should be increased (not decreased)
    if new_dimensions <= old_dimensions:
        print("Error in add_dimensions: new dimensions <= old dimensions. Exiting...")
        exit()

    # Expand existing nodes with additional dimensions, fill missing data with zeros
    dimensions_to_add = new_dimensions - old_dimensions    
    cluster_centres = np.concatenate(
        (cluster_centres, np.zeros([cluster_centres.shape[0], dimensions_to_add])),
        axis=1)

# Calculates the Euclidean distnace between two given vectors
def get_distance(x, y):
    difference = x - y
    powered = np.power(difference, 2)
    summed = np.sum(powered)
    distance = np.sqrt(summed)

    return distance

# Fetches index of the nearest cluster to the given vector
def find_nearest_cluster_index(x):
    difference = x - cluster_centres
    powered = np.power(difference, 2)
    summed = np.sum(powered, 1)
    distances = np.sqrt(summed)
    index = np.argmin(distances)
    return index

# Add a new cluster with given center
def add_new_cluster(new_center, new_output):
    global cluster_centres, cumul_cluster_outputs, cluster_occurences

    cluster_centres = np.concatenate((cluster_centres, [new_center]), 0)
    cumul_cluster_outputs = np.concatenate((cumul_cluster_outputs, [new_output]))
    cluster_occurences = np.concatenate((cluster_occurences, [1]))

# Updates cluster with given index
def update_cluster(cluster_index, new_y):
    global cluster_centres, cumul_cluster_outputs, cluster_occurences

    i = cluster_index
    cumul_cluster_outputs[i] += new_y
    cluster_occurences[i] += 1

# Appends new node to the existing array of nodes
def add_node(new_x, new_y):
    global cluster_centres, cumul_cluster_outputs, cluster_occurences

    # Check if input vector has increased in number of dimensions
    dims = cluster_centres.shape
    if len(new_x) > dims[1]:
        add_dimensions(len(new_x))

    # If this is the first node, make a new cluster
    if len(cumul_cluster_outputs) == 0:
        add_new_cluster(new_x, new_y)
        return

    # Find distance to nearest cluster
    nearest_cluster_index = find_nearest_cluster_index(new_x)
    distance = get_distance(new_x, cluster_centres[nearest_cluster_index, :])

    # If nearest cluster is too far, make a new cluster
    if distance > CLUSTER_RADIUS:
        add_new_cluster(new_x, new_y)

    # Otherwise update the nearest cluster
    else:
        update_cluster(nearest_cluster_index, new_y)

# Returns the regression for a sample x
def get_regression(x):
    global last_regression

    # Fix some weird errors
    if not hasattr(x, '__len__'):
        x = np.array([x])

    node_num = len(cumul_cluster_outputs)

    # If network doesn't have any nodes yet, return zero
    if node_num == 0:
        return 0

    # Check if input vector has increased in number of dimensions
    if len(x) > cluster_centres.shape[1]:
        add_dimensions(len(x))

    # Calculate estimated output for given input x
    distances = x - cluster_centres

    arr_Di = np.sum(distances * distances, 1)
    arr_ex = np.exp(-arr_Di/(2*SIGMA**2))
    
    upper_sum = np.sum(arr_ex * cumul_cluster_outputs)
    lower_sum = np.sum(arr_ex * cluster_occurences)

    y = upper_sum / lower_sum
    last_regression = y

    return y


# ***** Information fetching functions *****

def get_last_regression():
    return last_regression

def get_cluster_num():
    return cumul_cluster_outputs.shape[0]

def get_total_node_num():
    return np.sum(cluster_occurences)
    
def get_cluster_centers():
    return cluster_centres