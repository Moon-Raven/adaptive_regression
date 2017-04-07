import numpy as np
import matplotlib.pyplot as plt
import math

# GRNN meta-parameters
SIGMA = 0.8

def print_info():
    print("General Regression Neural Network")
    print("Sigma: " + str(SIGMA))

# Initialize empty array of nodes
def initialize_nodes():
    return {"x":np.empty(0), "y": np.empty(0)}

# Appends new node to the existing array of nodes
def add_node(nodes, new_node):
    nodes["x"] = np.append(nodes["x"], new_node[0])
    nodes["y"] = np.append(nodes["y"], new_node[1])
    return nodes

# Add some test nodes
def add_dummy_nodes(nodes):
    nodes = add_node(nodes, [-2, 5])
    nodes = add_node(nodes, [0, 2])
    nodes = add_node(nodes, [2, 7])
    return nodes

# Returns the regression for a sample x
def get_regression_sample(nodes, x):

    upper_sum = 0
    lower_sum = 0

    for i in range(len(nodes["x"])):
        xi = nodes["x"][i]
        yi = nodes["y"][i]

        distance = x - xi
        Di = np.dot(distance, distance)
        ex = math.exp(-Di/(2*SIGMA**2))

        upper_sum += ex*yi
        lower_sum += ex

    return upper_sum / lower_sum


# Returns the regression for array of input variables x
def get_regression_array(nodes, samples):

    N = len(samples)
    y = np.empty(N)

    for i in range(N):
        y[i] = get_regression_sample(nodes, samples[i])

    return y

# Plots the dependence between input and output variables estimated by the GRNN
def plot_regression(nodes):
    xmin = -5
    xmax = 5
    steps = 10000
    x = np.linspace(xmin, xmax, steps)
    y = get_regression_array(nodes, x)
    plt.plot(x, y)
    plt.show()


def main():
    print_info()

    nodes = initialize_nodes()
    nodes = add_dummy_nodes(nodes)
    plot_regression(nodes)

if __name__ == "__main__":
    main()