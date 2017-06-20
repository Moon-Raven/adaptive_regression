import pbrc
import grnn
import plant

import numpy as np
import matplotlib.pyplot as plt

foci_num = []
foci_positions = []
foci_ips = []
node_num = []
real_y_t = []
real_y = []
estimated_y = []
x = []
i = 0

def init_logger():
    pass

def collect_data():
    global i

    new_foci = pbrc.get_foci()
    foci_num.append(len(new_foci))
    foci_positions.append(new_foci)

    data = plant.get_last_data()
    x.append(data['x'])

    if type(data['y']) != type(None):
        real_y_t.append(i)
        real_y.append(data['y'])

    estimated_y.append(grnn.get_last_regression())

    node_num.append(grnn.get_node_num())

    i += 1

def plot_foci_num():
    plt.figure()
    plt.plot(foci_num)
    plt.title("Number of foci")
    plt.draw()

def plot_y():
    plt.figure()
    plt.title("Plant output")
    plt.plot(estimated_y, label='Estimated output')
    plt.plot(real_y_t, real_y, 'o', label='Real output')
    plt.legend()
    plt.draw()

def plot_node_num():
    plt.figure()
    plt.plot(node_num)
    plt.title("GRNN node number")
    plt.draw()