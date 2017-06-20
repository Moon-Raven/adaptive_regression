import pbrc
import grnn
import plant

import numpy as np
import matplotlib.pyplot as plt

dim = 2
last_foci_num = 0
foci_num = []
foci_positions = []
foci_ips = []
node_num = []
real_y_t = []
real_y = []
estimated_y = []
x = []
i = 0
foci_appearance_moments = []

def init_logger():
    pass

def collect_data():
    global i, last_foci_num

    new_foci = pbrc.get_foci()
    new_foci_num = len(new_foci)
    foci_num.append(new_foci_num)

    if new_foci_num > last_foci_num:
        for j in range(new_foci_num-last_foci_num):
            foci_appearance_moments.append(i)


    foci_positions.append(np.copy(new_foci))

    data = plant.get_last_data()
    x.append(data['x'])

    if type(data['y']) != type(None):
        real_y_t.append(i)
        real_y.append(data['y'])

    estimated_y.append(grnn.get_last_regression())

    node_num.append(grnn.get_node_num())

    last_foci_num = new_foci_num
    i += 1

def plot_foci_num():
    plt.figure()
    plt.plot(foci_num)
    plt.title("Number of foci")
    plt.xlabel('Time')
    plt.ylabel('Number of foci')
    plt.draw()

def plot_y():
    plt.figure()
    plt.title("Plant output")
    plt.plot(estimated_y, label='Estimated output')
    plt.plot(real_y_t, real_y, 'o', label='Real output')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Output value')
    plt.draw()

def plot_node_num():
    plt.figure()
    plt.plot(node_num)
    plt.title("GRNN node number")
    plt.xlabel('Time')
    plt.ylabel('Number of nodes')
    plt.draw()

def plot_foci_x():
    plt.figure()

    for j in range(last_foci_num):
        x_axis = np.arange(foci_appearance_moments[j], i-1)
        y_axis = np.empty([i-1-foci_appearance_moments[j],dim])
        for k in range(foci_appearance_moments[j], i-1):
            y_axis[k - foci_appearance_moments[j]] = foci_positions[k][j]

        for k in range(dim):
            plt.plot(x_axis,y_axis[:,0], label="focus #{0}_x{1}".format(j, k))

    plt.title("Focus positions")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Focus coordinate value')
    plt.draw()