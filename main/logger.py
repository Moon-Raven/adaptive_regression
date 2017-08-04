import pbrc
import grnn
import plant

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

dim = 2
last_foci_num = 0
foci_num = []
foci_positions = []
foci_ips = []
cluster_num = []
total_node_num = []
real_y_t = []
real_y = []
estimated_y = []
x = []
i = 0
foci_appearance_moments = []

def reset():
    global dim, last_foci_num, foci_num, foci_positions, foci_ips, cluster_num,\
           total_node_num, real_y_t, real_y, estimated_y, x, i, foci_appearance_moments
    dim = 2
    last_foci_num = 0
    foci_num = []
    foci_positions = []
    foci_ips = []
    cluster_num = []
    total_node_num = []
    real_y_t = []
    real_y = []
    estimated_y = []
    x = []
    i = 0
    foci_appearance_moments = []

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

    cluster_num.append(grnn.get_cluster_num())
    total_node_num.append(grnn.get_total_node_num())

    last_foci_num = new_foci_num
    i += 1

def plot_foci_num(draw=True):
    plt.figure()
    plt.plot(foci_num)
    plt.title("PBRC: Number of foci")
    plt.xlabel('Time')
    plt.ylabel('Number of foci')

    if draw == True:
        plt.draw()

def plot_y(draw=True):
    fig = plt.figure()
    fig.set_canvas(plt.gcf().canvas)
    plt.title("Estimation of plant output")
    plt.plot(estimated_y, label='Estimated output')
    plt.plot(real_y_t, real_y, 'o', label='Real output')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Plant output value')
    plt.grid()

    if draw == True:
        plt.draw()

def plot_node_num(draw=True):
    plt.figure()
    
    # Draw total node number
    plt.subplot(211)
    plt.plot(total_node_num)
    plt.title("GRNN: Total number of nodes")
    plt.xlabel('Time')
    plt.ylabel('Total number of nodes')
    plt.grid()

    # Draw number of clusters
    plt.subplot(212)
    plt.plot(cluster_num)
    plt.title("GRNN: Number of clusters")
    plt.xlabel('Time')
    plt.ylabel('Number of clusters')
    plt.grid()
    
    plt.tight_layout()

    if draw == True:
        plt.draw()

def plot_foci_x(legend = False, draw=True):
    plt.figure()

    plt_base_number = dim * 100 + 10

    for j in range(last_foci_num):
        x_axis = np.arange(foci_appearance_moments[j], i-1)
        y_axis = np.empty([i-1-foci_appearance_moments[j],dim])

        for k in range(foci_appearance_moments[j], i-1):
            y_axis[k - foci_appearance_moments[j]] = foci_positions[k][j]

        for k in range(dim):
            plt.subplot(plt_base_number+k+1)            
            plt.plot(x_axis,y_axis[:,0], label="focus #{0}_x{1}".format(j, k))

    for k in range(dim):
        plt.subplot(plt_base_number+k+1)        
        plt.title("PBRC: Foci positions, dimension #{0}".format(k))

        if legend == True:            
            plt.legend()

        plt.xlabel('Time')
        plt.ylabel('Focus coordinate value')
        plt.grid()

    plt.tight_layout()

    if draw == True:
        plt.draw()

def plot_to_file(file_name):
    pp = PdfPages(file_name + '.pdf')
    print("Saving plots to file " + file_name + "...")

    plot_y(False)
    pp.savefig()    

    plot_node_num(False)
    pp.savefig()

    plot_foci_num(False)
    pp.savefig()

    plot_foci_x(draw=False)
    pp.savefig()

    pp.close()