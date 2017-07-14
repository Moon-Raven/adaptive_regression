import pbrc
import grnn
import plant
import logger
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

def reset_all():
    pbrc.reset()
    grnn.reset()
    plant.reset()

def test_plant():
    for i in range(100):
        data = plant.get_next_data()
        foci = "x: {0:>30}; y: {1:>15}".format(str(data['x']), str(data['y']))

def test_pbrc():
    N = 100
    dim = 2
    focus_history = np.zeros([dim, N])

    pbrc.set_log_level(1)

    pbrc.add_focus([2,3])

    for i in range(N):
        pbrc.iterate(np.array([2.1, 3.1]))
        foci = pbrc.get_foci()

        if len(foci) != 1:
            print("ERROR FOCI NUM != 1")
            exit()

        f1 = foci[0]
        focus_history[:,i] = f1

    axes = plt.gca()
    axes.set_ylim([1.5,3.5])
    plt.grid(True)
    t = np.arange(0, N)
    plt.plot(focus_history[0,:], 'bo')
    plt.plot(focus_history[1,:], 'ro')
    plt.show()

def test_plant_and_pbrc():
    N = 1000
    dim = 2
    focus_history = np.zeros([dim, N])
    pbrc.set_log_level(0)

    pbrc.add_focus([2,3])
    pbrc.add_focus([4,5])

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']
        pbrc.iterate(x)
        foci = pbrc.get_foci()

        if len(foci) != 2:
            print("ERROR FOCI NUM is " + str(len(foci)))
            exit()

        focus_history[:,i] = foci[0]

    axes = plt.gca()
    plt.grid(True)
    t = np.arange(0, N)
    plt.plot(focus_history[0,:], 'bo')
    plt.plot(focus_history[1,:], 'ro')
    plt.show()

def test_grnn():
    grnn.set_sigma(1.2)
    grnn.add_node(np.array([0,0]), 2)
    grnn.add_node(np.array([5,5]), 3)
    
    x1 = np.linspace(2,3)
    x2 = np.linspace(2,3)
    x = np.stack((x1,x2))

    y = np.empty(x.shape[1])

    print(y)
    for i in range(x.shape[1]):
        new_y = grnn.get_regression(x[:,i])
        print("f({0}) = {1}".format(x[:,i], new_y))
        y[i] = new_y

    plt.plot(y)
    plt.show()

def test_plant_and_grnn_simple():
    N = 1000

    grnn.add_node(plant.f1, plant.get_y(plant.f1))
    grnn.add_node(plant.f2, plant.get_y(plant.f2))

    y = np.empty(N)

    for i in range(N):
        x = plant.get_next_data()['x']        
        estimated_y = grnn.get_regression(x)
        y[i] = estimated_y

    plt.plot(y)
    plt.show()

def test_plant_and_grnn_self_learning():
    N = 1000
    t = np.arange(0, N)

    estimated_y_array = np.empty(N)

    for i in range(N):
        data = plant.get_next_data()

        x = data['x']
        y = data['y']

        if y != None:
            grnn.add_node(x, y)

        estimated_y = grnn.get_regression(x)
        estimated_y_array[i] = estimated_y

    plt.plot(estimated_y_array)
    plt.show()

def test_plant_and_pbrc_ips():
    N = 1000
    dim = 2
    focus_history = np.zeros([dim, N])
    pbrc.set_log_level(0)

    pbrc.add_focus([2,3])
    pbrc.add_focus([4,5])

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']
        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if len(foci_ips) != 2:
            print("ERROR FOCI NUM is " + str(len(foci_ips)))
            exit()

        focus_history[:,i] = foci_ips

    axes = plt.gca()
    plt.grid(True)
    t = np.arange(0, N)
    plt.plot(focus_history[0,:], 'bo')
    plt.plot(focus_history[1,:], 'ro')
    plt.show()

def test_full_1():
    N = 1000
    dim = 2
    pbrc.set_log_level(0)

    estimated_y_array = np.zeros(N)
    
    pbrc.add_focus([2,3])
    pbrc.add_focus([4,5])

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if len(foci_ips) != 2:
            print("ERROR FOCI NUM is " + str(len(foci_ips)))
            exit()

        if y != None:
            grnn.add_node(foci_ips, y)

        print("******************** {0} ********************".format(i))
        print(grnn.get_nodes())

        estimated_y = grnn.get_regression(foci_ips)
        estimated_y_array[i] = estimated_y

    axes = plt.gca()
    plt.grid(True)
    t = np.arange(0, N)
    plt.plot(estimated_y_array)
    plt.show()

def plot_plant(n):
    dim = 2
    x = np.empty([2,n])

    y_t = []
    y = []

    for i in range(n):
        data = plant.get_next_data()
        x[:,i] = data['x']

        print(data['y'])
        if type(data['y']) != type(None):
            y_t.append(i)
            y.append(data['y'])

    for i in range(dim):
        plt.plot(x[i,:])

    plt.plot(y_t, y, 'o')
    plt.show()

def test_5_foci():
    foci_num = 5
    N = 20000
    estimated_y_array = np.zeros(N)
    real_y_t = []
    real_y = []
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)
    for i in range(foci_num):
        pbrc.add_focus(np.array([i,i]))

    
    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            real_y_t.append(i)
            real_y.append(y)
            grnn.add_node(foci_ips, y)

        estimated_y = grnn.get_regression(foci_ips)
        logger.collect_data()

    logger.plot_foci_num()
    logger.plot_y()
    logger.plot_node_num()
    logger.plot_foci_x()
    plt.show()

def plot_grnn_1d(start, end):
    N = 1000
    x = np.linspace(start, end, N)
    y = np.empty(len(x))
    for i in range(N):
        y[i] = grnn.get_regression(x[i])

    plt.plot(x, y)
    plt.title("GRNN output")
    plt.show()

def test_dynamic_grnn():
    x1 = np.array([0])
    y1 = 0
    x2 = np.array([5])
    y2 = 5

    x3 = np.array([2.5, 2.5])
    y3 = -5

    grnn.add_node(x1, y1)
    grnn.add_node(x2, y2)    
    grnn.add_node(x3, y3)  

    plot_grnn_1d(-1, 6)
    #print(grnn.get_regression(np.array([4, 2])))

def test_dynamic_system1():
    N = 4000
    pbrc.set_distance_threshold(0.325)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)
    
    for i in range(5):
        pbrc.add_focus(np.array([i,i]))

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

        estimated_y = grnn.get_regression(foci_ips)
        logger.collect_data()

    logger.plot_foci_num()
    logger.plot_y()
    logger.plot_node_num()
    logger.plot_foci_x()
    plt.show()

def test_dynamic_system_0_foci():
    N = 4000
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)
    plant.set_plant_type_x('5_foci_array')
    plant.set_plant_type_y('linear')

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

        estimated_y = grnn.get_regression(foci_ips)
        logger.collect_data()

    logger.plot_foci_num()
    logger.plot_y()
    logger.plot_node_num()
    logger.plot_foci_x()
    plt.show()

def test_freeze():
    N = 4000
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)

    for i in range(N):

        if i == 2000:
            pbrc.freeze_foci()

        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

        estimated_y = grnn.get_regression(foci_ips)
        logger.collect_data()

    logger.plot_foci_num()
    logger.plot_y()
    #logger.plot_node_num()
    logger.plot_foci_x()
    plt.show()

def get_static_regression(x, iterations_in_place = 20):
    for i in range(iterations_in_place):
        pbrc.iterate(x)

    foci_ips = pbrc.get_foci_ips()
    estimated_y = grnn.get_regression(foci_ips)
    return estimated_y

# Parameters should form a mesh grid
def get_function_estimation(x1, x2):
    xnum = 51

    sh = x1.shape
    estimated_y = np.empty(sh)

    # Get regression at each input vector after spending some time at that vector
    pbrc.freeze_foci()
    for i in range(sh[0]):
        for j in range(sh[1]):
            estimated_y[i,j] = get_static_regression(np.array([x1[i,j], x2[i,j]]))

    pbrc.unfreeze_foci()

    # Fix off-by-one
    estimated_y = estimated_y[:-1, :-1]
    return estimated_y

# Parameters should form a mesh grid
def get_real_function(x1, x2):

    sh = x1.shape
    real_y = np.empty(sh)

    for i in range(sh[0]):
        for j in range(sh[1]):
            real_y[i,j] = plant.get_y(np.array([x1[i,j], x2[i,j]]), False)

    # Fix off-by-one
    real_y = real_y[:-1, :-1]
    return real_y

def get_mesh_grid(x1min, x1max, x2min, x2max, xnum=51):
    x1len = x1max-x1min
    x2len = x2max-x2min
    dx1 = x1len/(xnum-1)
    dx2 = x2len/(xnum-1)
    x1, x2 = np.mgrid[x1min:x1max + dx1:dx1, x2min:x2max + dx2:dx2]
    return x1, x2

def get_real_and_estimated_function(x1min, x1max, x2min, x2max):

    x1, x2 = get_mesh_grid(x1min, x1max, x2min, x2max)

    # Get data
    estimated_y = get_function_estimation(x1, x2)
    real_y = get_real_function(x1, x2)

    return real_y, estimated_y

def plot_estimation(x1min, x1max, x2min, x2max):

    x1, x2 = get_mesh_grid(x1min, x1max, x2min, x2max)
    real_y, estimated_y = get_real_and_estimated_function(x1min, x1max, x2min, x2max)

    # Prepare colours
    levels = MaxNLocator(nbins=100).tick_values(min(estimated_y.min(), real_y.min()), 
                                                max(estimated_y.max(), real_y.max()))
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    
    im = ax0.pcolormesh(x1, x2, estimated_y, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax0)
    ax0.set_title('Estimation of the function')

    im = ax1.pcolormesh(x1, x2, real_y, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax1)
    ax1.set_title('Real function')

    plt.show()

def test_plot():
    N = 4000
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

    plot_estimation(-5,10,-5,10)

def test_peaks():
    N = 10404
    plant.set_y_period(5)
    plant.set_plant_type_x('zigzag')
    plant.set_plant_type_y('peaks')
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.3)
    plant.set_noise_amplitude(10)

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

    plot_estimation(-3, 3, -3, 3)

def get_one_error_peaks(noise_level):
    reset_all()
    N = 10404
    #N = 51*51
    plant.set_y_period(5)
    plant.set_plant_type_x('zigzag')
    plant.set_plant_type_y('peaks')
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.3)
    plant.set_noise_amplitude(noise_level)
    x1min, x1max, x2min, x2max = -3, 3, -3, 3

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

    real_y, estimated_y = get_real_and_estimated_function(x1min, x1max, x2min, x2max)

    errors = np.abs(real_y-estimated_y)

    return errors

def plot_errors_peaks():
    means = []
    noise_amplitudes = list(range(11))
    pp = PdfPages('results.pdf')

    for noise_amplitude in noise_amplitudes:
        errors = get_one_error_peaks(noise_amplitude)
        plt.figure()
        plt.hist(errors.flatten(),alpha=0.75)
        plt.title("Noise amplitude: {0}".format(noise_amplitude))
        plt.xlabel("Absolute error")
        plt.ylabel("Number of occurences")
        plt.grid()
        pp.savefig()

        means.append(np.mean(errors))
        
    # Plot means of errors
    plt.figure()
    plt.plot(noise_amplitudes, means, noise_amplitudes, means, 'go')
    plt.title("Mean absolute errors for different noise amplitudes")
    plt.xlabel("Noise amplitude")
    plt.ylabel("Mean absolute error")
    plt.grid()
    pp.savefig()
    pp.close()

def test_reset():
    print("GRNN node num: {0}".format(grnn.get_node_num()))
    print("Plant counter: {0}".format(plant.counter))
    print("PBRC foci num: {0}".format(len(pbrc.get_foci())))
    print("*"*50)

    for i in range(50):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)
        print("GRNN node num: {0}".format(grnn.get_node_num()))
        print("Plant counter: {0}".format(plant.counter))
        print("PBRC foci num: {0}".format(len(pbrc.get_foci())))
        print("*"*50)

    reset_all()

    print("GRNN node num: {0}".format(grnn.get_node_num()))
    print("Plant counter: {0}".format(plant.counter))
    print("PBRC foci num: {0}".format(len(pbrc.get_foci())))
    print("*"*50)

    for i in range(50):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)
        print("GRNN node num: {0}".format(grnn.get_node_num()))
        print("Plant counter: {0}".format(plant.counter))
        print("PBRC foci num: {0}".format(len(pbrc.get_foci())))
        print("*"*50)

def analyze_grnn():
    grnn.add_node(np.array([111,222,333,444,555]), 666)
    grnn.add_node(np.array([3, 4, 5, 6, 7, 8]), 777)
    print(grnn.get_nodes_x())

def test_dynamic_system_0_foci2():
    N = 4000
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)
    plant.set_plant_type_x('5_foci_array')

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

        estimated_y = grnn.get_regression(foci_ips)
        logger.collect_data()

    logger.plot_to_file('results')

def test_cluster_adding_grnn():
    grnn.add_node([1], 10)
    grnn.add_node([2], 20)
    grnn.add_node([3], 30)

    grnn.add_node([1.1], 5)
    grnn.add_node([1.5], 40)

    grnn.add_node([0.8], 2)
    grnn.add_node([3.25], 3)

    print(grnn.cluster_centres)
    print()
    print(grnn.cumul_cluster_outputs)
    print()
    print(grnn.cluster_occurences)
    print()

def test_cluster_calculation_grnn():
    grnn.add_node([1], 10)
    grnn.add_node([2], 20)
    
    print(grnn.get_regression(1.5))
    grnn.add_node([0.8], -100)
    print(grnn.get_regression(1.5))

def main():
    random.seed(0)
    #test_plant()
    #test_pbrc()
    #test_plant_and_pbrc()
    #test_grnn()
    #test_plant_and_grnn_simple()
    #test_plant_and_grnn_self_learning()
    #test_plant_and_pbrc_ips()
    #test_full_1()
    #plot_plant(1000)
    #test_5_foci()
    #test_dynamic_grnn()
    #test_dynamic_system1()
    #test_dynamic_system_0_foci()
    #test_freeze()
    #test_plot()
    #test_peaks()
    #test_reset()
    #plot_errors_peaks()
    #analyze_grnn()
    #test_cluster_adding_grnn()
    #test_cluster_calculation_grnn()
    test_dynamic_system_0_foci2()

if __name__ == "__main__":
    main()