import pbrc
import grnn
import plant
import logger

import numpy as np
import matplotlib.pyplot as plt

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
    N = 4000
    estimated_y_array = np.zeros(N)
    real_y_t = []
    real_y = []
    pbrc.set_distance_threshold(0.5)
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
        estimated_y_array[i] = estimated_y
        logger.collect_data()

    logger.plot_foci_num()
    logger.plot_y()
    logger.plot_node_num()
    plt.show()
    #axes = plt.gca()
    #plt.grid(True)
    #t = np.arange(0, N)
    #plt.plot(estimated_y_array)
    #plt.plot(real_y_t, real_y, 'o')
    #plt.show()


def main():
    #test_plant()
    #test_pbrc()
    #test_plant_and_pbrc()
    #test_grnn()
    #test_plant_and_grnn_simple()
    #test_plant_and_grnn_self_learning()
    #test_plant_and_pbrc_ips()
    #test_full_1()
    #plot_plant(1000)
    test_5_foci()

if __name__ == "__main__":
    main()