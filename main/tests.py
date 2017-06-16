import pbrc
import grnn
import plant

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
    pbrc.set_log_level(3)

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

def main():
    #test_plant()
    #test_pbrc()
    test_plant_and_pbrc()

if __name__ == "__main__":
    main()