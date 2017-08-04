import pbrc
import grnn
import plant
import logger
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

extension = '.pdf'
folder = '../results/Master/'

def set_save_options(image_extension, destination_folder):
    global extension, folder
    extension = image_extension
    folder = destination_folder

def reset_all():
    pbrc.reset()
    grnn.reset()
    plant.reset()
    logger.reset()

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

"""def test_dynamic_system_0_foci2():
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

    logger.plot_to_file('results')"""

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

def plot_linear_input():
    plant.set_plant_type_x('5_foci_array')
    N = 1300
    x = np.empty([N])

    for i in range(N):
        x[i] = plant.get_x(i)[0]

    plt.plot(x)
    plt.grid()
    plt.ylim([-0.5,4.5])
    plt.title('Ulazna funkcija procesa')
    plt.xlabel('Vreme')
    plt.ylabel('Vrednost ulaza za x1 i x2')
    plt.tight_layout()
    plt.savefig(folder+'input.png')

def set_plot_params():
    plt.grid()
    plt.tight_layout()

def plot_foci_num(save = True):
    plt.figure()

    plt.plot(logger.foci_num)
    plt.title("PBRC: Broj fokusa u sistemu")
    plt.xlabel('Vreme')
    plt.ylabel('Broj fokusa')    
    plt.grid()    
    plt.ylim([0, 10])
    plt.tight_layout()

    if save == True:
        plt.savefig(folder+'broj_fokusa.' + extension)
    else:
        plt.show()

def plot_foci_starting_positions(save = True):
    plt.figure()

    foci_pos = np.array(pbrc.original_foci_positions).transpose()        
    foci_x1 = foci_pos[0,:]    
    foci_x2 = foci_pos[1,:]    

    plt.scatter(foci_x1, foci_x2, c='g', s=150)
    plt.title("PBRC: Početni položaji fokusa")
    plt.xlabel('x1')
    plt.ylabel('x2')    
    plt.grid()    
    plt.tight_layout()


    if save == True:
        plt.savefig(folder+'polozaj_fokusa.' + extension)
    else:
        plt.show()

def plot_foci_movement(save = True):
    plt.figure()

    i = logger.i
    foci_appearance_moments = logger.foci_appearance_moments
    foci_positions = logger.foci_positions
    dim = logger.dim
    plt_base_number = dim * 100 + 10

    for j in range(logger.last_foci_num):
        x_axis = np.arange(foci_appearance_moments[j], i-1)
        y_axis = np.empty([i-1-foci_appearance_moments[j],dim])

        for k in range(foci_appearance_moments[j], i-1):
            y_axis[k - foci_appearance_moments[j]] = foci_positions[k][j]

        for k in range(1):        
            plt.plot(x_axis,y_axis[:,0], linewidth=1.5, label="Fokus #{0}_x{1}".format(j, k))

    k = 0
    plt.ylim([-0.1, 4.1])
    plt.title("PBRC: Promene položaja fokusa tokom vremena".format(k))

    if False:            
        plt.legend()

    plt.xlabel('Vreme')
    plt.ylabel('Vrednost x1 koordinate fokusa')
    plt.grid()
    plt.tight_layout()

    
    if save == True:
        plt.savefig(folder+'promene_polozaja_fokusa.' + extension)
    else:
        plt.show()

def plot_node_cluster_num(save = True):
    plt.figure()

    # Draw total node number
    plt.subplot(211)
    plt.plot(logger.total_node_num)
    plt.yticks(np.arange(0, max(logger.total_node_num)+1, 100))
    plt.title("GRNN: Ukupan broj pristiglih nodova")
    plt.xlabel('Vreme')
    plt.ylabel('Broj nodova')
    plt.grid()
    plt.tight_layout()

    # Draw number of clusters
    plt.subplot(212)
    plt.plot(logger.cluster_num)
    plt.title("GRNN: Broj grupa")
    plt.xlabel('Vreme')
    plt.ylabel('Broj grupa')
    plt.ylim([0, max(logger.cluster_num)+1])
    plt.grid()
    plt.tight_layout()
    
    if save == True:
        plt.savefig(folder+'broj_grupa_i_nodova.' + extension)
    else:
        plt.show()

def plot_output_comparison(save = True):
    fig = plt.figure()

    fig.set_canvas(plt.gcf().canvas)
    plt.title(u"Poređenje estimiranog i pravog izlaza")
    plt.ylim([-7,48])
    plt.plot(logger.estimated_y, label='Estimiran izlaz')
    plt.plot(logger.real_y_t, logger.real_y, 'o', label='Pravi izlaz')
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.12), ncol=2)
    plt.xlabel('Vreme')
    plt.ylabel('Vrednost izlaza')
    plt.grid()
    plt.tight_layout()

    if save == True:
        plt.savefig(folder+'poredjenje_izlaza.' + extension)
    else:
        plt.show()

def plot_multiple(input_type = '5_foci_array'):
    N = 4000
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)
    plant.set_plant_type_x(input_type)
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

    # Configure default tick label sizes
    #mpl.rcParams['xtick.labelsize'] = 20
    #mpl.rcParams['ytick.labelsize'] = 20
    #mpl.rc('font', family='Arial')
    # style file is located in C:\Users\My_User_Name\.matplotlib\stylelib
    plt.style.use('master')

    plot_foci_num()
    plot_foci_starting_positions()
    plot_foci_movement()
    plot_node_cluster_num()
    plot_output_comparison()
    plot_input_x1(input_type)

def plot_grnn_example():
    plt.style.use('master')
    grnn.set_sigma(0.5)
    xses = np.array([1, 2, 3, 4, 5])
    yses = np.array([2, 2, 0, 4, 3])

    for i in range(xses.shape[0]):
        grnn.add_node(np.array([xses[i]]), yses[i])

    x = np.linspace(0,6, 1000)
    y = np.empty(x.shape[0])

    for i in range(x.shape[0]):
        new_y = grnn.get_regression(x[i])
        y[i] = new_y

    plt.title('Primer GRNN-a')
    plt.xlabel('Ulaz')
    plt.ylabel('Izlaz')
    plt.scatter(xses, yses, c='r', s=150, label='Podaci za obuku')
    plt.plot(x, y, label='Regresorska površ')
    ax = plt.subplot(111)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.12), ncol=2)
    plt.grid()
    plt.xlim(0, 6)
    plt.tight_layout()
    #plt.show()

    plt.savefig(folder+'grnn_primer1.' + extension)

def plot_grnn_example_multi_sigma():
    plt.style.use('master')
    sigmas = [0.03, 0.5, 2, 10]

    for sigma in sigmas:
        grnn.reset()
        grnn.set_sigma(sigma)
        xses = np.array([1, 2, 3, 4, 5])
        yses = np.array([2, 2, 0, 4, 3])

        for i in range(xses.shape[0]):
            grnn.add_node(np.array([xses[i]]), yses[i])

        x = np.linspace(0,6, 1000)
        y = np.empty(x.shape[0])

        for i in range(x.shape[0]):
            new_y = grnn.get_regression(x[i])
            y[i] = new_y
        plt.plot(x, y, label='σ = {0}'.format(sigma), lw=1.5)

    plt.title('Primer GRNN-a za različite vrednosti σ')
    plt.xlabel('Ulaz')
    plt.ylabel('Izlaz')
    plt.scatter(xses, yses, c='r', s=150, label='Podaci za obuku')    
    ax = plt.subplot(111)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.4), ncol=1)
    plt.grid()
    plt.xlim(0, 6)
    plt.tight_layout()
    #plt.show()

    plt.savefig(folder+'grnn_razne_sigme.' + extension)

def plot_grouping():
    plt.style.use('master')
    plt.title('Primer grupisanja odbiraka GRNN-a')
    c = [np.array([2,5]), np.array([4, 1.5]), np.array([6, 5])]
    centers = np.array(c)
    plt.ylim(0,7)
    plt.xlim(0,8)

    # Plot centers
    plt.scatter(centers[:,0], centers[:,1], c=(1,0.2,0.2,0.5), s=22000, marker='o')

    # Plot areas
    plt.scatter(centers[:,0], centers[:,1], c='r', s=200, marker='x')

    nodes = np.array([[1.5, 5.5],
                      [2.5, 5.2],
                      [3.5, 2.0],
                      [5.5, 4.9],
                      [6.0, 6.0],
                      [6.5, 4.3]])

    # Plot nodes
    plt.scatter(nodes[:,0], nodes[:,1], c='c', s=500, marker='o')

    #Anotacija centara klastera
    ax = plt.subplot(111)
    for i, txt in enumerate(list(centers)):
        ax.annotate('g' + str(i+1), (centers[i,0]-0.07, centers[i, 1]-0.35))

    # Anotacija odbiraka
    y_vrednosti = [4,2,5,1,3,4]
    for i, txt in enumerate(list(nodes)):
        ax.annotate(y_vrednosti[i], (nodes[i,0]-0.065, nodes[i, 1]-0.09))

    #plt.scatter(xses, yses, c='r', s=150, label='Podaci za obuku')    
    #ax = plt.subplot(111)
    #ax.legend(loc='upper right', bbox_to_anchor=(1, 0.4), ncol=1)
    plt.grid()
    #plt.xlim(0, 6)
    plt.tight_layout()
    #plt.show()

    plt.savefig(folder+'grnn_grupisanje.' + extension)

def plot_input(input_type, limit = 1300):    
    plant.set_plant_type_x(input_type)
    N = limit
    x = np.empty([2, N])

    for i in range(N):
        x[:, i] = plant.get_x(i)

    mn = np.min(x)
    mx = np.max(x)
    rnge = mx-mn
    grace = 0.1*rnge
    ylimits = [mn - grace, mx + grace]

    plt.subplot(211)
    plt.title('Ulazna veličina x1')
    plt.ylim(ylimits)
    plt.grid()
    plt.xlabel('Vreme')
    plt.ylabel('x1')
    plt.plot(x[0,:])

    plt.subplot(212)
    plt.title('Ulazna veličina x2')
    plt.ylim(ylimits)
    plt.grid()
    plt.xlabel('Vreme')
    plt.ylabel('x2')
    plt.plot(x[1,:])

    plt.tight_layout()
    plt.show()
   
    #plt.savefig(folder+'input.png')

def plot_input_x1(input_type, save = True, N = 1300):
    plant.set_plant_type_x(input_type)
    x = np.empty([N])

    for i in range(N):
        x[i] = plant.get_x(i)[0]

    mn = np.min(x)
    mx = np.max(x)
    rnge = mx-mn
    grace = 0.1*rnge
    ylimits = [mn - grace, mx + grace]

    plt.figure()
    plt.plot(x)
    plt.grid()
    plt.ylim(ylimits)
    plt.title('Ulazna funkcija procesa')
    plt.xlabel('Vreme')
    plt.ylabel('Vrednost ulaza za x1 i x2')
    plt.tight_layout()

    if save == True:
        plt.savefig(folder+'ulazni_signal.' + extension)
    else:
        plt.show()

def plot_error_histogram(save = True):
    N = 4000
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)
    plant.set_plant_type_x('5_foci_array')

    real_y_array = np.empty([N])
    estimated_y_array = np.empty([N])

    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

        estimated_y = grnn.get_regression(foci_ips)
        estimated_y_array[i] = estimated_y
        real_y_array[i] = plant.get_y(x, False)

        logger.collect_data()

    abs_errors = np.abs(estimated_y_array - real_y_array)

    plt.figure()
    plt.hist(abs_errors, alpha=0.75)
    plt.title("Raspodela apsolutne greške")
    plt.xlabel("Apsolutna greška")
    plt.ylabel("Broj odbiraka")
    plt.grid()

    if save == True:
        plt.savefig(folder+'raspodela_greske.' + extension)
    else:
        plt.show()

def plot_different_errors(save = True):
    N = 4000
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)
    plant.set_plant_type_x('5_foci_array')

    real_y_array = np.empty([N])
    estimated_y_array = np.empty([N])

    noise_amplitudes = [0, 5, 10, 15, 20]
    mean_absolute_errors = np.empty(len(noise_amplitudes))

    for j in range(len(noise_amplitudes)):
        noise_amplitude = noise_amplitudes[j]
        reset_all()
        plant.set_noise_amplitude(noise_amplitude)

        for i in range(N):
            data = plant.get_next_data()
            x = data['x']    
            y = data['y']

            pbrc.iterate(x)
            foci_ips = pbrc.get_foci_ips()

            if y != None:
                grnn.add_node(foci_ips, y)

            estimated_y = grnn.get_regression(foci_ips)
            estimated_y_array[i] = estimated_y
            real_y_array[i] = plant.get_y(x, False)

            logger.collect_data()

        abs_errors = np.abs(estimated_y_array - real_y_array)
        mean_absolute_errors[j] = np.mean(abs_errors)

    plt.figure()
    plt.plot(noise_amplitudes, mean_absolute_errors, 
             noise_amplitudes, mean_absolute_errors, 'go', markersize=10)
    plt.title("Srednja apsolutna greška u zavisnosti od šuma")
    plt.xlabel("Amplituda šuma")
    plt.ylabel("Srednja apsolutna greška")
    plt.grid()

    if save == True:
        plt.savefig(folder+'greske_sum.' + extension)
    else:
        plt.show()

def plot_many_lambda(save = True):
    lambdas = [0.1, 0.6, 0.9, 1]
    coeff_num = 10
    x = np.linspace(0,10, 100)

    for lam in lambdas:
        y = [lam** xx for xx in x]
        plt.plot(x, y, linewidth=2.5, label = 'λ = {0}'.format(lam))

    plt.title('Koeficijenti odbiraka za različite vrednosti λ')
    plt.xlabel('Starost odbirka')
    plt.ylabel('Koeficijent odbirka')
    plt.ylim([-0.1, 1.1])
    plt.xticks(np.arange(0,10))
    plt.legend()
    #plt.scatter(xses, yses, c='r', s=150, label='Podaci za obuku')    
    #ax = plt.subplot(111)
    #ax.legend(loc='upper right', bbox_to_anchor=(1, 0.4), ncol=1)
    plt.grid()
    plt.tight_layout()

    if save == True:
        plt.savefig(folder+'lambda_koeficijenti.' + extension)
    else:
        plt.show()

def test_dynamic_system_0_foci2():
    N = 4000
    plant.set_noise_amplitude(20)
    pbrc.set_distance_threshold(0.05)
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

    #logger.plot_foci_num()
    logger.plot_y()
    #logger.plot_node_num()
    #logger.plot_foci_x()
    plt.show()

def plot_different_lambda_whole(save = True):
    N = 4000
    lambdas = [0.999, 0.99, 0.5]
    plt.figure()
    for lam in reversed(lambdas):
        reset_all()
        plant.set_noise_amplitude(0)
        pbrc.set_distance_threshold(0.5)
        pbrc.set_log_level(0)
        pbrc.set_lambda(lam)
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
        plt.plot(logger.estimated_y, linewidth=2, label='λ = {0}'.format(lam))

    plt.grid()
    ax = plt.subplot(111)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.3), ncol=1)
    plt.title('Estimacija izlaza za različite vrednosti λ')
    plt.xlabel('Vreme')
    plt.ylabel('Vrednost izlaza')

    if save == True:
        plt.savefig(folder+'lambda_rezultati.' + extension)
    else:
        plt.show()

def plot_foci_num_different_d(save = True):
    N = 4000
    ds = [2, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
    foci_nums = []
    plt.figure()

    for d in ds:
        reset_all()
        plant.set_noise_amplitude(0)
        pbrc.set_distance_threshold(d)
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
        foci_nums.append(pbrc.num_of_foci)

    plt.grid()
    plt.plot(ds, foci_nums)
    plt.plot(ds, foci_nums, 'o', markersize=10)
    #ax = plt.subplot(111)
    #ax.legend(loc='upper right', bbox_to_anchor=(1, 0.3), ncol=1)
    plt.title('Broj fokusa u sistemu za različite vrednosti d')
    plt.xlabel('d')
    plt.ylabel('Broj fokusa')

    if save == True:
        plt.savefig(folder+'broj_fokusa_od_d.' + extension)
    else:
        plt.show()

def plot_outputs_different_d(save = True):
    N = 4000
    ds = [0.05, 2]
    colours = ['b', 'r']
    plt.figure()

    for j in range(len(ds)):
        reset_all()
        plant.set_noise_amplitude(40)
        pbrc.set_distance_threshold(ds[j])
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
        plt.plot(logger.estimated_y, linewidth=1.5, color = colours[j], label='d = {0}'.format(ds[j]))

    plt.grid()
    ax = plt.subplot(111)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.3), ncol=1)
    plt.title('Estimacija izlaza za različite vrednosti d')
    plt.xlabel('Vreme')
    plt.ylabel('Vrednost izlaza')

    if save == True:
        plt.savefig(folder+'izlaz_razno_d.' + extension)
    else:
        plt.show()

def plot_cluster_num_different_r(save = True):
    N = 4000
    rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    cluster_nums = []
    plt.figure()

    for r in rs:
        reset_all()
        plant.set_noise_amplitude(0)
        pbrc.set_distance_threshold(0.5)
        pbrc.set_log_level(0)
        grnn.set_sigma(0.5)
        grnn.set_cluster_radius(r)
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
        cluster_nums.append(grnn.get_cluster_num())

    plt.grid()
    plt.plot(rs, cluster_nums)
    plt.plot(rs, cluster_nums, 'o', markersize=10)
    #ax = plt.subplot(111)
    #ax.legend(loc='upper right', bbox_to_anchor=(1, 0.3), ncol=1)
    plt.title('Broj grupa u GRNN-u za različite vrednosti r')
    plt.xlabel('r')
    plt.ylabel('Broj grupa')

    if save == True:
        plt.savefig(folder+'broj_grupa_od_r.' + extension)
    else:
        plt.show()

def plot_outputs_different_r(save = True):
    N = 4000
    rs = [0.1, 0.7, 1.0, 1.7]    
    plt.figure()

    for j in range(len(rs)):
        reset_all()
        plant.set_noise_amplitude(20)
        pbrc.set_distance_threshold(0.5)
        pbrc.set_log_level(0)
        grnn.set_sigma(0.5)
        grnn.set_cluster_radius(rs[j])
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
        plt.plot(logger.estimated_y, label='r = {0}'.format(rs[j]))

    plt.grid()
    ax = plt.subplot(111)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.024), ncol=2)
    plt.title('Estimacija izlaza za različite vrednosti r')
    plt.xlabel('Vreme')
    plt.ylabel('Vrednost izlaza')

    if save == True:
        plt.savefig(folder+'izlaz_razno_r.' + extension)
    else:
        plt.show()

def plot_outputs_different_sigma(save = True):
    N = 4000
    sigmas = [0.1, 0.7, 2]    
    plt.figure()

    for j in range(len(sigmas)):
        reset_all()
        plant.set_noise_amplitude(30)
        pbrc.set_distance_threshold(0.5)
        pbrc.set_log_level(0)
        grnn.set_sigma(sigmas[j])
        grnn.set_cluster_radius(0.3)
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
        plt.plot(logger.estimated_y, linewidth=1.0, label='σ = {0}'.format(sigmas[j]))

    plt.grid()
    ax = plt.subplot(111)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.024), ncol=2)
    plt.title('Estimacija izlaza za različite vrednosti σ, r=0.3')
    plt.xlabel('Vreme')
    plt.ylabel('Vrednost izlaza')

    if save == True:
        plt.savefig(folder+'izlaz_razno_sigma_veliko_r.' + extension)
    else:
        plt.show()

def main():
    random.seed(0)
    set_save_options('pdf', '../results/Master/')
    plt.style.use('master')

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
    #test_dynamic_system_0_foci()
    #plot_linear_input()
    #plot_multiple()
    #plot_grnn_example()
    #plot_grnn_example_multi_sigma()
    #plot_grouping()
    #plot_input('sqrt')
    #plot_multiple()
    #plot_error_histogram()
    #plot_different_errors()
    #plot_many_lambda()
    #test_dynamic_system_0_foci2()
    #plot_different_lambda_whole()
    #plot_foci_num_different_d()
    #plot_outputs_different_d()
    #plot_cluster_num_different_r()
    plot_outputs_different_r()
    #plot_outputs_different_sigma(True)

if __name__ == "__main__":
    main()