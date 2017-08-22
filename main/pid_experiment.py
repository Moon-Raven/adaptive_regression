import matplotlib.pyplot as plt
import numpy as np

import pbrc
import grnn
import logger

A0 = 0.7
B0 = 0.2
R0 = 10
kalman_lam = 0.96

minute = 60
TMAX = 3600*5
period = minute
big_period = 10 * period

extension = '.pdf'
folder = '../results/Master/'

def set_save_options(image_extension, destination_folder):
    global extension, folder
    extension = image_extension
    folder = destination_folder

# Get a parameter for process transfer function
def get_a(t):
    amin = 0.2
    amax = 0.7
    T = 6000
    f = 1/T
    arange = amax - amin
    s = np.sin(2*np.pi*f*t) + 1
    return (s*arange/2) + amin

# Get b parameter for process trasnfer function
def get_b(t):
    bmin = 0.1
    bmax = 0.2
    brange = bmax - bmin
    return bmin + t * brange/TMAX

# Get reference signal
def r(t):
    tt = t % (2*period)

    if tt < period:
        ret_val = R0
    else:
        ret_val = 0

    return ret_val

# Return mean aboslute error of algorithm's performance for given meta-parameters
def get_error(grnn_sigma, pbrc_lamda, pbrc_distance, grnn_cluster_radius):
    pbrc.set_distance_threshold(pbrc_distance)
    pbrc.set_distance_threshold(pbrc_distance)
    grnn.set_cluster_radius(grnn_cluster_radius)
    grnn.set_sigma(grnn_sigma)

    time_array = np.arange(TMAX)
    abs_errors = np.zeros(int(TMAX/period))

    # Set noise amplitude and filter coefficient
    noise_sigma = 0.1
    filter_alpha = 0.75

    up = 0
    y_filtered_p = 0
    u_filtered_p = 0
    y_noised_p = 0
    yp = 0
    ep = 0
    
    P = 0.5*np.eye(2)
    p = np.array([[0],[0]])

    integral = 0
    period_counter = 0

    for i in range(TMAX):
        t = time_array[i]

        # Fetch new process transfer function parameters
        a = get_a(t)
        b = get_b(t)

        # Regulate and simulate process
        e = r(t-1) - y_noised_p
        u = up + 0.2/B0 * (e - A0 * ep)
        y = a * yp + b * up
        
        # Simulate noise in feedback loop
        noise = noise_sigma * np.random.randn()
        y_noised = y + noise

        # Filter y and u for recursive least squares estimation
        y_filtered = filter_alpha * y_filtered_p + (1-filter_alpha) * y_noised
        u_filtered = filter_alpha * u_filtered_p + (1-filter_alpha) * u

        # Recursive least squares estimation
        x = np.array([[y_filtered_p], [u_filtered_p]])
        P = (P - ((P.dot(x).dot(x.T).dot(P)) / (kalman_lam+x.T.dot(P).dot(x))))/kalman_lam
        p = p + P.dot(x).dot(y_filtered - x.T.dot(p))

        # Update variables
        yp = y
        y_noised_p = y_noised
        up = u
        ep = e
        y_filtered_p = y_filtered
        u_filtered_p = u_filtered
        
        # Calculate error
        abs_error = np.abs(e)
        integral += abs_error

        pbrc.iterate(np.squeeze(p))

        # Check if one period of reference signal has passed
        if (t+1) % period == 0:
            foci_ips = pbrc.get_foci_ips()
            real_y = integral

            # Check if 10 periods of reference signal has passed
            if (t+1) % big_period == 0:
               grnn.add_node(foci_ips, real_y)        

            estimated_y = grnn.get_regression(foci_ips)

            abs_errors[period_counter] = np.abs(estimated_y - real_y)
            period_counter += 1

            integral = 0

    return np.mean(abs_errors)

# Rune the experiment with given meta-parameters
def simple_experiment(grnn_sigma, pbrc_lamda, pbrc_distance, grnn_cluster_radius,
                      plot_results = False, save_plots = False, plot_estimations = False,
                      plot_estimated_integral = False, plot_logger_results = False,
                      plot_system_output = False):
    
    pbrc.reset()
    grnn.reset()
    logger.reset()

    pbrc.set_distance_threshold(pbrc_distance)
    pbrc.set_distance_threshold(pbrc_distance)
    grnn.set_cluster_radius(grnn_cluster_radius)
    grnn.set_sigma(grnn_sigma)
    logger.set_use_plant(False)

    # Set noise amplitude and filter coefficient
    noise_sigma = 0.15
    filter_alpha = 0.75

    # Initialize logging data
    time_array = np.arange(TMAX)
    y_array = np.empty(TMAX)
    y_noised_array = np.empty(TMAX)
    y_filtered_array = np.empty(TMAX)
    a_array = np.empty(TMAX)
    b_array = np.empty(TMAX)
    real_a_array = np.empty(TMAX)
    real_b_array = np.empty(TMAX)
    estimated_integral_array = np.empty(TMAX)
    real_integral_array_t = []
    real_integral_array = []

    up = 0
    yp = 0
    y_filtered_p = 0
    u_filtered_p = 0
    y_noised_p = 0
    ep = 0
    
    P = 0.5*np.eye(2)
    pp = np.array([[0], [0]])

    integral = 0

    for i in range(TMAX):
        t = time_array[i]

        # Fetch new process transfer function parameters
        a = get_a(t)
        b = get_b(t)
        
        # Regulate and simulate process
        e = r(t-1) - y_noised_p
        u = up + 0.2/B0 * (e - A0 * ep)
        y = a * yp + b * up

        # Simulate noise in feedback loop
        noise = noise_sigma * np.random.randn()
        y_noised = y + noise

        # Filter y and u for recursive least squares estimation
        y_filtered = filter_alpha * y_filtered_p + (1-filter_alpha) * y_noised
        u_filtered = filter_alpha * u_filtered_p + (1-filter_alpha) * u

        # Recursive least squares estimation
        x = np.array([[y_filtered_p], [u_filtered_p]])
        P = (P - ((P.dot(x).dot(x.T).dot(P)) / (kalman_lam+x.T.dot(P).dot(x))))/kalman_lam
        p = pp + P.dot(x).dot(y_filtered - x.T.dot(pp))

        # Filter results to reduce noise
        filter_coeff = 0.0 # filter off
        p = filter_coeff * pp + (1-filter_coeff) * p

        # Log data
        a_array[i] = p[0]
        b_array[i] = p[1]
        y_noised_array[i] = y_noised
        y_filtered_array[i] = y_filtered
        y_array[i] = y
        real_a_array[i] = a
        real_b_array[i] = b

        # Update variables
        yp = y
        y_noised_p = y_noised
        up = u
        ep = e
        y_filtered_p = y_filtered
        u_filtered_p = u_filtered
        pp = p

        # Calculate errors
        abs_error = np.abs(e)
        integral += abs_error

        pbrc.iterate(np.squeeze(p))
        foci_ips = pbrc.get_foci_ips()

        # Check if one period of reference signal has passed
        if (t+1) % period == 0:        
            
            real_y = integral 
      
            # Check if 10 periods of reference have passed
            if (t+1) % big_period == 0:
               grnn.add_node(foci_ips, real_y)
               data_for_logger = real_y  
            else:
                real_integral_array.append(real_y)
                real_integral_array_t.append(t)
                data_for_logger = None

            estimated_integral = grnn.get_regression(foci_ips)
            logger.collect_data(data_for_logger)
            integral = 0
        else:
            estimated_integral = grnn.get_regression(foci_ips)
            logger.collect_data(None)

        # Log resulting estimation
        estimated_integral_array[t] = estimated_integral

    # Draw plots if needed
    if plot_results == True:

        if plot_logger_results == True:
            logger.plot_foci_num()
            logger.plot_y()
            logger.plot_node_num()
            logger.plot_foci_x()

        if plot_estimated_integral == True:
            plt.figure()
            plt.grid()
            plt.title('Estimacija integrala greške')
            plt.xlabel('Vreme')
            plt.ylabel('Integral greške')
            plt.plot(estimated_integral_array[599:], label = 'Estimacija')
            plt.plot(real_integral_array_t, real_integral_array, 'o',
                     color = (0.7,0,0,0.75),label = 'Stvarna vrednost')
            plt.plot(logger.real_y_t, logger.real_y, 'o',
                     markersize = 14, label = 'Podaci za obuku')
            
            plt.legend()

            if save_plots == True:
                plt.savefig(folder+'estimacja_integrala.' + extension)
            else:
                plt.show()

        # Unnecessary
        if False:
            plt.figure()
            plt.plot(y_noised_array, label = 'Noised measurements')
            plt.plot(y_filtered_array, label = 'Filtered measurements')
            plt.grid()
            plt.legend()
            plt.title('Filtering')
            plt.draw()

        if plot_system_output == True:
            plt.figure()
            plt.plot(y_array, label = 'y')
            plt.grid()
            plt.title('System output')
            plt.draw()

        if plot_estimations == True:
            plt.figure()
            plt.title('Estimacija parametra a')
            plt.plot(a_array, label = 'Estimacija')
            plt.plot(real_a_array, 'r', label = 'Stvarna vrednost')
            plt.xlabel('Vreme')
            plt.ylabel('a')
            ax = plt.subplot(111)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.024), ncol=2)
            plt.grid()

            if save_plots == True:
                plt.savefig(folder+'estimacija_a.' + extension)
            else:
                plt.show()

            plt.figure()
            plt.title('Estimacija parametra b')
            plt.plot(b_array, label = 'Estimacija')
            plt.plot(real_b_array, 'r', label = 'Stvarna vrednost')
            plt.xlabel('Vreme')
            plt.ylabel('b')
            ax = plt.subplot(111)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.024), ncol=2)
            plt.grid()

            if save_plots == True:
                plt.savefig(folder+'estimacija_b.' + extension)
            else:
                plt.show()

        if save_plots == False:
            plt.show()

# Find optimal parameters for algorithm using grid search
def find_optimal_parameters(output_file_name = 'results.txt'):
    #Order of parameters:
    # grnn_sigma, pbrc_lamda, pbrc_distance, grnn_cluster_radius

    min_sigma, max_sigma = 0.05, 1
    #min_lambda, max_lambda = 0.4, 0.999 # Better specify exactly
    min_focus_distance, max_focus_distance = 0.05, 0.5
    min_cluster_radius, max_cluster_radius = 0.05, 0.5

    steps = 3

    sigma_a = np.linspace(min_sigma, max_sigma, steps)
    #lambda_a = np.linspace(min_lambda, max_lambda, steps) # Better specify exactly
    lambda_a = np.array([0.5, 0.9, 0.99])
    distance_a = np.linspace(min_focus_distance, max_focus_distance, steps)
    radius_a = np.linspace(min_cluster_radius, max_cluster_radius, steps)
    
    # Prepare grid of all combinations
    search_space = np.stack(np.meshgrid(sigma_a, lambda_a, distance_a, radius_a), -1).reshape(-1, 4)

    N = search_space.shape[0]
    results = np.empty(N)

    # Test all combinations
    for i in range(N):
        x = search_space[i]
        results[i] = get_error(x[0], x[1], x[2], x[3])
        print("{0} -> {1}".format(x, results[i]))

    # Write results to file
    f = open(output_file_name, 'w')
    for i in range(N):
        f.write("{0} -> {1}\n".format(search_space[i], results[i]))

    min_ind = np.argmin(results)
    f.write("Min: {0} -> {1}".format(search_space[min_ind], results[min_ind]))
    f.close()

# Plot a and b transfer function parameters
def plot_a_b(save = True):

    a_array = np.empty(TMAX)
    b_array = np.empty(TMAX)

    for i in range(TMAX):
        a_array[i] = get_a(i)
        b_array[i] = get_b(i)

    plt.subplot(2,1,1)
    plt.title('Vrednost parametra a')
    plt.xlabel('Vreme')
    plt.ylabel('a')
    plt.grid()
    plt.plot(a_array)

    plt.subplot(2,1,2)
    plt.title('Vrednost parametra b')
    plt.xlabel('Vreme')
    plt.ylabel('b')
    plt.grid()
    plt.plot(b_array)

    plt.tight_layout()

    if save == True:
        plt.savefig(folder+'a_b.' + extension)
    else:
        plt.show()

# Plot reference input signal
def plot_input(save = True):
    r_array = np.empty(TMAX)

    for i in range(TMAX):
        r_array[i] = r(i)

    plt.title('Vrednost ulazne reference')
    plt.xlabel('Vreme')
    plt.ylabel('r')
    plt.grid()
    plt.plot(r_array[0:400])
    mn, mx = np.min(r_array), np.max(r_array)
    rangee = mx - mn
    grace = 0.1
    plt.ylim(mn - grace*rangee, mx + grace * rangee)

    if save == True:
        plt.savefig(folder+'ulazna_referenca.' + extension)
    else:
        plt.show()

# Plot estimations a and b transfer function parameters
def plot_estimations_a_b(save = True):
    simple_experiment(0.05, 0.99, 0.275, 0.05, True, save)

# Plot the algorithm results
def plot_estimated_integral(save_arg = True):
    simple_experiment(0.05, 0.99, 0.275, 0.05, plot_results = True,
                      save_plots = save_arg, plot_estimated_integral = True)

# Make all plots
def make_plots():
    set_save_options('pdf', '../results/Master/')
    #plt.style.use('master')
    plot_a_b()
    plot_input()
    plot_estimations_a_b()
    plot_estimated_integral()

# Different examples - uncomment and run program to try them
def main():
    #print(get_error(0.5, 0.5, 0.5, 0.5))

    #simple_experiment( 0.05 , 0.5  , 0.5  , 0.05, plot_results=True,
    #                 plot_estimated_integral=True)

    #simple_experiment(0.05 , 0.9 ,  0.05 , 0.05, plot_results=True,
    #                 plot_estimated_integral=True)

    #simple_experiment(0.05 , 0.99 , 0.05 , 0.05, plot_results=True,
    #                 plot_estimated_integral=True)

    simple_experiment(0.05 ,  0.99  , 0.275,  0.05, plot_results=True,
                     plot_estimated_integral=True)

    #find_optimal_parameters()
    
    #make_plots()

if __name__ == "__main__":
    main()