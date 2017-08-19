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

def get_a(t):
    amin = 0.2
    amax = 0.7
    T = 6000
    f = 1/T
    arange = amax - amin
    s = np.sin(2*np.pi*f*t) + 1
    return (s*arange/2) + amin


    return amin + t * arange/TMAX

def get_b(t):
    bmin = 0.1
    bmax = 0.2
    brange = bmax - bmin

    return bmin + t * brange/TMAX

def r(t):
    tt = t % (2*period)

    if tt < period:
        ret_val = R0
    else:
        ret_val = 0

    return ret_val

def get_error(grnn_sigma, pbrc_lamda, pbrc_distance, grnn_cluster_radius):
    pbrc.set_distance_threshold(pbrc_distance)
    pbrc.set_distance_threshold(pbrc_distance)
    grnn.set_cluster_radius(grnn_cluster_radius)
    grnn.set_sigma(grnn_sigma)

    time_array = np.arange(TMAX)
    abs_errors = np.zeros(int(TMAX/period))

    up = 0
    yp = 0
    ep = 0
    
    P = 0.5*np.eye(2)
    p = np.array([[0],[0]])

    integral = 0
    period_counter = 0

    for i in range(TMAX):
        t = time_array[i]

        a = get_a(t)
        b = get_b(t)

        e = r(t-1) - yp
        u = up + 0.2/B0 * (e - A0 * ep)
        y = a * yp + b * up
        
        # Kalman
        x = np.array([[yp], [up]])
        P = (P - ((P.dot(x).dot(x.T).dot(P)) / (kalman_lam+x.T.dot(P).dot(x))))/kalman_lam
        p = p + P.dot(x).dot(y - x.T.dot(p))

        # Update variables
        yp = y
        up = u
        ep = e
        
        # Calculate error
        abs_error = np.abs(e)
        integral += abs_error

        pbrc.iterate(np.squeeze(p))
        if (t+1) % period == 0:
            foci_ips = pbrc.get_foci_ips()
            real_y = integral
            grnn.add_node(foci_ips, real_y)            
            estimated_y = grnn.get_regression(foci_ips)

            abs_errors[period_counter] = np.abs(estimated_y - real_y)
            period_counter += 1

            integral = 0

    return np.mean(abs_errors)

def simple_experiment():
    pbrc.set_distance_threshold(0.1)
    grnn.set_sigma(0.5)
    logger.set_use_plant(False)

    time_array = np.arange(TMAX)
    y_array = np.empty(TMAX)
    a_array = np.empty(TMAX)
    b_array = np.empty(TMAX)
    real_a_array = np.empty(TMAX)
    real_b_array = np.empty(TMAX)

    up = 0
    yp = 0
    ep = 0
    
    P = 0.5*np.eye(2)
    p = np.array([[0],[0]])

    integral = 0

    for i in range(TMAX):
        t = time_array[i]

        a = get_a(t)
        b = get_b(t)

        e = r(t-1) - yp
        u = up + 0.2/B0 * (e - A0 * ep)
        y = a * yp + b * up
        
        # Kalman
        x = np.array([[yp], [up]])
        P = (P - ((P.dot(x).dot(x.T).dot(P)) / (kalman_lam+x.T.dot(P).dot(x))))/kalman_lam
        p = p + P.dot(x).dot(y - x.T.dot(p))

        # Data logging
        a_array[i] = p[0]
        b_array[i] = p[1]
        y_array[i] = y
        real_a_array[i] = a
        real_b_array[i] = b

        # Update variables
        yp = y
        up = u
        ep = e
        
        # Calculate errors
        abs_error = np.abs(e)
        integral += abs_error

        if (t+1) % period == 0:
            pbrc.iterate(np.squeeze(p))
            foci_ips = pbrc.get_foci_ips()

            real_y = None
            if (t+1) % big_period == 0:
                grnn.add_node(foci_ips, integral)
                real_y = integral

            estimated_y = grnn.get_regression(foci_ips)
            logger.collect_data(real_y)
            integral = 0

    #logger.plot_foci_num()
    logger.plot_y()
    #logger.plot_node_num()
    #logger.plot_foci_x()

    if False:
        plt.figure()
        plt.plot(y_array, label = 'y')
        plt.grid()
        plt.title('System output')
        plt.draw()

    if False:
        plt.figure()
        plt.title('Estimation of A')
        plt.plot(a_array, label = 'estimated a')
        plt.plot(real_a_array, 'r', label = 'real a')
        plt.legend()
        plt.grid()
        plt.draw()

    if False:
        plt.figure()
        plt.title('Estimation of B')
        plt.plot(b_array, label = 'estimated b')
        plt.plot(real_b_array, 'r', label = 'real b')
        plt.legend()
        plt.grid()
        plt.draw()

    plt.show()

def main():
    print(get_error(0.5, 0.6, 0.1, 0.3))
    simple_experiment(0.5, 0.6, 0.1, 0.3)

if __name__ == "__main__":
    main()