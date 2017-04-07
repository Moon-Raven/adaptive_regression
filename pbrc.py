import numpy as np
import matplotlib.pyplot as plt
import math

XMAX = 6 * math.pi
DATA_NUM_POINTS = 300;
LAMBDA = 0.6
DATA_TYPE = "sine"
DISTANCE_THRESHOLD = 0.12
LOG_LEVEL = 1

def log(s, level):
    if(level <= LOG_LEVEL):
        print(s)

def generate_dummy_data(N):
    x = np.linspace(0, XMAX, N)
    dummy_data = np.empty([N,1])

    if(DATA_TYPE == "sine"):
        big_sine = np.sin(x)
        small_sine = 0.1 * np.sin(3 * x)   
        dummy_data[:,0] = small_sine + big_sine
    elif(DATA_TYPE == "ramp"):
        half = math.floor(N/2)
        dummy_data[0:half,0] = x[0:half]
        dummy_data[half:,0] = x[half]

    return dummy_data

# Simulates the pbrc algorithm non-recursively
def simulate_pbrc_stupid(input_data):
    dim = input_data.shape[1]
    N = input_data.shape[0]
    
    res = np.empty(N)

    for i in range(N):
        res[i] = information_potential(input_data[i], input_data[0:i])

    return res

# Prepares the foci data structure for plotting
def make_foci_great_again(foci):
    N = len(foci)
    prev_m = 0
    great_foci = []
    for i in range(N):
        f = foci[i]
        m = len(f)

        if(m > prev_m):
            great_foci.append({"x":[], "y":[]})

        for j in range(m):
            great_foci[j]["x"].append(i)
            great_foci[j]["y"].append(f[j])
        prev_m = m

    return great_foci

# Plots the foci data structure created by 'make_foci_great_again'
def plot_great_foci(great_foci):
    i = 0
    for focus in great_foci:
        plt.plot(focus["x"], focus["y"], "o", label = "focus{0:2}".format(i))
        i += 1

# Calculates the distance between two vectors using l2 norm        
def distance(z1, z2):
    difference = z1 - z2;
    return np.sqrt(np.dot(difference, difference))

# Simulates the pbrc algorithm recursively
def simulate_pbrc(input_data):
    dim = input_data.shape[1]
    N = input_data.shape[0]

    foci = [np.array([0])]
    old_foci_distances = [0] * len(foci)
    num_of_foci = len(foci)

    Z = 0
    F = np.zeros(dim)
    S = 0
    z_old1 = np.zeros(dim)
    z_old2 = np.zeros(dim)

    # Logging data
    hist_num_of_foci = np.empty(N)
    hist_current_ip = np.empty(N)
    hist_foci = []
    hist_foci_ips = []

    for i in range(N):
        # Obtain current feature vector
        z = input_data[i]

        # Calculate the new information potential
        Z_new = LAMBDA*Z + 1
        F_new = LAMBDA*F + LAMBDA*Z*(z_old1 - z_old2)
        S_new = LAMBDA*S + 2*LAMBDA*(1-LAMBDA)*np.dot((z-z_old1),F_new)\
               +LAMBDA*(1-LAMBDA)*np.dot(z-z_old1, z-z_old1)*Z_new

        S = S_new
        Z = Z_new
        F = F_new        
        z_old2 = z_old1
        z_old1 = z

        current_ip = 1/(1+S_new)
      
        # Update information potential of all foci
        foci_ips = []

        for j in range(num_of_foci):
            new_distance = (1-LAMBDA)*np.dot(z-foci[j], z-foci[j]) + LAMBDA*old_foci_distances[j]
            foci_ips.append(1/(1+new_distance))                
            old_foci_distances[j] = new_distance
        
        log("#{0:4}: Current ip: {1}, max ip is focus{2:2}: {3:2.2f}".format(i, current_ip, np.argmax(np.array(foci_ips)), max(foci_ips)), 2)
        if((current_ip > foci_ips).any()):
            ind = np.argmin(np.absolute(z-foci))

            log("#{0:4}: Distance {1} to nearest focus{2:2}({3}) is {4:2.2f}".format(i, z, ind, foci[ind], \
                distance(z, foci[ind])), 2)

            if(distance(z, foci[ind]) < DISTANCE_THRESHOLD):
                log("#{2:4}: Changing focus{0:2} to {1}".format(ind, z, i), 2)
                foci[ind] = z
            else:
                foci.append(z)
                old_foci_distances.append(0)
                num_of_foci += 1
                log("#{2:4}: ADDING focus{0:2} at {1}".format(num_of_foci-1, z, i), 1)

        # Log data
        hist_current_ip[i] = current_ip
        hist_num_of_foci[i] = num_of_foci
        hist_foci.append(list(foci))
        hist_foci_ips.append(foci_ips)

    # Display results
    plt.plot(input_data, 'b', label="Input data")
    #plt.plot(hist_current_ip, 'r', label = "Current IP")
    plot_great_foci(make_foci_great_again(hist_foci))
    plt.legend()
    plt.show()

    return 0

def display_results(info):
    print("Results:")
    for i in range(len(info)):
        print("#{0:3}: {1}".format(i, info[i]))

# Calculates exponentially windowed mean square distance
def ewmsd(x, point_set):
    N = len(point_set)
    total_sum = 0

    for i in range(N):
        diff = x - point_set[i]

        temp_sum = 0
        for j in range(len(x)):
            temp_sum += diff[j] ** 2

        scaling = LAMBDA**(N-i)

        total_sum += temp_sum * scaling

    return (1-LAMBDA) * total_sum

# Calculates mean square distance of a point in relation to a set of points
def msd(x, point_set):
    N = len(point_set)
    total_sum = 0

    for i in range(N):
        diff = x - point_set[i]

        for j in range(len(x)):
            total_sum += diff[j] ** 2

    return total_sum/N

# Calculates information potential of a point in relation to a set of points
def information_potential(x, point_set):
    return 1/(1 + ewmsd(x, point_set))

def print_info():
    print("Number of points: " + str(DATA_NUM_POINTS))
    print("Lambda: " + str(LAMBDA))

def main():
    print_info()

    input_data = generate_dummy_data(DATA_NUM_POINTS)
    pbrc_info = simulate_pbrc(input_data)
    #display_results(pbrc_info)

if __name__ == "__main__":
    main()