import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

DATA_NUM_POINTS = 200;
LAMBDA = 0.95
DATA_TYPE = "sine"

def generate_dummy_data(N):
    x = np.linspace(0, 2 * math.pi, N)
    dummy_data = np.empty([N,1])

    if(DATA_TYPE == "sine"):
        big_sine = np.sin(x)
        small_sine = 0.1 * np.sin(7 * x)        
        dummy_data[:,0] = small_sine + big_sine
    elif(DATA_TYPE == "ramp"):
        half = math.floor(N/2)
        dummy_data[0:half,0] = x[0:half]
        dummy_data[half:,0] = x[half]

    return dummy_data

def simulate_pbrc_stupid(input_data):
    dim = input_data.shape[1]
    N = input_data.shape[0]
    
    res = np.empty(N)

    for i in range(N):
        res[i] = information_potential(input_data[i], input_data[0:i])

    return res

def simulate_pbrc(input_data):
    dim = input_data.shape[1]
    N = input_data.shape[0]

    foci = [np.array([0]), np.array([1]), np.array([-1])]
    old_foci_distances = [0, 0, 0]
    num_of_foci = len(foci)

    Z = 0
    F = np.zeros(dim)
    S = 0
    z_old1 = np.zeros(dim)
    z_old2 = np.zeros(dim)

    point_ips = np.empty(N)
    focus_ips = np.empty(N)

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

        if((z > foci_ips).any()):
            #print("Neko je veci!")
            pass

        print("#{0:3} Current ip: {1:4.2f}; focus ip : {2:4.2f}".format(i, current_ip, foci_ips[0]))

        point_ips[i] = current_ip
        focus_ips[i] = foci_ips[-1]

    plt.legend()
    plt.show()
    return 0

def display_results(info):
    print("Results:")
    for i in range(len(info)):
        print("#{0:3}: {1}".format(i, info[i]))

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