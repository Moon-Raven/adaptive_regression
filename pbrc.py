
import numpy as np
import matplotlib.pyplot as plt
import math

DATA_NUM_POINTS = 200;
LAMBDA = 0.95
THRESHOLD = 0.05
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
    #return np.array([[1,2,3],[1,2,3],[1,2,4]]);
    num_of_foci = 0
    dim = input_data.shape[1]
    N = input_data.shape[0]
    
    res = np.empty(N)

    for i in range(N):
        res[i] = information_potential(input_data[i], input_data[0:i])

    return res

def simulate_pbrc_stupid2(input_data):
    #return np.array([[1,2,3],[1,2,3],[1,2,4]]);
    num_of_foci = 0
    dim = input_data.shape[1]
    N = input_data.shape[0]
    
    res = np.empty(N)

    for i in range(N):
        res[i] = information_potential(input_data[i], input_data[0:i])

    return res

def information_potential2(x, point_set):
    return 1/(1 + ewmsd(x, point_set))

def ewmsd2(x, point_set):
    N = len(point_set)
    total_sum = 0

    for i in range(N):
        diff = x - point_set[i]

        temp_sum = 0
        for j in range(len(x)):
            temp_sum += diff[j] ** 2

        # Check if scaling factor is below threshold
        if(LAMBDA**(N-i) > THRESHOLD):
            scaling = LAMBDA**(N-i)
        else:
            scaling = 0

        total_sum += temp_sum * scaling

    return (1-LAMBDA) * total_sum
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

        # Check if scaling factor is below threshold
        if(LAMBDA**(N-i) > THRESHOLD):
            scaling = LAMBDA**(N-i)
        else:
            scaling = 0

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
    n = math.floor(math.log(THRESHOLD, LAMBDA))
    print("Number of points: " + str(DATA_NUM_POINTS))
    print("Window length: " + str(n))

def main():
    print_info()
    input_data = generate_dummy_data(DATA_NUM_POINTS)
    pbrc_info = simulate_pbrc_stupid(input_data)
    display_results(pbrc_info)

if __name__ == "__main__":
    main()