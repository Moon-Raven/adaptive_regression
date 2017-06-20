import numpy as np
import matplotlib.pyplot as plt
import math

# Configuration
LAMBDA = 0.6
DISTANCE_THRESHOLD = 0.3
LOG_LEVEL = 0

# ********************* Configuration functions *********************

def set_lambda(new_lambda):
    global LAMBDA
    LAMBDA = new_lambda

def set_distance_threshold(new_threshold):
    global DISTANCE_THRESHOLD
    DISTANCE_THRESHOLD = new_threshold

def set_log_level(new_log_level):
    global LOG_LEVEL
    LOG_LEVEL = new_log_level

# *** End of configuration functions ***



# ********************* Control *********************

# Static/global control variables
dim = 2
foci = []
old_foci_distances = []
num_of_foci = 0

Z = 0
F = np.zeros(dim)
S = 0
z_old1 = np.zeros(dim)
z_old2 = np.zeros(dim)

# Add new focus
def add_focus(z, starting_distance = 0):
    global foci
    global old_foci_distances
    global num_of_foci

    foci.append(z)
    old_foci_distances.append(starting_distance)
    num_of_foci += 1

# Perform an iteration of PBRC
def iterate(x):
    global dim
    global foci
    global old_foci_distances
    global num_of_foci
    global Z
    global F
    global S
    global z_old1
    global z_old2

    # Newly received feature vector
    z = x

    # Calculate the new information potential
    Z_new = LAMBDA*Z + 1
    F_new = LAMBDA*F + LAMBDA*Z*(z_old1 - z_old2)
    S_new = LAMBDA*S + 2*LAMBDA*(1-LAMBDA)*np.dot((z-z_old1),F_new)\
           +LAMBDA*(1-LAMBDA)*np.dot(z-z_old1, z-z_old1)*Z_new

    # Update static variables
    S = S_new
    Z = Z_new
    F = F_new        
    z_old2 = z_old1
    z_old1 = z

    current_ip = 1/(1+S_new)

    # Update information potential of all foci
    foci_ips = np.zeros([num_of_foci])

    for j in range(num_of_foci):
        new_distance = (1-LAMBDA)*np.dot(z-foci[j], z-foci[j]) + LAMBDA*old_foci_distances[j]
        foci_ips[j] = distance2ip(new_distance)                
        old_foci_distances[j] = new_distance

    # If current information potential is larger than IPs of all the foci, do something
    if((current_ip > foci_ips).all()):

        log("Current IP: {0}, IP of highest focus {1} is {2}".format(current_ip, max(foci_ips), np.argmax(foci_ips)), 2)

        # Find index of closest focus
        # Convert list of arrays to numpy matrix, should change
        # foci to be matrix in the first place :(
        np_foci = np.array(foci)
        zmf = z-foci
        z2 = np.power(zmf, 2)
        z3 = np.sum(z2,axis=1)
        z4 = np.sqrt(z3)
        ind = np.argmin(z4)
        log("Distance {0} to nearest focus{1:2}({2}) is {3:2.2f}".format(z, ind, foci[ind], \
            distance(z, foci[ind])), 2)

        # If it is close enough, update that focus
        if(distance(z, foci[ind]) < DISTANCE_THRESHOLD):
            log("Changing focus{0:2} to {1}".format(ind, z), 2)
            foci[ind] = z
            old_foci_distances[ind] = 0

        # If it is too far away, create new focus
        else:
            add_focus(z)
            log("ADDING focus{0:2} at {1}".format(num_of_foci-1, z), 1)

# *** End of control ***




# ********************* Information fetching functions *********************

# Returns all foci
def get_foci():
    return foci

def get_foci_ips():
    np_distances = np.array(old_foci_distances)
    foci_ips = np.apply_along_axis(distance2ip, 0, np_distances)
    return foci_ips
    
# *** End of information fetching functions ***




# ******************* Utility functions *********************

# Performs the given log if it's level is high enough
def log(s, level):
    if(level <= LOG_LEVEL):
        print(s)

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

# Calculates the distance between two vectors using l2 norm        
def distance(z1, z2):
    difference = z1 - z2;
    return np.sqrt(np.dot(difference, difference))

def distance2ip(focus_distance):
    return (1/(1+focus_distance))
# *** End of utility functions ***






