import numpy as np
import matplotlib.pyplot as plt
import math

# Configuration
LAMBDA = 0.6
DISTANCE_THRESHOLD = 0.3
LOG_LEVEL = 0

# ***** Configuration functions *****

def set_lambda(new_lambda):
    global LAMBDA
    LAMBDA = new_lambda

def set_distance_threshold(new_threshold):
    global DISTANCE_THRESHOLD
    DISTANCE_THRESHOLD = new_threshold

def set_log_level(new_log_level):
    global LOG_LEVEL
    LOG_LEVEL = new_log_level

def reset():
    global dim, foci, old_foci_distances, original_foci_positions, num_of_foci
    global Z, F, S, z_old1, z_old2, foci_frozen

    dim = 2
    foci = []
    old_foci_distances = []
    original_foci_positions = []
    num_of_foci = 0

    Z = 0
    F = np.zeros(dim)
    S = 0
    z_old1 = np.zeros(dim)
    z_old2 = np.zeros(dim)
    foci_frozen = False



# ***** Control functions *****

# Static/global control variables
dim = 2
foci = []
old_foci_distances = []
original_foci_positions = []
num_of_foci = 0

Z = 0
F = np.zeros(dim)
S = 0
z_old1 = np.zeros(dim)
z_old2 = np.zeros(dim)

foci_frozen = False

# Add new focus
def add_focus(z, starting_distance = 0):
    global foci, old_foci_distances, num_of_foci

    foci.append(z)
    old_foci_distances.append(starting_distance)
    original_foci_positions.append(z)

    num_of_foci += 1

# Resets the focus with the given index back to its original position
def reset_focus(ind):
    foci[ind] = original_foci_positions[ind]

# Make foci unable to move
def freeze_foci():
    global foci_frozen
    foci_frozen = True

# Make foci able to move again
def unfreeze_foci():
    global foci_frozen
    foci_frozen = False

# Perform an iteration of PBRC
def iterate(x):
    global dim, foci, old_foci_distances, num_of_foci, Z, F, S, z_old1, z_old2

    # Newly received feature vector
    z = x

    if num_of_foci == 0:
        add_focus(x)
        return

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
    if (current_ip > foci_ips).all() and foci_frozen == False:

        if len(foci_ips) != 0:
            log("Current IP: {0}, IP of highest focus {1} is {2}".format(current_ip, max(foci_ips), np.argmax(foci_ips)), 2)

        # Find index of closest focus
        np_foci = np.array(foci)
        zmf = z-foci
        z2 = np.power(zmf, 2)
        z3 = np.sum(z2,axis=1)
        z4 = np.sqrt(z3)
        ind = np.argmin(z4)
        log("Distance {0} to nearest focus{1:2}({2}) is {3:2.2f}".format(z, ind, foci[ind], \
            distance(z, foci[ind])), 2)

        # If it is close enough to the original focus position, update that focus
        if(distance(z, original_foci_positions[ind]) < DISTANCE_THRESHOLD):
            log("Changing focus{0:2} from {1} to {2}".format(ind, foci[ind], z), 2)
            foci[ind] = z
            old_foci_distances[ind] = 0

        # If it is too far away, reset that focus, check other close foci
        else:            
            log("RESETTING focus{0:2} from {1} back to {2}".format(ind, foci[ind], original_foci_positions[ind]), 1)
            reset_focus(ind)

            legit_foci_ind = [ind for ind in range(len(foci)) if distance(original_foci_positions[ind], z) < DISTANCE_THRESHOLD]
            
            if len(legit_foci_ind) == 0:
                log("ADDING focus{0:2} at {1}".format(num_of_foci-1, z), 1)
                add_focus(z)
            else:
                vals = [distance(z, foci[i]) for i in legit_foci_ind]
                xxx = np.array(vals)
                min_ind = np.argmin(xxx)
                min_foci_ind = legit_foci_ind[min_ind]
                log("Changing focus{0:2} from {1} to {2}".format(ind, foci[ind], z), 2)
                foci[min_foci_ind] = z
                old_foci_distances[min_foci_ind] = 0



# ***** Information fetching functions *****

# Returns all foci
def get_foci():
    return foci

def get_foci_ips():
    np_distances = np.array(old_foci_distances)
    foci_ips = np.apply_along_axis(distance2ip, 0, np_distances)
    return foci_ips



# ***** Utility functions *****

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






