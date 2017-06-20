import numpy as np
import random

f1 = np.array([2,3])
f2 = np.array([4,5])

Y_PERIOD = 10
counter = 0
noise_amplitude = 0.05
focus_switch_time = 500

last_data = None

plant_type = '5_foci_array'
#random_2_foci
#switch
#5_foci_array

# Represents input vector as a function of time (counter is a form of time)
def get_x(counter):

    # Uniform noise
    noise = random.uniform(-noise_amplitude, +noise_amplitude)

    if plant_type == 'switch':
        if(counter < focus_switch_time):
            x = f1
        else:
            x = f2

    elif plant_type == 'random_2_foci':
        active_focus = random.choice([f1, f2])

    elif plant_type == '5_foci_array':
        segment_len = 100
        segment_num = 4
        max_val = 4
        min_val = 0

        counter = counter % (segment_num*segment_len)
        i = counter % segment_len

        if counter < 1*segment_len:
            x = min_val
        elif counter < 2*segment_len:
            x = (max_val-min_val) * i/segment_len
        elif counter < 3*segment_len:
            x = max_val
        elif counter < 4*segment_len:
            x = max_val - (max_val-min_val) * i/segment_len
        x = np.array([x,x])

    return x + noise

# Calculates plant output for the given plant input
def get_y(x):
    alpha = np.array([10, 1])
    return np.dot(x, alpha)

# Fetches last plant data
def get_last_data():
    return last_data

# Always fetches input vector of plant, and fetches output vector if period has expired
def get_next_data():
    global counter
    global last_data

    # Get input vector
    x = get_x(counter)

    # Get output value if period has expired
    if counter % Y_PERIOD == 0:
        y = get_y(x)
    else:
        y = None

    # Tracking counter because of period
    counter += 1

    data = {'x' : x, 'y': y}
    last_data = data

    return data