import numpy as np
import random

Y_PERIOD = 10
counter = 0
noise_amplitude = 0.05

last_data = None

# How is plant input determined
# Options: 'random_2_foci', 'switch', '5_foci_array', 'zigzag', 'sqrt'
plant_type_x = '5_foci_array'

# How is y determined based on given x
# Options: 'peaks', 'linear'
plant_type_y = 'linear'


# Control functions
def reset():
    global counter, last_data
    counter = 0
    last_data = None

def set_plant_type_x(new_type):
    global plant_type_x
    plant_type_x = new_type

def set_plant_type_y(new_type):
    global plant_type_y
    plant_type_y = new_type

def set_y_period(new_period):
    global Y_PERIOD
    Y_PERIOD = new_period

def set_noise_amplitude(new_amplitude):
    global noise_amplitude
    noise_amplitude = new_amplitude


# Represents input vector as a function of time (counter is a form of time)
def get_x(counter):

    if plant_type_x == 'switch':
        f1 = np.array([2,3])
        f2 = np.array([4,5])
        focus_switch_time = 500
        if(counter < focus_switch_time):
            x = f1
        else:
            x = f2

    elif plant_type_x == 'random_2_foci':
        active_focus = random.choice([f1, f2])

    elif plant_type_x == '5_foci_array':
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

    elif plant_type_x == 'zigzag':
        xnum = 51
        total = xnum**2
        xmin = -3
        xmax = 3
        xlen = xmax-xmin
        dx = xlen/(xnum-1)
        x2 = int((counter%total)/xnum)
        tmp = counter % (2*xnum)
        if tmp < xnum:
            x1 = tmp
        else:
            x1 = 2*xnum-tmp-1
            
        return xmin + dx*np.array([x1,x2])

    elif plant_type_x == 'sqrt':
        segment_len = 100
        segment_num = 4
        max_val = 4
        min_val = 0

        counter = counter % (segment_num*segment_len)
        i = counter % segment_len

        if counter < 1*segment_len:
            x1 = min_val
            x2 = min_val
        elif counter < 2*segment_len:
            x1 = (max_val-min_val) * i/segment_len
            x2 = np.sqrt(x1)
        elif counter < 3*segment_len:
            x1 = max_val
            x2 = np.sqrt(max_val)
        elif counter < 4*segment_len:
            x1 = max_val - (max_val-min_val) * i/segment_len
            x2 = np.sqrt(x1)
        x = np.array([x1,x2])

    return x

# Calculates plant output for the given plant input
def get_y(x, add_noise = True):

    # Uniform noise
    noise = random.uniform(-noise_amplitude, +noise_amplitude)

    if plant_type_y == 'linear':
        alpha = np.array([10, 1])
        y = np.dot(x, alpha)

    elif plant_type_y == 'peaks':
        x1 = x[0]
        x2 = x[1]
        y = 3 * (1-x1)**2 * np.exp(-(x1**2) - (x2+1)**2) \
               - 10 * (x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2)\
               - 1/3 * np.exp(-(x1+1)**2 - x2**2)
               
    if add_noise == True:
        return y + noise
    else:
        return y

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