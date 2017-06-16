import numpy as np
from random import uniform

f1 = np.array([2,3])
f2 = np.array([4,5])

Y_PERIOD = 10
counter = 0
noise_amplitude = 0.1
focus_switch_time = 500

# Represents input vector as a function of time (counter is a form of time)
def get_x(counter):

    # Uniform noise
    noise = uniform(-noise_amplitude, +noise_amplitude)

    if(counter < focus_switch_time):
        active_focus = f1
    else:
        active_focus = f2

    return active_focus + noise

# Calculates plant output for the given plant input
def get_y(x):
    alpha = np.array([10, 1])
    return np.dot(x, alpha)

# Always fetches input vector of plant, and fetches output vector if period has expired
def get_next_data():
    global counter

    # Get input vector
    x = get_x(counter)

    # Get output value if period has expired
    if counter % Y_PERIOD == 0:
        y = get_y(x)
    else:
        y = None

    # Tracking counter because of period
    counter += 1

    return {'x' : x, 'y': y}    