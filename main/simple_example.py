import pbrc
import grnn
import example_plant as plant
import logger

import matplotlib.pyplot as plt

def do_example():

    # Set simulation parameters
    N = 4000
    plant.set_noise_amplitude(20)
    pbrc.set_distance_threshold(0.5)
    pbrc.set_log_level(0)
    grnn.set_sigma(0.5)
    plant.set_plant_type_x('5_foci_array')
    plant.set_plant_type_y('linear')

    # Perform simulation
    for i in range(N):
        data = plant.get_next_data()
        x = data['x']    
        y = data['y']

        pbrc.iterate(x)
        foci_ips = pbrc.get_foci_ips()

        if y != None:
            grnn.add_node(foci_ips, y)

        estimated_y = grnn.get_regression(foci_ips)
        logger.collect_data()

    # Display results
    logger.plot_foci_num()
    logger.plot_y()
    logger.plot_node_num()
    logger.plot_foci_x()
    plt.show()

if __name__ == "__main__":
    do_example()