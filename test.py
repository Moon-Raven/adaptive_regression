import pbrc
import numpy as np



def print_msd_and_ip(x, Z):
    print("MSD: " + str(pbrc.msd(x,Z)) + "; IP: " + str(pbrc.information_potential(x,Z)))

"""Z = np.array([[1,2,3],
              [5,2,6],
              [2,6,1]])"""
#print(str(pbrc.msd([1,2,3], Z)) + ", should be " + str(46/3) + ";")
#print("Information potential is " + str(pbrc.information_potential([1,2,3], Z)))

#Z = np.array([[1,1,1],
#              [3,3,3]])
#x = [2,2,2]

#print_msd_and_ip(x,Z)

