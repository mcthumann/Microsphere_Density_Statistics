# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:12:03 2024

@author: Student
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd


file_name = "just_mass_5_to_7.txt"
chunk_size = 100

def read_in(file_name):
    df = pd.read_csv(file_name)
    running_average = df[df.keys()[0]]
    density_measurements = df[df.keys()[1]]
    return np.array(running_average), np.array(density_measurements)



def binning(densities, chunk_size):
    binned_data = np.split(densities[0:len(densities)//chunk_size * chunk_size], len(densities) // chunk_size)
    averages = np.zeros(len(binned_data))
    print(len(averages))
    deviations = np.zeros(len(binned_data))
    for i in range(len(binned_data)):
        averages[i] = np.mean(binned_data[i])
        deviations[i] = np.std(binned_data[i])
        
    return averages, deviations



running_average, densities = read_in("just_mass_5_to_7.txt")
averages, deviations = binning(densities, chunk_size)
      

    
plt.hist(averages)
plt.axvline(np.average(averages))
plt.show()
print(np.average(averages))

#plt.hist(deviations)
    