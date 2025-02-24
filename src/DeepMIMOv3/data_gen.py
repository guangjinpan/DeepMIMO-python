import DeepMIMOv3 as DeepMIMO
import sys

import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import music

# Load the default parameters
parameters = DeepMIMO.default_params()

# Set scenario name
parameters['scenario'] = 'O1_3p5'

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = r'../../../DeepMIMO_scenarios'


parameters['num_paths'] = 5

# User rows 1-100
parameters['user_rows'] = [800]
parameters['user_row_first'] = 1
parameters['user_row_last'] = 1

# Activate only the first basestation
BS_list=[3]
parameters['active_BS'] = np.array(BS_list) 

parameters['OFDM']['bandwidth'] = 0.01 # 100 MHz
parameters['OFDM']['subcarriers'] = 64 # OFDM with 100 subcarriers
parameters['OFDM']['subcarriers_limit'] = 64 # Keep only first 64 subcarriers
parameters['OFDM']['selected_subcarriers'] = np.arange(64) # Keep only first 64 subcarriers

parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # Single antenna
parameters['bs_antenna']['shape'] = np.array([1, 32, 1]) # ULA of 32 elements
parameters['enable_BS2B'] = 0
parameters['bs_antenna']['rotation'] = np.array([0, 0, 0])
pprint(parameters)

# Generate data
dataset = DeepMIMO.generate_data(parameters)
useridx=60

# print(len(dataset[0]['user']['channel']))

# print(dataset[0]['user']['channel'][0][0,0,:])

print(dataset[0]['user']['distance'][useridx],dataset[0]['user']['distance'][useridx]/3e8)
channel = dataset[0]['user']['channel'][useridx]
print(channel.shape)
# x=np.fft.ifft(channel[0,:,0])
# plt.plot(np.abs(x)*1e5)

x = np.angle(channel[0,:,5])
plt.plot(x[1:]-x[0])
# plt.show()
# print(np.argsort(-np.abs(x)))
print("angle,", dataset[0]['user']['paths'][useridx])
print(dataset[0]['basestation'].keys(),dataset[0]['basestation']['location'][0],dataset[0]['user']['LoS'][useridx])
print("loc", dataset[0]['basestation']['location'][0][:2],dataset[0]['user']['location'][useridx][:2] ,music.calculate_distance_and_aoa(dataset[0]['basestation']['location'][0][:2],dataset[0]['user']['location'][useridx][:2]))
print(music.music_f(channel[0,:,:].transpose()))