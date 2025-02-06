import DeepMIMOv3 as DeepMIMO
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

# Load the default parameters
parameters = DeepMIMO.default_params()

# Set scenario name
parameters['scenario'] = 'O1_3p5'

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = r'./DeepMIMO_scenarios'


parameters['num_paths'] = 5

# User rows 1-100
parameters['user_rows'] = np.arange(2700)
parameters['user_row_first'] = 0
parameters['user_row_last'] = 0

# Activate only the first basestation
BS_list=[3]
parameters['active_BS'] = np.array(BS_list) 

parameters['OFDM']['bandwidth'] = 0.1 # 100 MHz
parameters['OFDM']['subcarriers'] = 100 # OFDM with 100 subcarriers
parameters['OFDM']['subcarriers_limit'] = 100 # Keep only first 64 subcarriers
parameters['OFDM']['selected_subcarriers'] = np.arange(100) # Keep only first 64 subcarriers

parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # Single antenna
parameters['bs_antenna']['shape'] = np.array([1, 32, 1]) # ULA of 32 elements
parameters['enable_BS2B'] = 0
pprint(parameters)

# Generate data
dataset = DeepMIMO.generate_data(parameters)

# print(len(dataset[0]['user']['channel']))

# print(dataset[0]['user']['channel'][0][0,0,:])
# print(np.sum(np.abs((dataset[0]['user']['channel'][0])**2)))
# print(np.linalg.norm(dataset[0]['user']['channel'][0]))
# print(dataset[0]['user']['paths'][0])
# print(dataset[0]['user']['channel'][0].shape)
# plt.plot(np.abs(np.fft.ifft(dataset[0]['user']['channel'][0][0,0,:])))
# plt.show()
for BS_i in range(len(dataset[0])):
    bs_i = BS_list[BS_i]
    for ue_i in range(len(dataset[0]['user']['channel'])):
        # print(ue_i)
        np.save(f"./dataset/data_O1_3p5/BS{bs_i}/{bs_i}_{ue_i}.npy",dataset[BS_i]['user']['channel'][ue_i])



# np.sum(np.abs(H) ** 2)