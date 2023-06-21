import numpy as np
import pickle as pkl

with open("C:/Users/Kin/Desktop/kinect_r/saved_depth.pkl", 'rb') as f:
    depth_data = pkl.load(f)
    print(np.shape(depth_data))