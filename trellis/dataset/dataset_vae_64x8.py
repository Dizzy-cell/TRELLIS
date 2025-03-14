
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import utils3d

from IPython import embed
import numpy as np

class VAEDataset64x8(Dataset):
    def  __init__ (self, data_dir):
        self.data_dir = data_dir
        self.file_names = sorted(glob.glob(os.path.join(data_dir, '*.ply')))

        self.resolution = 512
        self.cube_resolution = 64
        self.num_cube = 512
        self.each_ax = 8

        self.sample = 15

        self.cube_resolution_half = 32

        print(len(self.file_names))

    def  __len__ (self):
        return len(self.file_names)

    def  __getitem__ (self, idx):

        ss = self.get_voxels(self.file_names[idx])

        return ss, ss
    
    def get_voxels(self, instance):
        position = utils3d.io.read_ply(instance)[0]
        coords = ((torch.tensor(position) + 0.5) * self.resolution).int().contiguous()

        random_array = np.random.randint(0, coords.shape[0], size=(self.sample))

        ss = torch.zeros(self.sample + 1, 1, self.cube_resolution, self.cube_resolution, self.cube_resolution, dtype=torch.float)

        for i in range(self.sample):
            u, v, w = coords[random_array[i], :]

            u = u // self.cube_resolution  * self.cube_resolution
            v = v // self.cube_resolution  * self.cube_resolution
            w = w // self.cube_resolution  * self.cube_resolution

            choose_coords = coords[(coords[:,0] >= u ) & (coords[:,0] < u + self.cube_resolution)]
            choose_coords = choose_coords[(choose_coords[:,1] >= v) & (choose_coords[:,1] < v + self.cube_resolution)]
            choose_coords = choose_coords[(choose_coords[:,2] >= w ) & (choose_coords[:,2] < w + self.cube_resolution)]
            
            #print("Choose_coord", choose_coords.shape) # about (10000, 1)

            ss[i, :, choose_coords[:, 0] - u, choose_coords[:, 1] - v, choose_coords[:, 2] - w] = 1

        return ss
