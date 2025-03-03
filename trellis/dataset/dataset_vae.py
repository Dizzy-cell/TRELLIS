import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import utils3d

class VAEDataset(Dataset):
    def  __init__ (self, data_dir):
        self.data_dir = data_dir
        self.file_names = sorted(glob.glob(os.path.join(data_dir, '*.ply')))

        self.resolution = 64

    def  __len__ (self):
        return len(self.file_names)

    def  __getitem__ (self, idx):

        ss = self.get_voxels(self.file_names[idx])

        return ss, torch.clone(ss)
    
    def get_voxels(self, instance):
        position = utils3d.io.read_ply(instance)[0]
        coords = ((torch.tensor(position) + 0.5) * self.resolution).int().contiguous()
        ss = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.float)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return ss
