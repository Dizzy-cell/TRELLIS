
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import utils3d
import numpy as np

class FlowDataset(Dataset):
    def  __init__ (self, feature_dir, latent_dir):
        self.feature_dir = feature_dir
        self.latent_dir = latent_dir
        self.features_names = sorted(glob.glob(os.path.join(feature_dir, '*.npz')))

        self.latent_names = sorted(glob.glob(os.path.join(latent_dir, '*.npz')))

        self.resolution = 16
        self.in_channels = 8

        paths = {}
        re_feature = []
        for path in self.features_names:
            name = os.path.basename(path)
            paths[name] = [path]
        
        for path in self.latent_names:
            name = os.path.basename(path)
            if name in paths.keys():
                paths[name].append(path)
                re_feature.append(path)

        self.latent_names = sorted(re_feature)

        print("Prepare the dataset!")
        print(self.features_names[0], len(self.features_names))
        print(self.latent_names[0], len(self.latent_names))

        # from IPython import embed 
        # embed()

        rt = '/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/image_features'
        # for idx, path in enumerate(self.features_names):
        #     name = os.path.basename(path)[:-4]
        #     images = np.load(path)['patchtokens_img_norm']

        #     latent = np.load(self.latent_names[idx])['mean'] # (8, 16, 16, 16)
        #     #print(images.shape)
        #     dir_file = os.path.join(rt, name)
        #     os.makedirs(dir_file, exist_ok=True)
        #     for i in range(150):
        #         file_name = os.path.join(dir_file,  f"{i:03d}.npz")
        #         res = {'image': images[i],
        #                 'latent': latent}
        #         np.savez(file_name, **res)
            
        
        self.image_features =  sorted(glob.glob(os.path.join(rt, '*', '*.npz')))


        # feature: 
        # indices: (num_points, 3)           #(64, 64, 64)
        # patchtokens: (num_voxel, 3)
        # patchtokens_img: (150, 1024, 37, 37) 1369 + 1 + 4 = 1374

        # lantent:
        # mean: (8, 16, 16, 16)
    def  __len__ (self):
        return len(self.image_features)

    def  __getitem__ (self, idx):
        # feature = np.load(self.features_names[idx]) #patchtokens: (num_voxel, 1024) patchtokens_img:
        # latent = np.load(self.latent_names[idx])['mean'] # (8, 16, 16, 16)

        # feature_cat = feature['patchtokens_img_norm'] # (150, 1274, 1024)
        # print(idx)
        data = np.load(self.image_features[idx])
        feature_cat = data['image']                 #(1374, 1024)
        latent = data['latent']                     #(8, 16, 16, 16)   
        return feature_cat, latent
