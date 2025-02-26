
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import utils3d
import numpy as np
import trellis.modules.sparse as sp
import json
import cv2
import utils3d
#
#             c2w = torch.tensor(view['transform_matrix'])
#             c2w[:3, 1:3] *= -1
#             extrinsics = torch.inverse(c2w)
class SVAEDataset(Dataset):
    def  __init__ (self, data_dir, image_dir):
        self.data_dir = data_dir
        self.features_names = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        self.transforms_names = sorted(glob.glob(os.path.join(image_dir, "*",'transforms.json')))
        #self.images_names =  sorted(glob.glob(os.path.join(image_dir, "*",'.png')))
        self.resolution = 64

        self.num_image = 8

        paths = {}
        re_feature = []
        for path in self.transforms_names:
            directory = os.path.dirname(path)
            last_directory = os.path.basename(directory)
            paths[last_directory] = [path]
        
        for path in self.features_names:
            name = os.path.basename(path)[:-4]
            if name in paths.keys():
                paths[last_directory].append(path)
                re_feature.append(path)

        self.features_names = sorted(re_feature)

        print("SVAE Dataset")
        print(self.transforms_names[0])
        print(self.features_names[0])

        # from IPython import embed 
        # embed()

        # features_names: (1024, 64, 64, 64) -> coords: (0, 64, 64, 64) , features: (N, 1024)
        # intrinsics: (N, 3, 3)
        # extrinsics: (N, 4, 4)

        # mean: (8, 16, 16, 16) 
        # sparse struct latent: (1, 8)
        # slat: (1, 8), (batchsize, feature.shape] 
        # coords, feats : (resolution, channel) : (64, 8)

    def  __len__ (self):
        return len(self.features_names)
    
    def  __getitem__ (self, idx):
        feats = np.load(self.features_names[idx])
        intrinsics, extrinsics, images, yaws, pitchs, radius, random_ints = self.loadCameraImage(self.transforms_names[idx], num = self.num_image)

        # print(self.features_names[idx])
        # print(self.transforms_names[idx])
        # print(random_ints)
        # feats = sp.SparseTensor(
        #     feats = torch.from_numpy(feats['patchtokens']).float(),
        #     coords = torch.cat([
        #         torch.zeros(feats['patchtokens'].shape[0], 1).int(),
        #         torch.from_numpy(feats['indices']).int(),
        #     ], dim=1),
        # )

        return feats['patchtokens'], feats['indices'], intrinsics, extrinsics, images, yaws, pitchs, radius, random_ints
    
    def loadCameraImage(self, transforms_name, num = 4, max_num = 150):
        with open(transforms_name, 'r', encoding='utf-8') as file:
            camera = json.load(file)
        
        aabb = camera['aabb']
        scale = camera['offset']
        offset = camera['offset']
        frames = camera['frames']

        rt = os.path.dirname(transforms_name)

        random_ints = np.random.randint(0, max_num, size=num)
        camera_angle_x = torch.zeros(num)
        transform_matrix = torch.zeros(num, 4, 4)
        images = torch.zeros(num, 512, 512, 3)
        yaws = torch.zeros(num, 1)
        pitchs = torch.zeros(num, 1)
        radius = torch.zeros(num, 1)

        for i, idx in enumerate(random_ints):
            camera_angle_x[i] = torch.tensor(frames[idx]['camera_angle_x'])
            c2w = torch.tensor(frames[idx]['transform_matrix'])
            c2w[:3,1:3] *=-1
            w2c = torch.inverse(c2w)
            transform_matrix[i] = w2c
            name = frames[idx]['file_path']
            image_name = os.path.join(rt, name)
            image = cv2.imread(image_name)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            images[i] = torch.from_numpy(image_rgb)
            yaws[i] = torch.tensor(frames[idx]['yaw'])
            pitchs[i] = torch.tensor(frames[idx]['pitch'])
            radius[i] = torch.tensor(frames[idx]['radius'])

        intrinsics = utils3d.torch.intrinsics_from_fov_xy(camera_angle_x, camera_angle_x)
        extrinsics = transform_matrix
        random_ints = torch.tensor(random_ints)
        
        return intrinsics, extrinsics, images, yaws, pitchs, radius, random_ints
