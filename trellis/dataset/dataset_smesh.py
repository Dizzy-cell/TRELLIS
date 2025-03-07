import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import utils3d
import numpy as np

import json
import cv2

import OpenEXR

class SMeshDataset(Dataset):
    def  __init__ (self, feature_dir, latent_dir, image_dir):
        self.feature_dir = feature_dir
        self.latent_dir = latent_dir
        self.image_dir = image_dir
        #self.features_names = sorted(glob.glob(os.path.join(feature_dir, '*.npz')))

        self.transforms_names = sorted(glob.glob(os.path.join(image_dir, "*",'transforms.json')))

        self.latent_names = sorted(glob.glob(os.path.join(latent_dir, '*.npz')))

        self.resolution = 16
        self.in_channels = 8

        self.num_image = 8

        paths = {}
        re_path = []
        re_lant = []
        for path in self.transforms_names:
            directory = os.path.dirname(path)
            last_directory = os.path.basename(directory)
            paths[last_directory] = path
        
        for path in self.latent_names:
            name = os.path.basename(path)[:-4]
            if name in paths.keys():
                re_path.append(paths[name])
                re_lant.append(path)

        self.latent_names = sorted(re_lant)
        self.transforms_names = sorted(re_path)

        print("Prepare the dataset!")
        print(self.transforms_names[0], len(self.transforms_names))
        print(self.latent_names[0], len(self.latent_names))

        # from IPython import embed 
        # embed()

        # image_features_dir = '/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/image_features'
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
            
        
        #self.image_features =  sorted(glob.glob(os.path.join(image_features_dir, '*', '*.npz')))

        # feature: 
        # indices: (num_points, 3)           #(64, 64, 64)
        # patchtokens: (num_voxel, 3)
        # patchtokens_img: (150, 1024, 37, 37) 1369 + 1 + 4 = 1374

        # lantent:
        # mean: (8, 16, 16, 16)
    def  __len__ (self):
        return len(self.latent_names)

    def  __getitem__ (self, idx):
        latent = np.load(self.latent_names[idx])
        feats = latent['feats']
        indices = latent['coords']

        intrinsics, extrinsics, images, yaws, pitchs, radius, normals, random_ints = self.loadCameraImage(self.transforms_names[idx], num = self.num_image)
   
        return feats, indices, intrinsics, extrinsics, images, yaws, pitchs, radius, normals, random_ints

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
        masks = torch.zeros(num, 512, 512)
        normals = torch.zeros(num, 512, 512, 3)
        depths = torch.zeros(num, 512 ,512)
        albedos = torch.zeros(num, 512 ,512, 3) 

        # yaws = torch.zeros(num, 1)
        # pitchs = torch.zeros(num, 1)
        # radius = torch.zeros(num, 1)

        for i, idx in enumerate(random_ints):
            camera_angle_x[i] = torch.tensor(frames[idx]['camera_angle_x'])
            c2w = torch.tensor(frames[idx]['transform_matrix'])
            c2w[:3,1:3] *=-1
            w2c = torch.inverse(c2w)
            transform_matrix[i] = w2c
            name = frames[idx]['file_path']
            image_name = os.path.join(rt, name)

            # image = cv2.imread(image_name)
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            
            image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            image_rgb = image[:,:,:3] / 255.0
            images[i] = torch.from_numpy(image_rgb)
            masks[i] = torch.from_numpy(image[:,:,3] / 255.0)

            image_name = os.path.join(rt, name[:-4] + '_albedo.png' )
            #print(image_name)
            image = cv2.imread(image_name)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            albedos[i] = torch.from_numpy(image_rgb)

            image_name = os.path.join(rt, name[:-4] + '_depth.png' )
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE) / 255.0
            depths[i] = torch.from_numpy(image)

            image_name = os.path.join(rt, name[:-4] + '_normal.exr' )

            exr_file = OpenEXR.InputFile(image_name)

            dw = exr_file.header()['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            channels = exr_file.channels(['R', 'G', 'B'])
            r = np.frombuffer(channels[0], dtype=np.float16)
            g = np.frombuffer(channels[1], dtype=np.float16)
            b = np.frombuffer(channels[2], dtype=np.float16)

            r = np.reshape(r, (height, width))
            g = np.reshape(g, (height, width))
            b = np.reshape(b, (height, width))

            image = np.stack((r, g, b), axis=-1)
            normals[i] = torch.from_numpy(image)

        intrinsics = utils3d.torch.intrinsics_from_fov_xy(camera_angle_x, camera_angle_x)
        extrinsics = transform_matrix
        random_ints = torch.tensor(random_ints)
        
        return intrinsics, extrinsics, images, albedos, depths, masks, normals, random_ints
