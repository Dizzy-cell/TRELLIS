
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from .base import Pipeline
from torch.optim.lr_scheduler import StepLR

import trellis.modules.sparse as sp
from ..renderers import GaussianRenderer
from IPython import embed
import numpy as np
import tqdm
from ..representations import Octree, Gaussian, MeshExtractResult
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames
from PIL import Image

class TrainingFeaturePipeline(Pipeline):
    def  __init__ (self, models: Dict[str, nn.Module] = None, criterion: nn.Module = None, optimizer: optim.Optimizer = None):
        super(). __init__ (models)
        self.criterion = criterion

        self.bce = nn.BCEWithLogitsLoss() 

        for model in self.models.values():
            model.train()
        
        params = []
        for model in self.models.values():
            params.extend(model.parameters())
        optimizer = optim.AdamW(params, lr= 1 * 10e-6)  #10-4 -> 10-5 -> 10e-6 -> 10e-7

        self.optimizer = optimizer

        self.weight_kid = 0

        self.weight_res = 1  # 10000

        self.weight_bce = 10

        print(self.weight_res)

        #self.scheduler = StepLR(optimizer, step_size=100000, gamma=0.1)

        renderer = GaussianRenderer()
        options={}
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        #renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.bg_color = options.get('bg_color', (1, 1, 1))
        # white (1,1,1)
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = options.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True

        self.renderer = renderer

        self.loss_l1 = nn.functional.l1_loss

    def KID_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def logdir_loss(self, logvar):
        return logvar.exp().mean()

    def train(self, train_loader: Any, epochs: int = 1) -> None:
        for epoch in range(epochs):
            total_loss = 0.0
            for feats, indices, intrinsics_t, extrinsics_t, images, yaws_t, pitchs_t, radius_t, random_ints in train_loader:
                
                self.optimizer.zero_grad()  # Clear gradients
                feats_sp = self.prepare_input(feats[0], indices[0])
                outputs, mean, logvar = self.forward(feats_sp)
                
                intrinsics = intrinsics_t.to(self.device)
                extrinsics = extrinsics_t.to(self.device)
                images = images.to(self.device)
                
                images_gen = self.renderImage(outputs[0], extrinsics[0], intrinsics[0])

                loss, loss_l1, loss_scale, loss_alpha = self.loss_total(outputs[0], images_gen['color_train'], images[0])

                loss.backward()  # Backward pass

                total_loss += loss.item()

                self.optimizer.step()  # Update weights

                #print(f"Loss total: {loss.item()} Loss L1: {loss_l1.item()} Loss alpha: {loss_alpha.item()} Loss scale: {loss_scale.item()}")

            print(f"Epoch {epoch + 1}/{epochs}, Loss total: {total_loss / len(train_loader)} Loss L1: {loss_l1.item()} Loss alpha: {loss_alpha.item()} ")
            
            if epoch % 1000 ==0:
                self.save(f"./dev/{epoch}.pth")

    def forward(self, inputs: Any) -> Any:
        latents, mean, logvar = self.models['encoder'](inputs, sample_posterior = True
        , return_raw = True)
        outputs = self.models['decoder'](latents)

        return outputs, mean, logvar

    def evaluate(self, test_loader: Any) -> float:
        total_loss = 0.0
        with torch.no_grad():
            for feats, indices, intrinsics_t, extrinsics_t, images, yaws_t, pitchs_t, radius_t, random_ints in test_loader:
                
                # feats = feats.to(self.devices)
                feats_sp = self.prepare_input(feats[0], indices[0])
                outputs, mean, logvar = self.forward(feats_sp)
                
                intrinsics = intrinsics_t.to(self.device)
                extrinsics = extrinsics_t.to(self.device)
                images = images.to(self.device)
                
                # images = self.renderer.render(outputs[0], extrinsics[0][0], intrinsics[0][0])
                # image = np.clip(images['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

                images_gen = self.renderImage(outputs[0], extrinsics[0], intrinsics[0])

                loss, loss_l1, loss_scale, loss_scaling = self.loss_total(outputs[0], images_gen['color_train'], images[0])

                #print(loss)
                #outputs[0].save_ply('./dev/sample.ply')
                #embed()

                # image = Image.fromarray(images['color'][0])
                # output_path = 'dev/render_generate.png' 
                # image.save(output_path)
                
                # yaws = [0, 4.4708, 8.2751]          #(0, -180)
                # pitch = [0, 1.0470, 0.2370]         #(0, 0) 
                # r = [2.0, 2.0, 2.0]            
                # fov = [40, 40, 40]           
                # resolution = 512
                # bg_color=(0, 0, 0)

                # extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
                # res = render_frames(outputs[0], extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color})
               
                # image = Image.fromarray(res['color'][0])
                # output_path = 'dev/render_sample2.png' 
                # image.save(output_path)

                # ouputs: gaussian_model
                # mean: ()
                # var: (8028, 8)    
                # loss_res = self.weight_res * self.criterion(outputs, targets)  # Compute loss
                # loss_kid = self.weight_kid * self.KID_loss(mean, logvar)
                # loss_bce = self.weight_bce * self.bce(outputs, targets)
                # loss = loss_res + loss_kid + loss_bce
                total_loss += loss.item()

                # print(f"Loss Res: {loss_res.item()} Loss KID: {loss_kid.item()}")
                # print(f"Output sum: {(outputs > 0).sum()}") # 190051, 189769
                # print(f"input sum: {(inputs > 0).sum()}")   # 189770, 189770
                # print(f"mean: {mean} logvar:{logvar}")
                
                # embed()
        average_loss = total_loss / len(test_loader)
        print(f"Test Loss: {average_loss}")
        return average_loss
    def save(self, path: str) -> None:
        model_states = {name: model.state_dict() for name, model in self.models.items()}
        torch.save(model_states, path)
        print(f"Models saved to {path}")
    
    def load(self, path: str) -> None:
        model_states = torch.load(path)
        for name, model in self.models.items():
            model.load_state_dict(model_states[name])
        
        print(f"Model loaded ok!")

    def prepare_input(self, feats, indices):

        feats = feats.float().to(self.device)
        coords = torch.cat([
            torch.zeros(feats.shape[0], 1).int(),
            indices.int()
        ], dim = 1).to(self.device)

        feats = sp.SparseTensor(
            feats = feats,
            coords = coords,
        )
        return feats
    def prepare_input(self, feats, indices):

        feats = feats.float().to(self.device)
        coords = torch.cat([
            torch.zeros(feats.shape[0], 1).int(),
            indices.int()
        ], dim = 1).to(self.device)

        feats = sp.SparseTensor(
            feats = feats,
            coords = coords,
        )
        return feats
    
    def getYaw_Pitch_Radius(self, extrinsic_matrix):
        R = extrinsic_matrix[:3, :3].cpu().numpy()  
        T = extrinsic_matrix[:3, 3].cpu().numpy()

        yaw = np.arctan2(R[1, 0], R[0, 0]) 
        pitch = np.arcsin(-R[2, 0]) 

        radius = np.linalg.norm(T)  

        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)

        print("Yaw:", yaw)
        print("Pitch:", pitch)
        print("Radius:", radius)
    
    def renderImage(self, sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=False, **kwargs):
        rets = {}
        #for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        rets['color_train'] = torch.zeros(extrinsics.shape[0], 512, 512, 3).to(self.device)
        for j in range(extrinsics.shape[0]):
            extr = extrinsics[j]
            intr = intrinsics[j]    
            if not isinstance(sample, MeshExtractResult):
                res = self.renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
                if 'color' not in rets: rets['color'] = []
                if 'depth' not in rets: rets['depth'] = []

                rets['color_train'][j] = res['color'].permute(1, 2, 0)

                rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
                if 'percent_depth' in res:
                    rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
                elif 'depth' in res:
                    rets['depth'].append(res['depth'].detach().cpu().numpy())
                else:
                    rets['depth'].append(None)
            else:
                res = self.renderer.render(sample, extr, intr)
                if 'normal' not in rets: rets['normal'] = []
                rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        return rets
    
    def loss_total(self, sample, images_gen, images):
        
        loss_l1 = self.loss_l1(images_gen, images)
        # loss_ssim = 
        # loss_lpips = 

        loss_scale =  torch.mean((sample.get_scaling)**2)
        loss_alpha =  torch.mean((1 - sample.get_opacity)**2)

        loss_total = loss_l1 + loss_scale + loss_alpha

        #print(f"Loss_l1: {loss_l1} Loss_scale: {loss_scale} Loss_alphaL: {loss_alpha}")
        return loss_total, loss_l1, loss_scale, loss_alpha
