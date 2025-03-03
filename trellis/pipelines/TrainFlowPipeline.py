
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from .base import Pipeline
from torch.optim.lr_scheduler import StepLR

from IPython import embed
import json

import numpy as np
from tqdm import tqdm
from . import samplers

from torch.cuda.amp import autocast, GradScaler

class TrainingFlowPipeline(Pipeline):
    def  __init__ (self, models: Dict[str, nn.Module] = None, criterion: nn.Module = None, optimizer: optim.Optimizer = None):
        super(). __init__ (models)
        self.criterion = criterion

        self.loss_mse = nn.MSELoss()

        for model in self.models.values():
            model.train()
        
        params = []
        for model in self.models.values():
            params.extend(model.parameters())
        optimizer = optim.AdamW(params, lr= 1 * 1e-4)  #10-4 -> 10-5 -> 10e-6 -> 10e-7

        self.optimizer = optimizer

        print('Build sampler!')
        # from IPython import embed 
        # embed()

        json_config = '''
        {
            "sparse_structure_sampler": {
            "name": "FlowEulerGuidanceIntervalSampler",
            "args": {
                "sigma_min": 1e-5
            },
            "params": {
                "steps": 25,
                "cfg_strength": 5.0,
                "cfg_interval": [0.5, 1.0],
                "rescale_t": 3.0
            }
            }
        }
        '''
        config = json.loads(json_config)
        
        self.sigma_min = 1e-5

        self.sparse_structure_sampler = getattr(samplers, config['sparse_structure_sampler']['name'])(**config['sparse_structure_sampler']['args'])
        self.sparse_structure_sampler_params = config['sparse_structure_sampler']['params']

        #self.scheduler = StepLR(optimizer, step_size=100000, gamma=0.1)
    
    def KID_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def logdir_loss(self, logvar):
        return logvar.exp().mean()

    def train(self, train_loader: Any, epochs: int = 1) -> None:

        scaler = GradScaler()
     
        for epoch in range(epochs):
            tot_loss = 0.0
            for features, latent in tqdm(train_loader):
                images_features = features.to(self.device)
                #print(images_features.shape)

                latent = latent.to(self.device)
                features_dict = {'cond': images_features,
                                'neg_cond': torch.zeros_like(images_features)}

                self.optimizer.zero_grad()

                # loss: loss_flow
                with autocast():
                    loss, t = self.predictFromFeature(features_dict['cond'], latent, num_samples = images_features.shape[0])

                    loss = loss.mean()

                scaler.scale(loss).backward()

                from IPython import embed 
                embed()

                scaler.step(self.optimizer)
                scaler.update()

                #self.scheduler.step()
                tot_loss += loss.item()

                #print(f"Epoch {epoch + 1}/{epochs}, Loss Total: {loss.item()}")
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss Total: {tot_loss/len(train_loader)}")
            
            if epoch % 100 ==0:
                self.save(f"./dev/{epoch}.pth")

    def forward(self, inputs: Any) -> Any:
        latents, mean, logvar = self.models['encoder'](inputs, sample_posterior = True, return_raw = True)
        outputs = self.models['decoder'](latents)

        return outputs, mean, logvar

    def evaluate(self, test_loader: Any) -> float:
        total_loss = 0.0
        tot = 0 
        with torch.no_grad():
            for features, latent in test_loader:
                #images_features = features[0][:2].to(self.device)
                images_features = features.to(self.device)
                latent = latent.to(self.device)
                features_dict = {'cond': images_features,
                                'neg_cond': torch.zeros_like(images_features)}

                sample = self.sampleFromFeature(features_dict, num_samples = images_features.shape[0])

                loss = self.loss_mse(sample[0], latent[0])
                loss2 = self.loss_mse(sample[1], sample[0])
                print(loss, loss2)          #(0.3636) , (0.2763) 

                total_loss += loss.item()
                
                tot += 1
                if tot >= 3:
                    break

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

    def sampleFromFeature(self, cond : dict, num_samples: int = 1, sampler_params: dict = {}):
        flow_model = self.models['ss_flow']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        return z_s

    def predictFromFeature(self, cond, latent, num_samples: int = 1, sampler_params: dict = {}):
        flow_model = self.models['ss_flow']
        reso = flow_model.resolution

        with autocast():
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)

            t = torch.randn(num_samples, 1, 1, 1, 1).to(self.device)
            
            x_t = latent * (1 - t) + noise * t 
            #t_flow = torch.tensor([1000 * t] * num_samples, device=self.device, dtype=torch.float32)            
            t_flow = t.reshape(-1) * 1000

            pred_flow = flow_model(x_t, t_flow, cond)
            loss_flow = self.loss_mse(pred_flow,  -(latent - noise))

        return loss_flow,  t
    
    # v = flow
    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
