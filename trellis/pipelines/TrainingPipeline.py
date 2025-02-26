
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from .base import Pipeline
from torch.optim.lr_scheduler import StepLR

from IPython import embed

class TrainingPipeline(Pipeline):
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

        self.weight_bce = 100

        print(self.weight_res)

        #self.scheduler = StepLR(optimizer, step_size=100000, gamma=0.1)
    
    def KID_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def logdir_loss(self, logvar):
        return logvar.exp().mean()

    def train(self, train_loader: Any, epochs: int = 1) -> None:
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # for i in range(1000):

                self.optimizer.zero_grad()  # Clear gradients
                outputs, mean, logvar = self.forward(inputs)  # Forward pass
                loss_res = self.weight_res * self.criterion(outputs, targets)  # Compute loss
                loss_kid = self.weight_kid * self.KID_loss(mean, logvar)
                loss_bce = self.weight_bce * self.bce(outputs, targets)
                loss = loss_res + loss_kid + loss_bce
                loss.backward()  # Backward pass

                # torch.nn.utils.clip_grad_norm_(self.models['encoder'].parameters(), max_norm=1.0)
                # torch.nn.utils.clip_grad_norm_(self.models['decoder'].parameters(), max_norm=1.0)
                self.optimizer.step()  # Update weights

                #self.scheduler.step()

                #     print(f"Loss: {loss.item(), loss_res.item(), loss_kid.item()}")
                # embed()

                # for name, param in self.models['encoder'].named_parameters():
                #     if param.grad is not None:  
                #         print(f"Gradient for {name}: {param.grad.max()}")

                # for name, param in self.models['decoder'].named_parameters():
                #     if param.grad is not None:  
                #         print(f"Gradient for {name}: {param.grad.max()}")

                # embed()
            print(f"Output sum: {(outputs > 0).sum()}") # 190051, 189769
            print(f"input sum: {(inputs > 0).sum()}")   # 189770, 189770
             
            print(f"Epoch {epoch + 1}/{epochs}, Loss Res: {loss_res.item()} Loss KID: {loss_kid.item()} Loss BCE: {loss_bce.item()}")
            
            if epoch % 1000 ==0:
                self.save(f"./dev/{epoch}.pth")

    def forward(self, inputs: Any) -> Any:
        latents, mean, logvar = self.models['encoder'](inputs, sample_posterior = True, return_raw = True)
        outputs = self.models['decoder'](latents)

        return outputs, mean, logvar

    def evaluate(self, test_loader: Any) -> float:
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs, mean, logvar = self.forward(inputs)
                loss_res = self.weight_res * self.criterion(outputs, targets)  # Compute loss
                loss_kid = self.weight_kid * self.KID_loss(mean, logvar)
                loss_bce = self.weight_bce * self.bce(outputs, targets)
                loss = loss_res + loss_kid + loss_bce
                total_loss += loss.item()

                print(f"Loss Res: {loss_res.item()} Loss KID: {loss_kid.item()}")
                print(f"Output sum: {(outputs > 0).sum()}") # 190051, 189769
                print(f"input sum: {(inputs > 0).sum()}")   # 189770, 189770
                #print(f"mean: {mean} logvar:{logvar}")
                
                #embed()

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
        
