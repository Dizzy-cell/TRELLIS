
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

        self.bce = nn.BCEWithLogistsLoss()

        for model in self.models.values():
            model.train()
        
        params = []
        for model in self.models.values():
            params.extend(model.parameters())
        optimizer = optim.Adam(params, lr=10e-6)  #10-4 

        self.optimizer = optimizer

        self.scheduler = StepLR(optimizer, step_size=100000, gamma=0.1)

    def train(self, train_loader: Any, epochs: int = 1) -> None:
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()  # Clear gradients
                outputs = self.forward(inputs)  # Forward pass
                loss = self.criterion(outputs, targets)  # Compute loss
                loss.backward()  # Backward pass

                # torch.nn.utils.clip_grad_norm_(self.models['encoder'].parameters(), max_norm=1.0)
                # torch.nn.utils.clip_grad_norm_(self.models['decoder'].parameters(), max_norm=1.0)
                self.optimizer.step()  # Update weights

                self.scheduler.step()

                # print(f"Loss: {loss.item()}")
                # embed()

                # for name, param in self.models['encoder'].named_parameters():
                #     if param.grad is not None:  
                #         print(f"Gradient for {name}: {param.grad.max()}")

                # for name, param in self.models['decoder'].named_parameters():
                #     if param.grad is not None:  
                #         print(f"Gradient for {name}: {param.grad.max()}")

                # embed()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def forward(self, inputs: Any) -> Any:
        latents = self.models['encoder'](inputs, sample_posterior = True)
        outputs = self.models['decoder'](latents)
        #outputs = nn.Sigmoid()(outputs)
        return outputs

    def evaluate(self, test_loader: Any) -> float:
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # print(f"Loss: {loss.item()}")
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
