import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.data import Data

class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt = None
        self.loss_func = None
        self.metrics = None
        self.device = None
        
    def compile(self, optimizer, loss_function, metrics = None, device = "cpu"):
        self.opt = optimizer
        self.loss_func = loss_function
        self.metrics = metrics
        self.device = device
        
    def _validate(self, val_data):
        total_samples = 0
        self.eval()
        running_loss = 0.0
        accuracy = 0.0
        info = dict()

        with torch.no_grad():
            for batch, data in enumerate(val_data):
                if len(data) > 2:
                    x_data, y_data = [x.to(self.device) for x in data[0:-1]], data[-1].to(self.device)
                else:
                    x_data, y_data = data[0].to(self.device), data[-1].to(self.device)
                    
                pred = self.forward(x_data)
                loss = self.loss_func(pred, y_data)

                # Accumulate validation loss
                running_loss = running_loss + loss.item() * y_data.shape[0]
                total_samples = total_samples + y_data.shape[0]
                
                if self.metrics is not None:
                    accuracy = accuracy + self.metrics(pred, y_data) * y_data.shape[0]

        info["val_loss"] = running_loss / total_samples
        
        if self.metrics is not None:
            info["val_accuracy"] = accuracy / total_samples

        return info
    
    def fit(self, x_data, y_data, batch_size, epochs, validation_split = 0):
        assert validation_split >= 0 and validation_split < 1
        if validation_split != 0:
            x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=validation_split)
            training_data = DataLoader(Data(x_train, y_train), batch_size, True)
            validation_data = DataLoader(Data(x_val, y_val), batch_size, True)
        else:           
            training_data = DataLoader(Data(x_data, y_data), batch_size, True)
        
        num_batches = len(training_data)
        history = []
        
        for epoch in range(epochs):
            accuracy = 0.0
            num_samples = 0
            
            with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', leave=True) as pbar:
                for batch, data in enumerate(training_data):
                    if len(data) > 2:
                        x_data, y_data = [x.to(self.device) for x in data[0:-1]], data[-1].to(self.device)
                    else:
                        x_data, y_data = data[0].to(self.device), data[-1].to(self.device)
                    info = dict()
                    self.train()
                    
                    # Compute prediction error
                    pred = self.forward(x_data) 
                    loss = self.loss_func(pred, y_data)

                    # Backpropagation
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    info["loss"] = loss.item()
                    
                    if self.metrics is not None:
                        self.eval()
                        num_samples = num_samples + y_data.shape[0]
                        accuracy = accuracy + self.metrics(self.forward(x_data), y_data) * y_data.shape[0]
                        if batch == num_batches - 1:
                            info["accuracy"] = accuracy / num_samples
                    
                    if validation_split != 0 and batch == num_batches - 1:
                        info.update(self._validate(validation_data))
                        
                    pbar.set_postfix(info)
                    pbar.update(1)
            
            history.append(loss.item())
        
        return history
    
    def evaluate(self, x_data, y_data):
        self.eval()
        return self.metrics(self.forward(x_data), y_data)