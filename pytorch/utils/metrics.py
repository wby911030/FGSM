import torch

class CategoryAccuracy(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.classes = num_class
    
    def forward(self, y_pred, y_true):
        assert y_true.shape[0] == y_pred.shape[0]
        y_true = torch.nn.functional.one_hot(y_true, self.classes).to(torch.float32)
        rights = torch.sum(torch.argmax(y_true, 1) == torch.argmax(y_pred, 1)).item()
        return rights / y_true.shape[0]