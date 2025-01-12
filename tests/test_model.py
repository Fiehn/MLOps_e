import torch
import src.mlops.model 

def test_model():
    model = src.mlops.model.Model()
    x = torch.rand(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)

