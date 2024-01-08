from BjarkeCCtemplate.models.model import myawesomemodel
import torch
import numpy as np 
import pytest

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model(device):
    if not torch.cuda.is_available():
        pytest.skip("test was skipped because no cuda support")
    model = torch.load('models/model0.001_256_20.pt')
    model.eval()  # Set the model to evaluation mode

    # Assuming your input is a numpy array, convert it to a PyTorch tensor
    # For example, let's create a dummy input tensor
    input_tensor = torch.randn(1, 28, 28).to(device=device)
    model.to(device=device)
    # If your model expects a different input shape, adjust the tensor accordingly
    # For example, if your model expects a batch size and a channel dimension, you might need to reshape
    input_tensor = input_tensor.unsqueeze(0)  # Adds a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    assert output[0].shape == torch.Size([10])
    # output is your model's prediction
