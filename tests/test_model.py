from BjarkeCCtemplate.models.model import myawesomemodel
import torch
import numpy as np 

def test_model():

    model = torch.load('models/model0.001_256_20.pt')
    model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
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
