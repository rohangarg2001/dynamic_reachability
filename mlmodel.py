import torch
import torch.nn as nn
import numpy as np
import os

class SinusoidalActivation(nn.Module):
    def __init__(self, w0=30.0):
        super(SinusoidalActivation, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class OCR_NN(nn.Module):
    def __init__(self, options):
        super(OCR_NN, self).__init__()
        self.options = options
        self.w0 = self.options['w0']
        self.fc1 = self._siren_layer(nn.Linear(options['dim_x'], 512))
        self.fc2 = self._siren_layer(nn.Linear(512, 512))
        self.fc3 = self._siren_layer(nn.Linear(512, 512))
        self.fc4 = self._siren_layer(nn.Linear(512, 512))
        self.fc5 = nn.Linear(512, options['dim_output'])
        self.activation = SinusoidalActivation(w0=self.w0)

    def _siren_layer(self, layer):
        nn.init.uniform_(
            layer.weight, 
            -np.sqrt(6 / layer.in_features) / self.w0,
            np.sqrt(6 / layer.in_features) / self.w0
        )
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x 

class OCR_NN_ReLu(nn.Module):
    def __init__(self, options):
        super(OCR_NN_ReLu, self).__init__()
        self.options = options
        self.fc1 = nn.Linear(options['dim_x'], 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, options['dim_output'])

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x 


def compute_first_derivative(model, x):
    """
    Compute the first derivative of the model's output with respect to
    the input x (only first 3 dimensions of x are considered).
    
    Args:
        model: The neural network model.
        x: Input tensor of shape [batch_size, input_dim] with requires_grad=True.
    
    Returns:
        Tensor of shape [batch_size, 3], containing gradients with respect to
        the first 3 input dimensions.
    """
    # Ensure input tensor requires gradients
    if not x.requires_grad:
        x.requires_grad = True

    # Forward pass
    output = model(x)

    # Check if output is connected to the computational graph
    if output.grad_fn is None:
        raise RuntimeError("Model output is not connected to the computational graph.")

    # Backward pass for gradient computation
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,  # Retain graph for further operations if needed
        create_graph=False,  # No need for higher-order derivatives here
        only_inputs=True
    )[0]

    # Slice to get only first 3 dimensions of gradients
    return gradients[:, :3]

def save_model(ocr_nn, model_name, options):
    if os.path.isdir('./models/') == False:
        os.makedirs('./models/')
    
    if ocr_nn != None:
        torch.save({
            'phi_h_state_dict' : ocr_nn.state_dict(),
            'options' : options
        }, './models/' + model_name + '.pth')

def load_model(modelname, model_path='./models/'):
    model = torch.load(model_path + modelname + '.pth')

    ocr = OCR_NN(model['options'])
    ocr.load_state_dict(model['phi_h_state_dict'])

    return ocr