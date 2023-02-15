import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# The structure of Auto-Encoder
class encoder_mlp(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_mlp, self).__init__()
        self.layer = nn.Linear(t_len, op_size)
    def forward(self, x):
        x = self.layer(x)
        return x

class decoder_mlp(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_mlp, self).__init__()
        self.layer = nn.Linear(op_size, t_len)
    def forward(self, x):
        x = self.layer(x)
        return x

class encoder_conv1d(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_conv1d, self).__init__()
        self.layer = nn.Conv1d(t_len, op_size,1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.layer(x)
        x = x.permute([0,2,1])
        return x

class decoder_conv1d(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_conv1d, self).__init__()
        self.layer = nn.Conv1d(op_size, t_len,1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.layer(x)
        x = x.permute([0,2,1])
        return x

class encoder_conv2d(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(t_len, op_size,1)
    def forward(self, x):
        x = x.permute([0,3,1,2])
        x = self.layer(x)
        x = x.permute([0,2,3,1])
        return x

class decoder_conv2d(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(op_size, t_len,1)
    def forward(self, x):
        x = x.permute([0,3,1,2])
        x = self.layer(x)
        x = x.permute([0,2,3,1])
        return x

# Koopman 1D structure
class Koopman_Operator1D(nn.Module):
    def __init__(self, op_size, modes_x = 16):
        super(Koopman_Operator1D, self).__init__()
        self.op_size = op_size
        self.scale = (1 / (op_size * op_size))
        self.modes_x = modes_x
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(op_size, op_size, self.modes_x, dtype=torch.cfloat))
    # Complex multiplication
    def time_marching(self, input, weights):
        # (batch, t, x), (t, t+1, x) -> (batch, t+1, x)
        return torch.einsum("btx,tfx->bfx", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]
        # Fourier Transform
        x_ft = torch.fft.rfft(x)
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes_x] = self.time_marching(x_ft[:, :, :self.modes_x], self.koopman_matrix)
        #Inverse Fourier Transform
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class KNO1d(nn.Module):
    def __init__(self, encoder, decoder, op_size, modes_x = 16, decompose = 4, linear_type = True, normalization = False):
        super(KNO1d, self).__init__()
        # Parameter
        self.op_size = op_size
        self.decompose = decompose
        # Layer Structure
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = Koopman_Operator1D(self.op_size, modes_x = modes_x)
        self.w0 = nn.Conv1d(op_size, op_size, 1)
        self.linear_type = linear_type # If this variable is False, activate function will be worked after Koopman Matrix
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(op_size)
    def forward(self, x):
        # Reconstruct
        x_reconstruct = self.enc(x)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)
        # Predict
        x = self.enc(x) # Encoder
        x = torch.tanh(x)
        x = x.permute(0, 2, 1)
        x_w = x
        for i in range(self.decompose):
            x1 = self.koopman_layer(x) # Koopman Operator
            if self.linear_type:
                x = x + 1
            else:
                x = torch.tanh(x + x1)
        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)
        x = x.permute(0, 2, 1)
        x = self.dec(x) # Decoder
        return x, x_reconstruct

# Koopman 2D structure
class Koopman_Operator2D(nn.Module):
    def __init__(self, op_size, modes_x, modes_y):
        super(Koopman_Operator2D, self).__init__()
        self.op_size = op_size
        self.scale = (1 / (op_size * op_size))
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(op_size, op_size, self.modes_x, self.modes_y, dtype=torch.cfloat))

    # Complex multiplication
    def time_marching(self, input, weights):
        # (batch, t, x,y ), (t, t+1, x,y) -> (batch, t+1, x,y)
        return torch.einsum("btxy,tfxy->bfxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Fourier Transform
        x_ft = torch.fft.rfft2(x)
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.time_marching(x_ft[:, :, :self.modes_x, :self.modes_y], self.koopman_matrix)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self.time_marching(x_ft[:, :, -self.modes_x:, :self.modes_y], self.koopman_matrix)
        #Inverse Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class KNO2d(nn.Module):
    def __init__(self, encoder, decoder, op_size, modes_x = 10, modes_y = 10, decompose = 6, linear_type = True, normalization = False):
        super(KNO2d, self).__init__()
        # Parameter
        self.op_size = op_size
        self.decompose = decompose
        self.modes_x = modes_x
        self.modes_y = modes_y
        # Layer Structure
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = Koopman_Operator2D(self.op_size, self.modes_x, self.modes_y)
        self.w0 = nn.Conv2d(op_size, op_size, 1)
        self.linear_type = linear_type # If this variable is False, activate function will be worked after Koopman Matrix
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(op_size)
    def forward(self, x):
        # Reconstruct
        x_reconstruct = self.enc(x)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)
        # Predict
        x = self.enc(x) # Encoder
        x = torch.tanh(x)
        x = x.permute(0, 3, 1, 2)
        x_w = x
        for i in range(self.decompose):
            x1 = self.koopman_layer(x) # Koopman Operator
            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)
        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)
        x = x.permute(0, 2, 3, 1)
        x = self.dec(x) # Decoder
        return x, x_reconstruct
