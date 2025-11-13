import math

import numpy as np
import scipy
import torch
from torch import nn


class RBF(nn.Module):
    def __init__(self, n_kernels=10, mul_factor=20.0, bandwidth=None):
        """
        Initialize the RBF kernel with a specified number of kernels, multiplication factor, and bandwidth.

        Parameters
        ----------
        n_kernels : int, optional
            Number of kernels (default is 10).
        mul_factor : float, optional
            Multiplication factor for the bandwidth multipliers (default is 20.0).
        bandwidth : float, optional
            Bandwidth value for RBF kernel (default is None).

        """
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[1]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2 # (B, WIN, WIN)
        bandwidth = self.get_bandwidth(L2_distances)
        kl = 0
        for mp in self.bandwidth_multipliers:
            kl += torch.exp(-L2_distances / (mp * bandwidth))
        return kl

class RbfKernel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X, Y):

        L2_square_distances = torch.pow(torch.cdist(X, Y), 2)
        res = torch.exp(-L2_square_distances / 0.001)
        
        return res

class Matern(nn.Module):
    def __init__(self, nu=1.5, length_scale=1.0):
        """
        Initialize the Matern kernel with specified smoothness and length scale parameters.

        Parameters
        ----------
        nu : float, optional
            Smoothness parameter (default is 1.5).
        length_scale : float, optional
            Length scale parameter (default is 1.0).

        """
        super().__init__()
        self.nu = nu  # ν parameter (smoothness)
        self.length_scale = length_scale  # l parameter (length scale)


    def compute_matern(self, dists):
        """Compute the Matern kernel based on distances."""
        if self.nu == 0.5:
            K = torch.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * torch.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * torch.exp(-K)
        elif self.nu == np.inf:
            K = torch.exp(-(dists**2) / 2.0)
        else:
            K = dists.clone()
            K[K == 0.0] += 1e-9  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill_((2 ** (1.0 - self.nu)) / torch.lgamma(torch.tensor(self.nu).exp()))
            K *= tmp**self.nu
            if self.nu == 1.0:
                K *= torch.special.modified_bessel_k1(tmp)
            else: # general case; expensive to evaluate
                K *= torch.from_numpy(scipy.special.kv(self.nu, tmp.cpu().numpy())).to(dists.device)

        return K

    def forward(self, X):
        # Calculate the squared Euclidean distance between all points
        L2_distances = torch.cdist(X / self.length_scale, X / self.length_scale)  # (B, WIN, WIN)

        # Compute the Matern kernel values based on distances
        return self.compute_matern(L2_distances)

class MMDModule(nn.Module):
    def __init__(self, kernel: nn.Module):
        super().__init__()
        self.kernel = RbfKernel()
            
    def forward(self, X, Y): # X.shape == (L, WIN, CH)
        X = X.float()
        Y = Y.float()
        X_Y = torch.concatenate([X, Y], dim=1)
        K = self.kernel(X_Y, X_Y)
        
        X_size = X.shape[1]
        XX = K[:, :X_size, :X_size].mean(axis=(-1, -2))
        XY = K[:, :X_size, X_size:].mean(axis=(-1, -2))
        YY = K[:, X_size:, X_size:].mean(axis=(-1, -2))
        return (XX - 2 * XY + YY)

# class MMDModule(nn.Module):
#     def __init__(self, kernel='rbf', **kwargs):
#         """
#         Initialize the MMDModule with a specified kernel and its parameters.

#         Parameters
#         ----------
#         kernel : str, optional
#             Type of kernel to use ("rbf" or "matern", default is 'rbf').
#         **kwargs
#             Additional keyword arguments for the kernel.
        
#         """
#         super().__init__()
#         if kernel == 'rbf':
#             self.kernel = RBF(**kwargs)
#         elif kernel == 'matern':
#             self.kernel = Matern(**kwargs)
#         else:
#             raise ValueError(f'Unsupported kernel: {kernel}')

#     def forward(self, X, Y): # X.shape == (L, WIN, CH)
#         X = X.float()
#         Y = Y.float()
#         K = self.kernel(torch.concatenate([X, Y], dim=1))
#         X_size = X.shape[1]
#         XX = K[:, :X_size, :X_size].mean(axis=(-1, -2))
#         XY = K[:, :X_size, X_size:].mean(axis=(-1, -2))
#         YY = K[:, X_size:, X_size:].mean(axis=(-1, -2))
#         return XX - 2 * XY + YY
