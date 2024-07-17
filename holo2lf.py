"""
Implementations of the Light-field ↔ Hologram conversion. Note that lf2holo method is basically the OLAS method.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, Matthew O'Toole, G. Wetzstein.
SIGGRAPH 2022
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def holo2lf(input_field, n_fft=(9, 9), hop_length=(1, 1), win_func=None,
            win_length=None, device=torch.device('cuda'), impl='torch', predefined_h=None,
            return_h=False, h_size=(1, 1), win_type='hann', beta=12.):
    """
    Hologram to Light field transformation.
    :param input_field: input field shape of (N, 1, H, W), if 1D, set H=1.
    :param n_fft: a tuple of numbers of fourier basis.
    :param hop_length: a tuple of hop lengths to sample at the end.
    :param win_func: window function applied to each segment, default hann window.
    :param win_length: a tuple of lengths of window function. if win_length is smaller than n_fft, pad zeros to the windows.
    :param device: torch cuda.
    :param impl: implementation ('conv', 'torch', 'olas')
    :return: A 4D representation of light field, shape of (N, 1, H, W, U, V)
    """
    if impl == 'conv':
        return holo2lf_conv(input_field, predefined_h=predefined_h, hop_length=hop_length, h_size=h_size,
                            return_h=return_h, device=device)
    else:
        input_length = input_field.shape[-2:]
        batch_size, _, Ny, Nx = input_field.shape

        # for 1D input (n_fft = 1), don't take fourier transform toward that direction.
        n_fft_y = min(n_fft[0], input_length[0])
        n_fft_x = min(n_fft[1], input_length[1])

        if win_length is None:
            win_length = n_fft

        win_length_y = min(win_length[0], input_length[0])
        win_length_x = min(win_length[1], input_length[1])

        if win_func is None:
            if win_type == 'hann':
                w_func = lambda length: torch.hann_window(length + 1, device=device)[1:]
            elif win_type == 'kaiser':
                w_func = lambda length: torch.kaiser_window(window_length=length + 1, periodic=True, beta=beta, device=device)[1:]
            # w_func = lambda length: torch.ones(length)
            win_func = torch.ger(w_func(win_length_y), w_func(win_length_x))

        win_func = win_func.to(input_field.device)
        win_func /= win_func.sum()

        if impl == 'torch':
            # 1) use STFT implementation of PyTorch
            if len(input_field.squeeze().shape) > 1:  # with 2D input
                # input_field = input_field.view(-1, input_field.shape[-1])  # merge batch & y dimension
                input_field = input_field.reshape(np.prod(input_field.size()[:-1]), input_field.shape[-1])  # merge batch & y dimension

                # take 1D stft along x dimension
                stft_x = torch.stft(input_field, n_fft=n_fft_x, hop_length=hop_length[1], win_length=win_length_x,
                                    # onesided=False, window=torch.fft.ifftshift(win_func[win_length_y//2, :]), pad_mode='constant',
                                    onesided=False, window=win_func[win_length_y//2, :], pad_mode='constant',
                                    normalized=False, return_complex=True)

                if n_fft_y > 1:  # 4D light field output
                    stft_x = stft_x.reshape(batch_size, Ny, n_fft_x, Nx//hop_length[1]).permute(0, 3, 2, 1)
                    stft_x = stft_x.contiguous().view(-1, Ny)

                    # take one more 1D stft along y dimension
                    stft_xy = torch.stft(stft_x, n_fft=n_fft_y, hop_length=hop_length[0], win_length=win_length_y,
                                        #  onesided=False, window=torch.fft.ifftshift(win_func[:, win_length_x//2]), pad_mode='constant',
                                         onesided=False, window=win_func[:, win_length_x//2], pad_mode='constant',
                                         normalized=False, return_complex=True)

                    # reshape tensor to (N, 1, Y, X, fy, fx)
                    stft_xy = stft_xy.reshape(batch_size, Nx//hop_length[1], n_fft[1], n_fft[0], Ny//hop_length[0])
                    stft_xy = stft_xy.unsqueeze(1).permute(0, 1, 5, 2, 4, 3)
                    freq_space_rep = torch.fft.fftshift(stft_xy, (-2, -1))

                else:  # 3D light field output
                    stft_xy = stft_x.reshape(batch_size, Ny, n_fft_x, Nx//hop_length[1]).permute(0, 1, 3, 2)
                    stft_xy = stft_xy.unsqueeze(1).unsqueeze(4)
                    freq_space_rep = torch.fft.fftshift(stft_xy, -1)

            else:  # with 1D input  -- to be deprecated
                freq_space_rep = torch.stft(input_field.squeeze(),
                                            # n_fft=n_fft, hop_length=hop_length, onesided=False, window=torch.fft.ifftshift(win_func),
                                            n_fft=n_fft, hop_length=hop_length, onesided=False, window=win_func,
                                            win_length=win_length, normalized=False, return_complex=True)
        elif impl == 'olas':
            # 2) Our own implementation:
            # slide 1d representation to left and right (to amount of win_length/2) and stack in another dimension
            overlap_field = torch.zeros(*input_field.shape[:2],
                                        (win_func.shape[0] - 1) + input_length[0],
                                        (win_func.shape[1] - 1) + input_length[1],
                                        win_func.shape[0], win_func.shape[1],
                                        dtype=input_field.dtype).to(input_field.device)

            # slide the input field
            for i in range(win_length_y):
                for j in range(win_length_x):
                    overlap_field[..., i:i+input_length[0], j:j+input_length[1], i, j] = input_field

            # toward the new dimensions, apply the window function and take fourier transform.
            win_func = win_func.reshape(1, 1, 1, 1, *win_func.shape)
            win_func = win_func.repeat(*input_field.shape[:2], *overlap_field.shape[2:4], 1, 1)
            overlap_field *= win_func  # apply window

            # take Fourier transform (it will pad zeros when n_fft > win_length)
            # apply no normalization since window is already normalized
            if n_fft_y > 1:
                overlap_field = torch.fft.fftshift(torch.fft.ifft(overlap_field, n=n_fft_y, norm='forward', dim=-2), -2)
            freq_space_rep = torch.fft.fftshift(torch.fft.ifft(overlap_field, n=n_fft_x, norm='forward', dim=-1), -1)

            # take every hop_length columns, and when hop_length == win_length it should be HS.
            freq_space_rep = freq_space_rep[:,:, win_length_y//2:win_length_y//2+input_length[0]:hop_length[0],
                                                 win_length_x//2:win_length_x//2+input_length[1]:hop_length[1], ...]

        return freq_space_rep.abs()**2  # LF = |U|^2
    


    
def holo2lf_conv(input_field, predefined_h=None, hop_length=(1,1), h_size=(1, 1), return_h=False, device=torch.device('cuda')):
    # TODO: define holo2lf_conv
    pass