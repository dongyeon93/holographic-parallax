"""
Utils
"""

import math
import random
import numpy as np

import os
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.fft as tfft
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field



def crop_image(field, target_shape, pytorch=True, stacked_complex=True, lf=False):
    """Crops a 2D field, see pad_image() for details
    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if lf:
        size_diff = np.array(field.shape[-4:-2]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-4:-2]) % 2
    else:
        if pytorch:
            if stacked_complex:
                size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-3:-1]) % 2
            else:
                size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-2:]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if lf:
            return field[(..., *crop_slices, slice(None), slice(None))]
        else:
            if pytorch and stacked_complex:
                return field[(..., *crop_slices, slice(None))]
            else:
                return field[(..., *crop_slices)]
    else:
        return field


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def phasemap_8bit(phasemap, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed

    Input
    -----
    :param phasemap: input phasemap tensor, which is supposed to be in the range of [-pi, pi].
    :param inverted: a boolean value that indicates whether the phasemap is inverted.

    Output
    ------
    :return: output phasemap, with uint8 dtype (in [0, 255])
    """

    output_phase = ((phasemap + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    return phase_out_8bit


def pad_stacked_complex(field, pad_width, padval=0):
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, value=padval)
        imag = nn.functional.pad(imag, pad_width, value=0)
        return torch.stack((real, imag), -1)


def srgb_gamma2lin(im_in):
    """ converts from sRGB to linear color space """
    thresh = 0.04045
    if torch.is_tensor(im_in):
        low_val = im_in <= thresh
        im_out = torch.zeros_like(im_in)
        im_out[low_val] = 25 / 323 * im_in[low_val]
        im_out[torch.logical_not(low_val)] = ((200 * im_in[torch.logical_not(low_val)] + 11)
                                                / 211) ** (12 / 5)
    else:
        im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055) ** (12/5))

    return im_out


def srgb_lin2gamma(im_in):
    """ converts from linear to sRGB color space """
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out


def switch_lf(input, mode='elemental'):
    spatial_res = input.shape[2:4]
    angular_res = input.shape[-2:]
    if mode == 'elemental':
        lf = input.permute(0, 1, 2, 4, 3, 5)
    elif mode == 'whole':
        lf = input.permute(0, 1, 4, 2, 5, 3)  # show each view
    return lf.reshape(1, 1, *(s*a for s, a in zip(spatial_res, angular_res)))


def FT2(tensor):
    """ Perform 2D fft of a tensor for last two dimensions """
    tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
    tensor_ft_shift = torch.fft.fft2(tensor_shift, norm='ortho')
    tensor_ft = torch.fft.fftshift(tensor_ft_shift, dim=(-2,-1))
    return tensor_ft


def iFT2(tensor):
    """ Perform 2D ifft of a tensor for last two dimensions """
    tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
    tensor_ift_shift = torch.fft.ifft2(tensor_shift, norm='ortho')
    tensor_ift = torch.fft.fftshift(tensor_ift_shift, dim=(-2,-1))
    return tensor_ift


def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def phase_ramps(f, angles):
    """
    The output has the same shape of f.
    f: input field, not being used though, 
    angles: a list (or a tensor) of angles in normalized spatial frequency (e.g. [-1/2, 1/2])
    """
    ramps = []
    N = len(angles)
    for i, angle in enumerate(angles):
        x = torch.linspace(0, f.shape[-1]-1, f.shape[-1])
        y = torch.linspace(0, f.shape[-2]-1, f.shape[-2])
        Y, X = torch.meshgrid(y, x)
        phase_ramp = torch.exp(1j * 2 * math.pi * (angle[0] * Y + angle[1] * X))
        ramps.append(phase_ramp)
    ramps = torch.stack(ramps, 0).unsqueeze(1)
    
    return ramps.to(f.device)


def get_psnr_ssim(recon_amp, target_amp, multichannel=False):
    """get PSNR and SSIM metrics"""
    psnrs, ssims = {}, {}

    # amplitude
    psnrs['amp'] = psnr(target_amp, recon_amp)
    ssims['amp'] = ssim(target_amp, recon_amp, multichannel=multichannel)

    # linear
    target_linear = target_amp**2
    recon_linear = recon_amp**2
    psnrs['lin'] = psnr(target_linear, recon_linear)
    ssims['lin'] = ssim(target_linear, recon_linear, multichannel=multichannel)

    # srgb
    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    psnrs['srgb'] = psnr(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, multichannel=multichannel)

    return psnrs, ssims