"""
Propagation models
"""
import torch
from torchvision.transforms.functional import resize as resize_tensor
import math
import utils
import random
import imageio
import os
import torch.fft as tfft

def prop_model(prop_model, **opt):
    if prop_model.lower() in ('sideband', 'side_band'):
        forward_prop = prop_sideband(**opt)
    elif prop_model.lower() == 'asm':
        forward_prop = ASM(**opt)
    else:
        print('  - prop model is None ....')
        forward_prop = None

    return forward_prop

class prop_sideband(torch.nn.Module):
    """
    Propagation with sideband filtering, suitable for amplitude enconding
        :param feature size: a tuple (dy,dx), slm pixel pitch
        :param prop_dists: a tuple (d1, d2, ...), propagation distances             
        :param wavelength: a scalar, wavelength of light source
        :param sideband_margin: a scalar, margin pixels to avoid DC noise
    """
    def __init__(self, feature_size, prop_dists, wavelength, 
                 sideband_margin=0, aperture=1.0, stoch_pupil=False, **opt):
        super(prop_sideband, self).__init__()
        self.sideband_margin = sideband_margin
        self.aperture = aperture
        self.ASM = ASM(feature_size, prop_dists, wavelength)
        self.F_filter = None
        self.stoch_pupil = stoch_pupil
        if self.stoch_pupil:
            print('  - stochastic pupil sampling ...')
        self.opt = opt

    def forward(self, input_field, F_filter=None, full_recon=False):
        """
        :param input_field: input field shape of (N,1,H,W)
        :param F_filter: Fourier filter shape of (2H,2W). If None, use ideal rect filter
        :return field: output field shape of (N,D,H,W), D is len(self.prop_dists)
        """
        # TODO: you can cache the fileter or fourier kernel for speedup
        field = self.sideband_filtering(input_field, F_filter)
        field = self.ASM(field)

        if (self.stoch_pupil and not full_recon) or self.opt['is_perspective']:
            field = self.pupil_aware(field, min_pupil_size=self.opt['min_pupil_size'],
                                            max_pupil_size=self.opt['max_pupil_size'],
                                            pupil_range=self.opt['pupil_range'])
        return field

    def sideband_filtering(self, input_field, F_filter=None):
        """
        Perform sideband filtering in Fourier domain
        Mask lower half and shift upper half to the center
        """
        # zero padding
        pad_shape = [2 * size for size in input_field.shape[-2:]]
        field = utils.pad_image(input_field, pad_shape, stacked_complex=False)
        ft_field = utils.FT2(field)
        
        # apply Fourier filter
        if F_filter is None:
            # ideal rect filter
            if self.F_filter is None:
                self.F_filter = compute_filter(field_res=pad_shape,
                                                shape='rect', 
                                                aperture=self.aperture, 
                                                sideband=True).to(input_field.device)
            F_filter = self.F_filter

        ft_field = ft_field * F_filter.view(* [1] * len(input_field.shape[:-2]), *F_filter.shape[-2:])

        # mask and shift frequency domain (y-axis)
        ft_out_field = torch.zeros_like(ft_field) # allocate output

        # calculate margin pixels
        margin = self.sideband_margin
        ny = pad_shape[-2]
        shifted_pixels = round(ny/4 * self.aperture)

        # use lower half
        ft_out_field[..., -(ny//2 - margin) - shifted_pixels: -shifted_pixels, :] = ft_field[..., ny//2 + margin:, :]
            
        # iFT and crop to original size
        out_field = utils.iFT2(ft_out_field)
        out_field = utils.crop_image(out_field, input_field.shape[-2:], stacked_complex=False)
        return out_field
    
    def pupil_aware(self, field, min_pupil_size=0.1, max_pupil_size=0.3, pupil_range=(1.0, 1.0)):
        if self.opt['is_perspective']:
            # perspective recon with pupil parameters of params.py
            pupil_size = self.opt['pupil_rad']
            pupil_pos = self.opt['pupil_pos']
        else:
            # randomized pupil parameters of pupil-aware holography
            pupil_size = random.uniform(min_pupil_size, max_pupil_size)
            pupil_pos = (random.uniform(-pupil_range[0], pupil_range[0]),
                        random.uniform(-pupil_range[1], pupil_range[1]))

        field, _ = view_from_pupil(field, pupil_pos=pupil_pos, pupil_rad=pupil_size)
        return field
    
    def prop(self, u_in, H, linear_conv=True, padtype='zero'):
        if linear_conv:
            # preprocess with padding for linear conv.
            input_resolution = u_in.size()[-2:]
            conv_size = [i * 2 for i in input_resolution]
            if padtype == 'zero':
                padval = 0
            elif padtype == 'median':
                padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
            u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

        U1 = tfft.fftshift(tfft.fftn(u_in, dim=(-2, -1), norm='ortho'), (-2, -1))
        U2 = U1 * H
        u_out = tfft.ifftn(tfft.ifftshift(U2, (-2, -1)), dim=(-2, -1), norm='ortho')

        if linear_conv:
            u_out = utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)

        return u_out


def compute_filter(field_res, shape='rect', aperture=1.0, sideband=False, pupil_pos=(0, 0), pupil_rad=1.0):
    """
    Generate Fourier filter
        :param shape: str, shape of fourier filter (rect, circ ..)
        :param aperture: a scalr or list for aperture size. If 1.0, use full aperture
        :param sideband: bool. If True, assume sideband filtering and half the aperture of y axis
    """
    # sampling
    ny, nx = field_res
    
    # zero-centered coordinate
    ix = torch.linspace(-nx/2, nx/2, nx)
    iy = torch.linspace(-ny/2, ny/2, ny)
    Y,X = torch.meshgrid(iy, ix, indexing='ij')

    # normalize 
    X = X / torch.max(X.abs())
    Y = Y / torch.max(Y.abs())

    if not isinstance(aperture, list):
        aperture = [aperture, aperture]

    # consider sideband filtering
    if sideband:
        Y = Y - aperture[0]/2 # shift center
        aperture[0] = aperture[0]/2

    # compute filter
    
    rect_filter = (Y.abs() <aperture[0]) * (X.abs() < aperture[1])
    circ_filter = (Y/aperture[1] - pupil_pos[0])**2 + (X/aperture[1] -pupil_pos[1])**2 < pupil_rad**2
        
    if shape == 'rect':
        F_filter = rect_filter
    elif shape == 'circ':
        F_filter = circ_filter
    else:
        raise ValueError(f'Unsupported filter shape: {shape}')

    return F_filter


class ASM(torch.nn.Module):
    """
    Angular spectrum method
        :param feature size: a tuple (dy,dx), slm pixel pitch
        :param prop_dists: a tuple (d1, d2, ...), propagation distances    
        :param wavelength: a scalar, wavelength of light source
    """
    def __init__(self, feature_size, prop_dists, wavelength, **kwargs):
        super(ASM, self).__init__()        
        self.feature_size = feature_size
        self.prop_dists = prop_dists
        self.wavelength = wavelength
        self.Hs = None # transfer functions

    def forward(self, input_field):
        """ 
        :params input_field: complex-valued input field shape of (N,1,H,W) 
        :return u2: complex-valued ouptu field shape of (N,D,H,W), D is len(self.prop_dists)
        """
        # initialize transfer functions
        if self.Hs is None:
            self.compute_Hs(input_field)

        # zero-padding
        pad_shape = self.Hs.shape[-2:]
        u1 = utils.pad_image(input_field, pad_shape, stacked_complex=False)

        # propagation
        U1 = utils.FT2(u1)
        U2 = U1 * self.Hs
        u2 = utils.iFT2(U2)
        u2 = utils.crop_image(u2, input_field.shape[-2:], stacked_complex=False)
        
        return u2

    def compute_Hs(self, input_field):
        # variables
        dy, dx = self.feature_size
        ny, nx = input_field.shape[-2:]
        pad_ny, pad_nx = 2*ny, 2*nx

        # frequency domain sampling
        dfx = 1/(pad_nx * dx)
        dfy = 1/(pad_ny * dy)
        # freuency coordinate
        # ix = torch.arange(math.ceil(-pad_nx/2),math.ceil(pad_nx/2))
        # iy = torch.arange(math.ceil(-pad_ny/2),math.ceil(pad_ny/2))
        # zero-centered coordinate
        ix = torch.linspace(-pad_nx/2, pad_nx/2, pad_nx)
        iy = torch.linspace(-pad_ny/2, pad_ny/2, pad_ny)
        FY, FX = torch.meshgrid(iy*dfy, ix*dfx, indexing='ij')

        Hs = []
        for dist in self.prop_dists:
            # transfer function
            H_exp = 2 * torch.pi * dist * torch.sqrt((1/self.wavelength**2 - FX**2 - FY**2))
            H_exp = torch.reshape(H_exp, [* [1]*len(input_field.shape[:-2]), pad_ny, pad_nx]).to(input_field.device)
            # bandlimited ASM
            fy_max = 1/math.sqrt((2*dist*(1/(dy*pad_ny)))**2 + 1) / self.wavelength
            fx_max = 1/math.sqrt((2*dist*(1/(dx*pad_nx)))**2 + 1) / self.wavelength
            H_filter = (torch.abs(FX)<fx_max) & (torch.abs(FY)<fy_max)
            H_filter = torch.reshape(H_filter, [* [1]*len(input_field.shape[:-2]), pad_ny, pad_nx]).to(input_field.device)
            # transfer function for single dist
            H = H_filter * torch.exp(1j * H_exp)

            # append to list
            Hs.append(H)
        self.Hs = torch.cat(Hs, dim=-3) # shape of (N,D,H,W)


def compute_eyebox(input_field, resize=True):
    """
    Eyebox = time-muliplexed Fourier domain
    If resize is True, eyebox is resized to square
    """
    eyebox = utils.FT2(input_field).abs()
    eyebox = (eyebox ** 2).mean(dim=0, keepdims=True).sqrt() # TM

    # resize to square
    if resize:
        eyebox = resize_tensor(eyebox, [min(*eyebox.shape[-2:])] * 2)
    
    return eyebox


def view_from_pupil(input_field, pupil_pos=(0,0), pupil_rad=0.5,aperture=1.0):
    """
    Simulate perspective view by cropping pupil from eyebox
    NOTE: Fourier plane is shifted to center the pupil
    Simply FT -> crop -> shift to center -> iFT
    Coordination of eyebox normalized to [-1,1]
        :param input_field: shape of (N,D,H,W)
        :param pupil_pos: a tuple (py, px), positoin of pupil center
        :param pupil_rad: a scalar, radius of pupil
        :return recon_amp: shape of (1,D,H,W), shape of reconstructed view
        :return pupil: shape of (min(H,W), min(H,W)), pupil mask
    """
    # zero padding
    ny, nx = input_field.shape[-2:]
    pad_ny, pad_nx = 2*ny, 2*nx
    pad_field = utils.pad_image(input_field, (pad_ny, pad_nx), stacked_complex=False)

    # zero-centered domain
    ix = torch.linspace(-pad_nx/2, pad_nx/2, pad_nx)
    iy = torch.linspace(-pad_ny/2, pad_ny/2, pad_ny)
    Y,X = torch.meshgrid(iy, ix, indexing='ij')

    # normalize 
    X = X / torch.max(X.abs())
    Y = Y / torch.max(Y.abs())

    # pupil mask
    pupil_bool = (Y - pupil_pos[0]*aperture) ** 2 + (X - pupil_pos[1]*aperture) **2 < (pupil_rad*aperture) ** 2
    pupil_mask = torch.zeros(pad_ny, pad_nx).to(input_field.device)
    pupil_mask[pupil_bool>0] = 1.0

    # apply pupil
    eyebox = utils.FT2(pad_field)
    eyebox = eyebox * pupil_mask.view( * [1]*(len(input_field.shape)-2), pad_ny, pad_nx)
    recon = utils.crop_image(utils.iFT2(eyebox), (ny,nx), stacked_complex=False)

    return recon, pupil_mask
