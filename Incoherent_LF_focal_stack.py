"""
Generate incoherent focal stack from LF

Any questions about the code can be addressed to Dongyeon Kim (dongyeon93@snu.ac.kr)

If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Holographic Parallax Improves 3D Perceptual Realism 
D. Kim, S.-W. Nam, S. Choi, J.-M. Seo, G. Wetzstein, and Y. Jeong
SIGGRAPH 2024

"""

import torch
import torch.fft
import math
import configargparse
import json
import os
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from image_loader import get_image_filenames
from image_loader import TargetLoader
from Incoherent_focal_stack import compute_depthmap_dists
from prop_models import compute_filter
import params
import utils

def LF_focal_stack(target_amp, target_mask, recon_dists, prop_dists, wavelengths, aperture, channel, feature_size, roi_res, pupil_pos, pupil_rad, is_perspective=False, sideband=True, **opt):
    """
        :param target_amp: shape of (1,1,H,W,U,V), target LF tensor
        :param target_mask: shape of (1,1,H,W,U,V), target mask tensor
        :param recon_dists: len D list, propagation distance of each plane from SLM. Far to near
        :param prop_idsts: len 1 list, propagation distance of reference plane from SLM
        :param wavelengths: tuple of wavelengths
        :param feature_size: tuple, SLM pixel pitch
        :param roi_res: tuple, resolution of roi region
        :param sideband: bool. If true, consider sideband filtering by half of the y-axis 
        :param return (D,1,H,W), simulated focal stack
    """
    ny, nx, nu, nv = target_amp.shape[2:]
    Np = len(recon_dists)
    dists = [prop_dists[0] - d for d in  recon_dists]
    layer_color = torch.zeros(Np,1,ny,nx).to(dev)
    wavelength = wavelengths[channel]
    Ly = ny*feature_size[0]
    Lx = nx*feature_size[1]

    fy = torch.linspace(-1/2/feature_size[0]+1/2/Ly, 1/2/feature_size[0]-1/2/Ly,nu)
    fx = torch.linspace(-1/2/feature_size[1]+1/2/Lx, 1/2/feature_size[1]-1/2/Lx,nv)
    
    init_pixel = ()
    for i in range(len(roi_res)):
        init_pixel += ((int)(target_amp.shape[2+i]/2-roi_res[i]/2),)

    if is_perspective:
        mag = 10
        tmp_filter = compute_filter((nu*mag, nv*mag), shape='circ', aperture=pupil_rad, sideband=False,pupil_pos = pupil_pos)
        tmp_filter = tmp_filter.to(target_amp.device)

        pupil_mask = torch.zeros_like(target_amp)
        for i in range(nu):
            for j in range(nv):
                # integrate filter area
                pupil_mask[..., i,j] = torch.sum(tmp_filter[mag*i:mag*(i+1), mag*j:mag*(j+1)]) / (mag**2)

    for n in range(Np):
        layer_sum = torch.zeros(1,1,ny,nx).to(dev)
        dist = -dists[n]
        for u in range(nu):
            angle_y = torch.asin((wavelength*fy[u])) ## angle in y
            py = torch.floor(dist*torch.tan(angle_y)/feature_size[0]).int() ## pixel shift in y
            for v in range(nv):
                angle_x = torch.asin((wavelength*fx[v])) ## angle in x
                px = torch.floor(dist*torch.tan(angle_x)/feature_size[1]).int() ## pixel shift in x
                if is_perspective:
                    amp = target_amp[:,:,:,:,u,v] * target_mask[:,:,:,:,u,v] *pupil_mask[:,:,:,:,u,v]
                else:
                    amp = target_amp[:,:,:,:,u,v] * target_mask[:,:,:,:,u,v]

                start_row, end_row = (int)(init_pixel[0]+py), (int)(init_pixel[0]+py+roi_res[0])
                start_col, end_col = (int)(init_pixel[1]+px), (int)(init_pixel[1]+px+roi_res[1])

                ## Superimpose the orthographic images with pixel shifts calculated above. 
                layer_sum[:,:,start_row:end_row,start_col:end_col]= \
                    layer_sum[:,:,start_row:end_row,start_col:end_col]+ \
                    amp[:,:,start_row-py:end_row-py, start_col-px:end_col-px]
        
        layer_color[n,:,:,:] = layer_sum
    return layer_color

if __name__=='__main__':
    mm, um, nm = 1e-3, 1e-6, 1e-9

    print('-- Start focal stack generation')

    # Command line argument processing / Parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
    params.add_parameters(p, 'gd')
    opt = p.parse_args()
    opt = params.set_configs(opt)
    dev = torch.device('cuda')

    # aperture for full-color fs
    if opt['physical_iris'] is True:
        opt['aperture'] = opt['wavelengths'][2]/opt['wavelength']

    target_loader = TargetLoader(**opt)
    if opt['is_perspective']:
        lf_path = os.path.join(opt['out_path'],f'pos_{opt.pupil_pos[0]:.2f}_{opt.pupil_pos[1]:.2f}_rad_{opt.pupil_rad:.2f}')
        print(f'pos_{opt.pupil_pos[0]:.2f}_{opt.pupil_pos[1]:.2f}_rad_{opt.pupil_rad:.2f}')
    else:
        lf_path = os.path.join(opt['out_path'])
    utils.cond_mkdir(lf_path)

    for i, target in enumerate(target_loader):
        if isinstance(target, tuple):
            target_amp, target_mask = target
            target_amp = target_amp.to(dev)
            target_mask = target_mask.to(dev)
        else:
            target_amp = target.to(dev)
            target_mask = None

    LF_fs = LF_focal_stack(target_amp, target_mask, **opt)
    for D in range(len(opt['recon_dists'])):
        tmp = utils.crop_image(LF_fs[D,:,:,:].squeeze().detach().cpu().numpy(),opt['roi_res'],pytorch=False)
        img_array = np.array(tmp/np.max(tmp)*255, dtype=np.uint8)
        image = Image.fromarray(img_array)
        image.save(os.path.join(lf_path, f'{D:02d}.png'))
        

    