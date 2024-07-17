"""
Generate incoherent focal stack from RGBD
Approximate occlusion by blending

Any questions about the code can be addressed to Dongyeon Kim (dongyeon93@snu.ac.kr)

If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
High-contrast, speckle-free, true 3D holography via binary CGH optimization
B. Lee, D. Kim, S. Lee, C. Chen and B. Lee.
Scientific Reports 2022

"""
import torch
import torch.fft
import math
import configargparse
import json
import os
import imageio as imageio
import numpy as np

# import params
import utils

def ITF(field, z, dx, dy, wavelength, sideband=True, aperture=1.0):
    """
    Simulate Incoherent Transfer Fuction
        :param field: shape of (H,W), tensor
        :param z: scalar, propagation distance
        :param dx, dy: pixel pitch
        :return out: shape of (1,1,H,W)
    """
    m, n = field.shape
    Lx, Ly = float(dx * n), float(dy * m)

    angX = math.asin(wavelength / (2 * dx))
    angY = math.asin(wavelength / (2 * dy))
    marX = math.fabs(z) * math.tan(angX)
    marX = math.ceil(marX / dx)
    marY = math.fabs(z) * math.tan(angY)
    marY = math.ceil(marY / dy)
    pad_field = torch.nn.functional.pad(field, (marX, marX, marY, marY)).to(field.device)

    fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * Ly), 1 / (2 * dy) - 0.5 / (2 * Ly), m+ 2*marY).to(field.device)
    fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * Lx), 1 / (2 * dx) - 0.5 / (2 * Lx), n+ 2*marX).to(field.device)
    dfx = (1 / dx) / n
    dfy = (1 / dy) / m
    fY, fX = torch.meshgrid(fy, fx, indexing='ij')

    # aperture for bandlimit
    if sideband:
        aperture = (aperture/2, aperture)
    else:
        aperture = (aperture, aperture)

    # rectangular fourier filter
    nfX = fX / torch.max(fX.abs())
    nfY = fY / torch.max(fY.abs())
    BL_FILTER = (nfY.abs() < aperture[0]) * (nfX.abs() < aperture[1])
    # energy normalization
    BL_FILTER = BL_FILTER / torch.sqrt(torch.sum(BL_FILTER) / torch.numel(BL_FILTER))

    # set transfer function
    GammaSq = (1 / wavelength) ** 2 - fX ** 2 - fY ** 2
    TF = torch.exp(-2 * 1j * math.pi * torch.sqrt(GammaSq) * z)
    TF = TF * BL_FILTER
    cpsf = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(TF), norm='ortho'))  # coherent psf
    ipsf = torch.abs(cpsf) ** 2  # incoherent psf
    OTF = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ipsf), norm='ortho'))

    max_fx = 1 / (wavelength * ((2 * dfx * z) ** 2 + 1) ** 0.5)
    max_fy = 1 / (wavelength * ((2 * dfy * z) ** 2 + 1) ** 0.5)
    FT = (torch.abs(fX) < max_fx) * (torch.abs(fY) < max_fy)  # Cutting aliasing
    AS = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(pad_field), norm='ortho'))
    PropagatedField = abs(torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(AS * OTF * FT), norm='ortho')))
    out = PropagatedField[marY:m+marY,marX:n+marX]  #imcrop

    return out.unsqueeze(0).unsqueeze(0)


def Incoherent_focal_stack(image, depth_mask, prop_dists, wavelengths, feature_size, aperture, sideband=True, alpha=0.5, **opt):
    """
        :param image: shape of (C,H,W), target rgb tensor
        :param depth_mask: shape of (D,H,W), target depthmap mask
        :param prop_dists: len D list, propagation distance of each plane. Far to near.
        :param wavelengths: tuple of wavelengths
        :param feature_size: tuple, SLM pixel ptich
        :param aperture: tuple, aperture size of each color channel
        :param sideband: bool. If true, consider sideband filtering by half the y-axis aperture size
        :param alpha: coefficient for occlusion blending
        :return: shape of (D,C,H,W), simulated focal stack
    """
    n_channel, ny, nx = image.shape
    Np = len(prop_dists)
    dev = image.device

    layer_sum_color = torch.zeros(Np, n_channel, ny, nx).to(dev) # [D,C,H,W]
    for ch, wavelength in enumerate(wavelengths):
        for n in range(Np):
            layer_sum = torch.zeros(1, 1, ny, nx).to(dev) # [1,1,H,W]
            mask = torch.zeros(1, Np, ny, nx).to(dev) # [1,D,H,W]

            # To achieve k-th focal stack, apply incoherent propagation for all Np layers
            for k in range(Np):
                dz = prop_dists[n] - prop_dists[k]
                k_depth_mask = depth_mask[k, :, :] # [1,1,H,W] depth-k mask

                # Generate occlusion mask function by sequential propagation
                mask[:, k, :, :] = ITF(k_depth_mask.squeeze(), dz, *feature_size, wavelength, sideband, aperture[ch])
                mask[:, k, :, :] = 1.0 - mask[:, k, :, :] / torch.max(mask[:, k, :, :])
                mask[:, k, :, :] = torch.nan_to_num(mask[:, k, :, :], 1.0)

                # Incoherent propagation and Summation
                layer_intensity = (image[ch,:,:].abs() ** 2).unsqueeze(0) * k_depth_mask # [1,1,H,W]
                layer_intensity_prop = ITF(layer_intensity.squeeze(), dz, *feature_size, wavelength, sideband, aperture[ch])
                if k == 0: # first layer
                    layer_sum = (1.0 * mask[:,k,:,:]) * layer_sum + layer_intensity_prop
                elif k == (Np - 1): # last layer
                    layer_sum = (alpha * mask[:,k,:,:] + (1.0 - alpha) * mask[:,k-1,:,:]) * layer_sum \
                                    + alpha * layer_intensity_prop
                else:
                    layer_sum = (alpha * mask[:,k,:,:] + (1.0 - alpha) * mask[:,k-1,:,:]) * layer_sum + layer_intensity_prop
            layer_sum_color[n, ch, :, :] = torch.sqrt(layer_sum.abs()) 
    return layer_sum_color / torch.max(layer_sum_color)


def compute_depthmap_dists(num_planes, depthmap_dists_range, eyepiece, prop_dists):
    """
    Calculate propagation and division dists for focal stack planes
        :param num_planes: scalar, number of fs planes
        :param depthmap_dists_range: scalar, total depth range of depthmap in mm. Reference plane is in the middle of this range
        :param eyepiece: scalar, focal length of eyepiece
        :param prop_dists: list of len 1, propagation distance of reference plane
        :return fs_dists: len num_planes list, propagation distance of fs planes in meters
        :return division_dists: len (num_planes-1) list, normalized to [0,1]. Depthmap is converted to mask using this
    """
    # dist to diopter
    far_near_dists = [eyepiece, eyepiece - depthmap_dists_range]
    far_near_diopters = [1/dist - 1/eyepiece for dist in far_near_dists]

    # linear spacing diopters
    division_diopters = np.linspace(*far_near_diopters, 2 * num_planes - 1)    
    prop_diopters = division_diopters[0::2]
    division_diopters = division_diopters[1::2]

    # diopter to dist
    ref_plane_dist = depthmap_dists_range / 2
    fs_dists = [prop_dists[0] - ref_plane_dist + (eyepiece - 1 / (d + 1/eyepiece)) for d in prop_diopters]
    division_dists = [1 / (d + 1/eyepiece) for d in division_diopters]

    # normalize division_dists to [0,1]
    far, near = far_near_dists
    division_dists = [(dist - near) / (far - near) for dist in division_dists]

    print(f'- clip dists diopters: {[round(d*1e2)/1e2 for d in far_near_diopters]}')
    print(f'- depthmap division diopters: {[round(d*1e2)/1e2 for d in division_diopters]}')
    print(f'- depthmap division dists: {[round(d*1e2)/1e2 for d in division_dists]}, normalized')
    print(f'- fs plane dists: {[round(d*1e3,2) for d in fs_dists]} mm')

    return fs_dists, division_dists


def read_rgbd(rgb_path, depth_path, close_is_0=True):
    # read rgb
    im = imageio.imread(rgb_path)
    im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

    # linearize intensity and convert to amplitude
    im = utils.srgb_gamma2lin(im)
    im = np.sqrt(im)  # to amplitude
    
    # move channel dim to torch convention
    im = np.transpose(im, axes=(2, 0, 1))
    
    # read depthmap
    depthmap = imageio.imread(depth_path)[:,:,0] # grayscale
    depthmap = utils.im2float(depthmap, dtype=np.float64)  # convert to double, max 1

    if close_is_0 is False:
        depthmap = 1.0 - depthmap

    # numpy to float
    im = torch.from_numpy(im).float()
    depthmap = torch.from_numpy(depthmap).float()

    return im, depthmap # shape of [3,H,W], [H,W]


def gen_depthmap_mask(depthmap, division_dists, **opt):
    """
    Divide [H,W] depthmap into [D,H,W] depthmap mask
    """
    division_dists = [2.0, *division_dists, -1.0]

    # generate mask from far to close
    depthmap_mask = []
    for idx in range(len(division_dists) - 1):
        # distance range
        far_dist = division_dists[idx]
        near_dist = division_dists[idx+1]
    
        tmp_mask = torch.ones_like(depthmap)
        mask_idx = (depthmap > far_dist)
        tmp_mask[mask_idx] = 0

        mask_idx = (depthmap <= near_dist)
        tmp_mask[mask_idx] = 0

        depthmap_mask.append(tmp_mask.unsqueeze(0))
    depthmap_mask = torch.cat(depthmap_mask, dim=0)

    return depthmap_mask # shape of [D,H,W]


def save_dict_as_txt(path, dict):
    """ Save dictionary as txt file
    path = 'dir/fname.txt'
    WARNING: only scalar or list acceptable
    """
    with open(path, 'w') as file:
        json.dump(dict, file, indent=2)


if __name__=='__main__':
    import params
    from image_loader import get_image_filenames
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
        opt['aperture'] = [opt['wavelengths'][2] / w for w in opt['wavelengths']]
    else:
        opt['aperture'] = [opt['aperture']] * 3

    # load all image names in dir
    image_filenames = get_image_filenames(opt['data_path']) 
    image_filenames.sort()
    print(f'- targets: {image_filenames}')

    # iterate for images
    for idx in range(len(image_filenames) // 2):
        # get paths
        depth_path, rgb_path = image_filenames[2*idx:2*(idx+1)]
        _, image_name = os.path.split(rgb_path)
        image_name = image_name[:-8]
        
        print(f'----Loading: {image_name}')
        print(f'RGB: {rgb_path}, Depthmap: {depth_path}')
        
        # read target
        target_image, target_depthmap = read_rgbd(rgb_path, depth_path, opt['close_is_0'])
        # move to GPU
        target_image = target_image.to(dev)
        target_depthmap = target_depthmap.to(dev)

        # gen target mask
        target_mask = gen_depthmap_mask(target_depthmap, opt['division_dists'])

        # focal stack generation
        fs = Incoherent_focal_stack(target_image, target_mask, **opt)

        # save target
        fs_path = os.path.join(opt['out_path'], f'{image_name}')
        utils.cond_mkdir(fs_path)
        for D in range(opt['num_planes']):
            imageio.imwrite(os.path.join(fs_path, f'{D:02d}.png'), fs[D,...].permute(1,2,0).cpu().detach().numpy())

        # save parameters
        save_dict_as_txt(os.path.join(fs_path, 'params.txt'), opt)
