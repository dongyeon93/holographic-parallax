"""
Field optimization using various supervision techniques, 
"""
import os
import params
import algorithms
import image_loader
import torch
import configargparse
import utils
from torchvision.transforms.functional import resize as resize_tensor
from PIL import Image
import numpy as np

import scipy.io as sio
import imageio.v2 as imageio
from holo2lf import holo2lf
import prop_models
import quantization
from torch.utils.tensorboard import SummaryWriter


def get_lf(field, **opt):    
    field = utils.crop_image(field, opt['roi_res'], stacked_complex=False)
    lf = holo2lf(field, n_fft=opt['n_fft'], hop_length=opt['hop_len'],
                 win_length=opt['win_len'], device=field.device, impl='torch').sqrt()
    lf = (lf ** 2).mean(dim=0, keepdims=True).sqrt()
    return lf

# NOTE: gen_target_mask moved to image_loader.py

def imsave_tensor(path, image, color=False):
    assert torch.is_tensor(image), 'image is not tensor'
    image = image.squeeze()
    
    # 3 channel RGB image
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = image.permute(1,2,0)
        color = True

    # imageio.imwrite(path, image.cpu().detach().numpy()) ## imageio normalizes min-max

    tmp = image.cpu().detach().numpy()
    img_array = np.array(tmp/np.max(tmp)*255, dtype=np.uint8) ## PIL save 
    if color:
        image = Image.fromarray(img_array,"RGB")
    else:
        image = Image.fromarray(img_array)

    image.save(path)

def save_torch_pt(path, image):
    assert torch.is_tensor(image), 'image is not tensor'
    image = image.squeeze()
    
    # 3 channel RGB image
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = image.permute(1,2,0)
    
    torch.save(image.cpu().detach(), path)


def visualize_and_save(batch_idx, results, target_mask, opt):

    if opt.is_perspective ==True:
        print(f'-- Saving to {opt.out_path}')
        print(f'-- Saving perspective view for channel: {opt.channel}, pupil pos: ({opt.pupil_pos[0]:.2f},{opt.pupil_pos[1]:.2f}), rad: {opt.pupil_rad:.2f}')
        
    
    final_field = results['recon_field']
    final_slm = results['final_phase']

    # lightfield
    if opt.target_type == '4d':
        # recon lf
        final_amp = get_lf(final_field, **opt)
        lf = utils.switch_lf(final_amp, 'whole')
        lf = resize_tensor(lf, opt.roi_res) # resize before save
        imsave_tensor(os.path.join(opt.out_path, f'{batch_idx}_lf.png'), lf) 

        # target mask
        # draw border line
        target_mask[...,  0:10, :, :,:] = 0
        target_mask[..., -10:, :, :,:] = 0
        target_mask[...,  :, 0:10, :,:] = 0
        target_mask[...,  :,-10:, :,:] = 0
        target_mask = utils.switch_lf(target_mask, 'whole')
        target_mask = resize_tensor(target_mask, opt.roi_res)
        imsave_tensor(os.path.join(opt.out_path, f'{batch_idx}_lf_mask.png'), target_mask)

    # captured amplitude
    if opt['citl']:
        captured_amp = results['recon_amp']
        captured_amp = torch.clip(captured_amp, 0, 1)
        for d in range(captured_amp.shape[0]):
            imsave_tensor(os.path.join(opt.out_path, f'{batch_idx}_captured_{d}.png'), captured_amp[:,d,:,:])

    # multi-plane results
    D = final_field.shape[1]
    for d in range(D):
        single_field = final_field[:, d:d+1, :, :]
        # recon amp at reference plane
        recon_amp = (single_field.abs() ** 2).mean(dim=0, keepdims=False).sqrt()
        save_torch_pt(os.path.join(opt.out_path, f'{batch_idx}_recon_reference_{d}.pt'), recon_amp)
        imsave_tensor(os.path.join(opt.out_path, f'{batch_idx}_recon_reference_{d}.png'), recon_amp)
        
        # save leftmost perspective view
        perspective, pupil_mask = prop_models.view_from_pupil(single_field, pupil_pos=opt.pupil_pos, pupil_rad=opt.pupil_rad, aperture=opt.aperture)
        perspective = (perspective.abs() ** 2).mean(dim=0, keepdims=True).sqrt()
        pupil_mask = resize_tensor(pupil_mask.unsqueeze(0), [min(*pupil_mask.shape[-2:])]*2) # resize to square
        pupil_path = os.path.join(opt.out_path, f'pos_{opt.pupil_pos[0]:.2f}_{opt.pupil_pos[1]:.2f}_rad_{opt.pupil_rad:.2f}')
        utils.cond_mkdir(pupil_path)
        save_torch_pt(os.path.join(pupil_path, f'{batch_idx}_recon_perspective_{d}.pt'), perspective)
        imsave_tensor(os.path.join(pupil_path, f'{batch_idx}_recon_perspective_{d}.png'), perspective)
        imsave_tensor(os.path.join(pupil_path, f'{batch_idx}_pupil_mask.png'), pupil_mask)

    # save slm
    encoded_slm = quantization.encode_binary_slm(final_slm)
    for N in range(encoded_slm.shape[0]):
        imsave_tensor(os.path.join(opt.out_path, f'{batch_idx}_slm_{N}.bmp'), encoded_slm[N, ...])

    # save eyebox
    ref_field = final_field[:,D//2:D//2+1,:,:] # extract reference plane
    eyebox = prop_models.compute_eyebox(ref_field)
    eyebox = (eyebox - eyebox.min()) / (torch.flatten(eyebox).sort()[0][-1000] - eyebox.min()) # normalize, ignore peak
    eyebox = torch.clip(eyebox, 0, 1)
    imsave_tensor(os.path.join(opt.out_path, f'{batch_idx}_eyebox.png'), eyebox)            



def field_init(init_phase, slm_mode, **opt):
    if slm_mode.lower() in ('amp'):
        init_field = init_phase + 0.5  # optimize for phase patterns
    elif slm_mode.lower() in ('complex'):
        init_field = torch.exp(1j * init_phase)  # optimize for complex fields
    elif slm_mode.lower() in ('phase'):
        init_field = init_phase  # optimize for phase patterns

    return init_field


def phase_init(batch_idx, is_recon, **opt):
    # load optimized slm from path
    if is_recon:
        # number of 8bit images to read
        n_slm = 1 if opt['num_frames'] < 8 else opt['num_frames'] // 8
        slms = []
        for N in range(n_slm):
            slm = imageio.imread(os.path.join(opt['slm_path'], f'{batch_idx}_slm_{N}.bmp'))
            slm = torch.from_numpy(utils.im2float(slm))
            slm = torch.reshape(slm, [1,1,*slm.shape]) # shape of (1,1,H,W)
            slms.append(slm)
        slms = torch.cat(slms, dim=0) # shape of (N,1,H,W)
        # decode 8bit to binary
        init_phase = quantization.decode_binary_slm(slms, opt['num_frames']) - 0.5
    else:
        init_phase = (opt['init_phase_range'] * (-0.5 + 1.0 * torch.rand(opt['num_frames'], 1, *opt['slm_res'])))
    
    return init_phase

    
def main():
    # Command line argument processing / Parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
    params.add_parameters(p, 'gd')
    opt = p.parse_args()
    opt = params.set_configs(opt)
    dev = torch.device('cuda')

    # tensorboard
    summary_name = f'{opt.qt_method}_{opt.bit_depth}'
    summaries_dir = os.path.join(opt.out_path, f'summaries/{summary_name}')
    utils.cond_mkdir(summaries_dir)
    writer = SummaryWriter(summaries_dir)

    forward_prop = prop_models.prop_model(**opt)  # image formation model
    camera_prop = prop_physical.prop_camera(**opt).to(dev)  if opt.citl else None
    algorithm = algorithms.load_alg(opt.cgh_method, opt.mem_eff)  # algorithm to optimize the slm patterns
    target_loader = image_loader.TargetLoader(**opt)  # target loader
    qt = quantization.qt_model(**opt)  # quantization

    for i, target in enumerate(target_loader):
        if i > 0: break
        if isinstance(target, tuple):
            target_amp, target_mask = target
            target_amp = target_amp.to(dev)
            target_mask = target_mask.to(dev)
        else:
            target_amp = target.to(dev)
            target_mask = None

        # initial field
        init_phase = phase_init(i, **opt).to(dev)
        init_field = field_init(init_phase, opt.slm_mode)

        # gradient-descent based optimizer
        results = algorithm(init_field, target_amp, target_mask, 
                            forward_prop=forward_prop,
                            camera_prop=camera_prop, 
                            qt=qt, tb_writer=writer, **opt)

        # TODO: Lint below into a few lines (i.e. def visualize(final_field))
        visualize_and_save(i, results, target_mask, opt)

if __name__ == "__main__":
    main()
