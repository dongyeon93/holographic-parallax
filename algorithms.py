"""
Various algorithms for LF/RGBD/RGB supervision.

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

import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import utils
from holo2lf import holo2lf
import os
from quantization import Quantizations, binarize

def load_alg(alg_type, mem_eff=False):
    if 'sgd' in alg_type.lower():
        if mem_eff:
            algorithm = eff_gradient_descent
        else:
            algorithm = gradient_descent
    elif 'gd_qt_test':
        algorithm = gradient_descent_qt_test
    return algorithm


def loss_func(slm_amp, target_amp, loss_fn, forward_prop=None, camera_prop=None, qt_func=None, 
              target_mask=None, nonzeros=None, dev=torch.device('cuda'), lf_supervision=False, cur_iter=0, **opt):
    """ Calculate the loss function from the slm amp. """
    
    # quantization
    if qt_func is not None:
        qt_field = qt_func(slm_amp, cur_iter=cur_iter, hard=opt['hard'])
    else:
        qt_field = slm_amp

    if not torch.is_complex(qt_field) and opt['slm_mode'].lower() not in ('amp'):
        field = torch.exp(1j * qt_field)
    elif opt['slm_mode'].lower() in ('amp'):
        field = qt_field * torch.exp(1j * torch.zeros_like(qt_field))  # amp mode

    if forward_prop is not None:
        field = forward_prop(field)
    field = utils.crop_image(field, opt['roi_res'], stacked_complex=False)

    if lf_supervision:
        recon_amp_t = holo2lf(field, n_fft=opt['n_fft'], hop_length=opt['hop_len'],
                              win_length=opt['win_len'], device=dev, impl='torch',
                              win_type=opt['win_func'], beta=opt['beta'],).sqrt()
    else:
        recon_amp_t = field.abs()

    if opt['time_joint']:  # time-multiplexed forward model
        recon_amp = (recon_amp_t**2).mean(dim=0, keepdims=True).sqrt()
    else:
        recon_amp = recon_amp_t

    if opt['citl']:  # surrogate gradients for CITL
        captured_amp = camera_prop(qt_field)
        captured_amp = utils.crop_image(captured_amp, opt['roi_res'],
                                        stacked_complex=False)
        recon_amp = recon_amp + captured_amp - recon_amp.detach()

    if target_mask is not None:
        final_amp = torch.zeros_like(recon_amp)
        final_amp[nonzeros] += (recon_amp[nonzeros] * target_mask[nonzeros])
    else:
        final_amp = recon_amp

    with torch.no_grad():
        s = (final_amp * target_amp).mean() / \
            (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

    return loss_fn(s * final_amp, target_amp), s * final_amp, field


def gradient_descent(init_field, target_amp, target_mask=None, forward_prop=None, qt=None, tb_writer=None,
                     camera_prop=None, loss_fn=nn.MSELoss(), **opt):

    # assert forward_prop is not None
    dev = init_field.device
    loss_fn = loss_fn.to(dev)

    slm_field = init_field.requires_grad_(True)  # phase at the slm plane
    optvars = [{'params': slm_field}]
    optimizer = optim.Adam(optvars, lr=opt['lr_optim'])

    loss_vals = []
    lf_supervision = len(target_amp.shape) > 4
    best_loss = 1e9
    best_iter = 0
    best_field = slm_field
    best_amp = slm_field.abs()
    best_phase = slm_field.angle()

    if target_mask is not None:
        target_amp = target_amp * target_mask
        nonzeros = target_mask > 0
    if opt['roi_res'] is not None:
        target_amp = utils.crop_image(target_amp, opt['roi_res'], stacked_complex=False, lf=lf_supervision)
        if target_mask is not None:
            target_mask = utils.crop_image(target_mask, opt['roi_res'], stacked_complex=False, lf=lf_supervision)
            nonzeros = target_mask > 0

    for t in tqdm(range(opt['num_iters'])):
        optimizer.zero_grad()

        # quantization
        if qt is not None:
            qt_field = qt(slm_field, cur_iter=t)
        else:
            qt_field = slm_field

        if not torch.is_complex(qt_field) and opt['slm_mode'].lower() not in ('amp'):
            field = torch.exp(1j * qt_field)
        elif opt['slm_mode'].lower() in ('amp'):
            field = qt_field * torch.exp(1j * torch.zeros_like(qt_field))  # amp mode

        if forward_prop is not None:
            field = forward_prop(field)

        field = utils.crop_image(field, opt['roi_res'], stacked_complex=False)

        if lf_supervision:
            recon_amp_t = holo2lf(field, n_fft=opt['n_fft'], hop_length=opt['hop_len'],
                                  win_length=opt['win_len'], device=dev, impl='torch',
                                  win_type=opt['win_func'], beta=opt['beta'],).sqrt()
        else:
            recon_amp_t = field.abs()

        if opt['time_joint']:  # time-multiplexed forward model
            recon_amp = (recon_amp_t**2).mean(dim=0, keepdims=True).sqrt()
        else:
            recon_amp = recon_amp_t

        if opt['citl']:  # surrogate gradients for CITL
            captured_amp = camera_prop(binarize(slm_field))
            captured_amp = utils.crop_image(captured_amp, opt['roi_res'],
                                            stacked_complex=False)
            recon_amp = recon_amp + captured_amp - recon_amp.detach()

        if target_mask is not None:
            final_amp = torch.zeros_like(recon_amp)
            final_amp[nonzeros] += (recon_amp[nonzeros] * target_mask[nonzeros])
        else:
            final_amp = recon_amp

        with torch.no_grad():
            s = (final_amp * target_amp).mean() / \
                (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

        loss_val = loss_fn(s * final_amp, target_amp)
        loss_val.backward()
        optimizer.step()

        with torch.no_grad():
            if t % 100 == 0:
                if tb_writer is not None:
                    tb_writer.add_scalar('loss', loss_val, t)
            if loss_val < best_loss:
                best_phase = binarize(slm_field)
                best_loss = loss_val
                best_amp = s * recon_amp
                best_field = s * field # best recon_field
                best_iter = t + 1

    # report best result
    print(f'-- Best loss: {best_loss.item()}')

    return {'loss_vals': loss_vals,
            'best_iter': best_iter,
            'best_loss': best_loss,
            'recon_amp': best_amp,
            'recon_field': best_field,
            'final_phase': best_phase}


def eff_gradient_descent(init_field, target_amp, target_mask=None, forward_prop=None, qt=None, tb_writer=None,
                         camera_prop=None, loss_fn=nn.MSELoss(), **opt):

    # assert forward_prop is not None
    dev = init_field.device
    loss_fn = loss_fn.to(dev)

    slm_field = init_field.requires_grad_(True)  # phase at the slm plane
    optvars = [{'params': slm_field}]
    optimizer = optim.Adam(optvars, lr=opt['lr_optim'])
    num_frames = init_field.shape[0]

    loss_vals = []
    lf_supervision = len(target_amp.shape) > 4
    best_loss = 1e9
    assert opt['time_joint'] == True

    if target_mask is not None:
        target_amp = target_amp * target_mask
        nonzeros = target_mask > 0
    if opt['roi_res'] is not None:
        target_amp = utils.crop_image(target_amp, opt['roi_res'], stacked_complex=False, lf=lf_supervision)
        if target_mask is not None:
            target_mask = utils.crop_image(target_mask, opt['roi_res'], stacked_complex=False, lf=lf_supervision)
            nonzeros = target_mask > 0

    for t in tqdm(range(opt['num_iters'])):
        optimizer.zero_grad()

        # amplitude reconstruction without graph
        with torch.no_grad():
            
            # quantization
            if qt is not None:
                qt_field = qt(slm_field, cur_iter=t)
            else:
                qt_field = slm_field

            if not torch.is_complex(qt_field) and opt['slm_mode'].lower() not in ('amp'):
                field = torch.exp(1j * qt_field)
            elif opt['slm_mode'].lower() in ('amp'):
                field = qt_field * torch.exp(1j * torch.zeros_like(qt_field))  # amp mode

            if forward_prop is not None:
                field = forward_prop(field)
            field = utils.crop_image(field, opt['roi_res'], stacked_complex=False)

            if lf_supervision:
                recon_field = holo2lf(field, n_fft=opt['n_fft'], hop_length=opt['hop_len'],
                                    win_length=opt['win_len'], device=dev, impl='torch',
                                    win_type=opt['win_func'], beta=opt['beta'],).sqrt()
            else:
                recon_field = field.abs()

            recon_amp_t = recon_field.abs()
            del qt_field, recon_field
            torch.cuda.empty_cache()
        
        # surrogate gradients for CITL
        if opt['citl']:  
            captured_amp = camera_prop(binarize(slm_field))
            captured_amp = utils.crop_image(captured_amp, opt['roi_res'],
                                            stacked_complex=False)

        frame_slice = 1
        
        for f in range(num_frames // frame_slice):            
            if qt is not None:
                qt_field = qt(slm_field[f*frame_slice:(f+1)*frame_slice, ...], cur_iter=t)
            else:
                qt_field = slm_field[f*frame_slice:(f+1)*frame_slice, ...]

            if not torch.is_complex(qt_field) and opt['slm_mode'].lower() not in ('amp'):
                field_sf = torch.exp(1j * qt_field)
            elif opt['slm_mode'].lower() in ('amp'):
                field_sf = qt_field * torch.exp(1j * torch.zeros_like(qt_field))  # amp mode

            if forward_prop is not None:
                field_sf = forward_prop(field_sf)
            field_sf = utils.crop_image(field_sf, opt['roi_res'], stacked_complex=False)

            if lf_supervision:
                recon_amp_sf = holo2lf(field_sf, n_fft=opt['n_fft'], hop_length=opt['hop_len'],
                                    win_length=opt['win_len'], device=dev, impl='torch',
                                    win_type=opt['win_func'], beta=opt['beta'],).sqrt()
            else:
                recon_amp_sf = field_sf.abs()

            ### insert graph from single frame ###
            recon_amp_t_with_grad = recon_amp_t.clone().detach()
            recon_amp_t_with_grad[f*frame_slice:(f+1)*frame_slice, ...] = recon_amp_sf

            if opt['time_joint']:  # time-multiplexed forward model
                recon_amp = (recon_amp_t_with_grad**2).mean(dim=0, keepdims=True).sqrt()
            else:
                recon_amp = recon_amp_t_with_grad

            if opt['citl']:  # surrogate gradients for CITL
                # Capture full recon before loop
                # captured_amp = camera_prop(binarize(slm_field))
                # captured_amp = utils.crop_image(captured_amp, opt['roi_res'],
                #                                 stacked_complex=False)
                recon_amp = recon_amp + captured_amp - recon_amp.detach()

            if target_mask is not None:
                final_amp = torch.zeros_like(recon_amp)
                final_amp[nonzeros] += (recon_amp[nonzeros] * target_mask[nonzeros])
            else:
                final_amp = recon_amp

            with torch.no_grad():
                s = (final_amp * target_amp).mean() / \
                    (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

            loss_val = loss_fn(s * final_amp, target_amp)
            loss_val.backward(retain_graph=False)
            
        # update phase variables
        optimizer.step()

        with torch.no_grad():
            if t % 100 == 0:
                if tb_writer is not None:
                    tb_writer.add_scalar('loss', loss_val, t)
            if loss_val < best_loss:
                best_phase = slm_field if qt is None else binarize(slm_field)
                best_loss = loss_val
                best_amp = s * recon_amp
                best_field = s * field # best recon_field
                best_iter = t + 1

        del field, qt_field, field_sf, recon_amp_sf, recon_amp_t_with_grad, recon_amp, final_amp, recon_amp_t
        torch.cuda.empty_cache()

    # report best result
    print(f'-- Best loss: {best_loss.item()}')
    
    return {'loss_vals': loss_vals,
            'best_iter': best_iter,
            'best_loss': best_loss,
            'recon_amp': best_amp,
            'recon_field': best_field,
            'final_phase': best_phase}


def gradient_descent_qt_test(init_field, target_amp, target_mask=None, forward_prop=None, qt=None, tb_writer=None,
                          camera_prop=None, loss_fn=nn.MSELoss(), **opt):

    # assert forward_prop is not None
    dev = init_field.device
    loss_fn = loss_fn.to(dev)

    slm_field = init_field.requires_grad_(True)  # phase at the slm plane
    optvars = [{'params': slm_field}]
    optimizer = optim.Adam(optvars, lr=opt['lr_optim'])

    loss_vals = []
    lf_supervision = len(target_amp.shape) > 4
    best_loss = 1e9
    best_iter = 0
    best_field = slm_field
    best_amp = slm_field.abs()
    best_phase = slm_field.angle()

    nonzeros = None
    if target_mask is not None:
        target_amp = target_amp * target_mask
        nonzeros = target_mask > 0
    if opt['roi_res'] is not None:
        target_amp = utils.crop_image(target_amp, opt['roi_res'], stacked_complex=False, lf=lf_supervision)
        if target_mask is not None:
            target_mask = utils.crop_image(target_mask, opt['roi_res'], stacked_complex=False, lf=lf_supervision)
            nonzeros = target_mask > 0

    for t in tqdm(range(opt['num_iters'])):
        optimizer.zero_grad()
        loss_val, recon_amp, recon_field = loss_func(slm_field, 
                                                     target_amp, 
                                                     loss_fn, 
                                                     forward_prop=forward_prop, 
                                                     camera_prop=camera_prop, 
                                                     qt_func=qt, 
                                                     target_mask=target_mask, 
                                                     nonzeros=nonzeros, 
                                                     lf_supervision=lf_supervision,
                                                     dev=dev, 
                                                     cur_iter=t,
                                                     **opt)
        loss_val.backward()
        optimizer.step()

        with torch.no_grad():
            loss_val_q, recon_amp_q, recon_field_q = loss_func(slm_field, 
                                                               target_amp, 
                                                               loss_fn, 
                                                               forward_prop=forward_prop, 
                                                               camera_prop=camera_prop, 
                                                               qt_func=Quantizations(qt_method='lee_ste'), 
                                                               target_mask=target_mask, 
                                                               nonzeros=nonzeros, 
                                                               lf_supervision=lf_supervision,
                                                               dev=dev, 
                                                               cur_iter=t,
                                                               **opt)
            if t % 100 == 0:
                if tb_writer is not None:
                    tb_writer.add_scalar('loss', loss_val, t)
                    tb_writer.add_scalar('loss/real', loss_val_q, t)

            if loss_val_q < best_loss:
                if qt is not None:
                    best_phase = qt(slm_field, t)
                else:
                    best_phase = slm_field
                best_loss = loss_val_q
                best_amp = recon_amp
                best_field = recon_field # best recon_field
                best_field_q = recon_field_q # best recon_field
                best_iter = t + 1

    # report best result
    print(f'-- Best loss: {best_loss.item()}')

    return {'loss_vals': loss_vals,
            'best_iter': best_iter,
            'best_loss': best_loss,
            'recon_amp': best_amp,
            'recon_field': best_field,
            'recon_field_q': best_field_q,
            'final_phase': best_phase}