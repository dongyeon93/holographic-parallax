"""
Default parameter settings for SLMs as well as laser/sensors

"""
import datetime
import math
import os
import sys
import numpy as np
import utils
from Incoherent_focal_stack import compute_depthmap_dists

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

def str2tuple(v):
    """ Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, tuple):
        return v
    elif ',' in v:
        return tuple(map(int, v.split(",")))
    elif str(int(v)) == v:
        return (int(v),)
    else:
        raise ValueError('Tuple value expected.')

def str2float_tuple(v):
    """ Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, tuple):
        return v
    elif ',' in v:
        return tuple(map(float, v.split(",")))
    elif str(float(v)) == v:
        return (float(v),)
    else:
        raise ValueError('Tuple value expected.')

def str2bool(v):
    """ Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


class PMap(dict):
    # use it for parameters
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def add_parameters(p, mode='train'):
    if not isinstance(mode, str):
        if isinstance(mode, list) or isinstance(mode, tuple):
            mode = ', '.join(mode)
        else:
            raise TypeError(f'  - mode is not a list ')
    
    p.add_argument('--channel', type=int, default=1, help='set color, by default green')
    p.add_argument('--exp', type=str, default='', help='Name of experiments')
    p.add_argument('--slm_type', type=str, default='flcos', help='SLM model')
    p.add_argument('--sensor_type', type=str, default='flir', help='Sensor model')
    p.add_argument('--ls_type', type=str, default='wiki', help='Laser model')
    p.add_argument('--setup_type', type=str, default='thinvr', help='setup dependent paramters .. (prop dist, ... etc)')
    # p.add_argument('--lf_type', type=str, default='olas', help='type of target light field ...')  # NOTE: Not really used ...
    p.add_argument('--roi_res', type=str2tuple, default=(900, 1600), help='Region of interest')
    p.add_argument('--image_res', type=str2tuple, default=(1200, 1200), help='Region of interest')
    p.add_argument('--num_frames', type=int, default=8, help='Number of frames')
    p.add_argument('--num_recon_frames', type=int, default=30, help='Number of pupil recon frames')
    p.add_argument('--physical_iris', type=str2bool, default=True, help='If true, consider wavelength-dependent iris size')
    p.add_argument('--data_size', type=int, default=None, help='Number of target data. If None, use all dataset')
    p.add_argument('--bit_depth', type=int, default=8, help='Number of bits for quantization')
    p.add_argument('--slm_mode', type=str, default='phase', help='amp or phase')
    p.add_argument('--sideband_margin', type=int, default=10, help='Number of pixels for margin used in prop_sideband')
    p.add_argument('--crop_to_roi', type=str2bool, default=True, help='If True, crop target rather than resize, else resize rather than crop')
    p.add_argument('--eyepiece', type=float, default=40*mm, help='Focal length of eyepiece in meter')
    p.add_argument('--prop_dists', nargs='+', default=None, help='List of propagation distances for ASM in meter. 160mm-eyepiece for None')
    p.add_argument('--aperture', type=float, default=1.0, help='Relative aperture size. If 1.0, use full frequency bandwidth')
    p.add_argument('--aperture_shape', type=str, default='rect', help = 'aperture_shape for GT (rect, circ)')

    # lf
    p.add_argument('--ang_res', type=str2tuple, default=(7, 7), help='amp or phase')
    p.add_argument('--win_func', type=str, default='hann', help='amp or phase')
    p.add_argument('--beta', type=float, default=12., help='amp or phase')

    # recon
    p.add_argument('--is_perspective', type=str2bool, default=False, help='If True, generate focal stack with perspective')
    p.add_argument('--is_recon', type=str2bool, default=False, help='If True, reconstruct focal stack')
    p.add_argument('--slm_path', type=str, default='./slm', help='Directory of optimized slm used for reconstruction')
    p.add_argument('--pupil_pos', type=str2float_tuple, default=(0, -0.5), help='pupil center position normalized with eyebox size')
    p.add_argument('--pupil_rad', type=float, default=0.5, help='pupil radius size normalized with eyebox size')
    p.add_argument('--ref_path', type =str, default='./data/lf_fs_dataset', help = 'light field focal stack reference directory')
    p.add_argument('--pupil_path', type=str, default='./data/pupil_trajectory', help = 'pupil trajectory (x,y,pd,t) directory')
    
    # pupil-aware
    p.add_argument('--stoch_pupil', type=str2bool, default=False, help='if True, run pupil-aware holography algorithms')
    p.add_argument('--min_pupil_size', type=float, default=0.2, help='minimum random pupil size')
    p.add_argument('--max_pupil_size', type=float, default=0.4, help='maximum random pupil')
    p.add_argument('--pupil_range', type=str2float_tuple, default=(0.1, 0.3), help='radius of the region that pupil is sampled in Fourier plane')


    # hardware models
    p.add_argument('--monitor_num', type=int, default=1, help='SLM display monitor number')
    p.add_argument('--show_preview', type=str2bool, default=False, help='If True, show preview during calibration')
    p.add_argument('--calib_phase_path', type=str, default='./calibration', help='Calibration SLM path')

    if 'gd' in mode.lower():
        p.add_argument('--lr_optim', type=float, default=1e-2, help='learning rate for phase optimization')
        p.add_argument('--target_type', type=str, default='2d', help='2d, 2.5d, 3d, 4d')
        p.add_argument('--num_iters', type=int, default=500, help='Number of iterations')
        p.add_argument('--time_joint', type=str2bool, default=True, help='If True, optimize for time-averaged intensity')
        p.add_argument('--citl', type=str2bool, default=False, help='If True, use camera captured image for loss function \
                                                                     while using surrogate gradient from the forward model')
        p.add_argument('--cgh_method', type=str, default='SGD', help='CGH algorithm, e.g. GS/SGD/DPAC/HOLONET/UNET, ...')
        p.add_argument('--data_path', type=str, default='./data/unity_lf_dataset', help='Directory for data') # 4d
        p.add_argument('--out_path', type=str, default='../results', help='Directory for output')
        p.add_argument('--prop_model', type=str, default='sideband', help='Simulated propagation model used')
        p.add_argument('--init_phase_range', type=float, default=1.0, help='Range for phase initialization')
        p.add_argument('--reg_lf_var', type=float, default=0.0, help='Coefficients for STFT regularization')
        p.add_argument('--mem_eff', type=str2bool, default=False, help='If True, run memory-efficient version')
        p.add_argument('--qt_method', type=str, default='htanh', help='SLM quantization method')

        # lightfield parameters
        p.add_argument('--total_num_lf_views', nargs=2, type=int, default=(25,25), help='Total number of LF views inside directory. We select a few views from total views')
        p.add_argument('--invert_lf', type=str2bool, default=False, help='If True, invert LF data order')

        # focal stack parameters
        p.add_argument('--num_planes', type=int, default=9, help='Number of focal stack planes')
        p.add_argument('--close_is_0', type=str2bool, default=False, help='If True, closer objects are darker in depthmap')
        p.add_argument('--far2near_dist', type=float, default=6, help='Far-near clipping distance used in unity rgbd rendering')
    
    # Gumbel-softmax related parameters
    p.add_argument('--tau_max', type=float, default=2.0, help='maximum value of tau')
    p.add_argument('--tau_min', type=float, default=1.2, help='minimum value of tau')
    p.add_argument('--r', type=float, default=0.5, help='decay ratio')
    p.add_argument('--c_s', type=float, default=3000, help='a number to boost the score (higher more robust to gumbel noise)')
    p.add_argument('--s_max', type=float, default=0.16, help='width of score max')
    p.add_argument('--s_min', type=float, default=0.08, help='width of score min')
    p.add_argument('--p', type=float, default=0.5, help='arith')
    p.add_argument('--hard', type=str2bool, default=False, help='If true, use quantize the forward pass and use STE')
    

def set_configs(opt):
    opt.chan_str = ('red', 'green', 'blue')[opt.channel]

    # set hardware parameters ()
    add_params_slm(opt.slm_type, opt)
    add_params_sensor(opt.sensor_type, opt)
    add_params_light_src(opt.ls_type, opt)
    add_params_setup(opt.setup_type, opt)
    add_params_hw(opt)

    # set software parameters (ROI size, ...)
    add_params_lf(opt)
    add_params_forward_prop(opt.prop_model, opt)

    # data path (list to str)
    if isinstance(opt.data_path, list):
        opt.data_path = opt.data_path[0] if len(opt.data_path)==1 else opt.data_path

    # out path
    utils.cond_mkdir(opt.out_path)

    return PMap(vars(opt))


def add_params_slm(slm_type, opt):
    """ Setting for specific SLM. """
    if slm_type.lower() in ('leto'):
        opt.feature_size = (8.0 * um, 8.0 * um)  # SLM pitch
        opt.slm_res = (1080, 1920)  # resolution of SLM
    elif slm_type.lower() in ('pluto'):
        opt.feature_size = (8.0 * um, 8.0 * um)
        opt.slm_res = (1080, 1920)
    elif slm_type.lower() in ('ti'):
        opt.feature_size = (10.8 * um, 10.8 * um)
        opt.slm_res = (800, 1280)
    elif slm_type.lower() in ('lf_sim'):
        opt.feature_size = (8.0 * um, 8.0 * um)
        opt.slm_res = (1200, 1200)
    elif slm_type.lower() in ('flcos'):
        opt.feature_size = (8.2 * um, 8.2 * um)
        opt.slm_res = (1200, 1920)
        opt.slm_mode = 'amp'
    opt.image_res = opt.slm_res

    opt.size_slm = tuple([pp * rr for pp, rr in zip(opt.feature_size, opt.slm_res)])
    return opt


def add_params_sensor(sensor_type, opt):

    return opt


def add_params_light_src(ls_type, opt):
    """ Setting for the specific light source. """
    if 'samba' in ls_type.lower():
        opt.wavelengths = (532 * nm, 532 * nm, 532 * nm)
    elif 'fisba' in ls_type.lower():
        opt.wavelengths = (638 * nm, 520 * nm, 450 * nm)
    elif 'wiki' in ls_type.lower():
        opt.wavelengths = (638 * nm, 520 * nm, 450 * nm)
    else:
        raise ValueError('  - doublecheck light source type ...')

    # select channel
    opt.wavelength = opt.wavelengths[opt.channel]

    # wavelength-dependent aperture size
    if opt.physical_iris:
        opt.aperture *= opt.wavelengths[-1] / opt.wavelength

    return opt


def add_params_setup(setup_type, opt):
    if opt.prop_dists is None:
        opt.prop_dists = [160*mm - opt.eyepiece] # Compensate change of eyepiece
    else:
        opt.prop_dists = [float(d) for d in opt.prop_dists]

    return opt


def add_params_lf(opt):
    if 'olas' in opt.data_path:
        opt.total_num_lf_views = (25, 25)

    # lf params
    # NOTE: now we get 'ang_res' and 'total_num_lf_views' from add_parameters
    opt.n_fft = opt.ang_res
    opt.hop_len = (1,1)
    opt.win_len = opt.n_fft

    return opt


def add_params_forward_prop(model_type, opt):
    ''' define multi-plane propagation dists '''
    # distance conversion used in unity rendering
    ortho_size = 2
    opt.depthmap_dists_range = opt.roi_res[0] * opt.feature_size[0] * opt.far2near_dist / ortho_size / 2
    print(f'- depthmap range: {opt.depthmap_dists_range * 1e2 :.2f} mm')
    mp_dists, opt.division_dists = compute_depthmap_dists(num_planes=opt.num_planes,
                                                        depthmap_dists_range=opt.depthmap_dists_range,
                                                        eyepiece=opt.eyepiece,
                                                        prop_dists=opt.prop_dists)
    
    # used for propagation in 3D and 2.5D supervision
    if opt.target_type in ('3d', '2.5d'):
        opt.prop_dists = mp_dists

    # used for reconstruction in 4D supervision
    elif opt.target_type in ('4d'):
        opt.recon_dists = mp_dists

    return opt


def add_params_hw(opt):
    ''' parameters for SLM & camera and their calibration '''
    # CITL optimization
    if opt.citl:
        # camera
        params_cam = PMap()
        params_cam.gamma = 1.0
        params_cam.exposure = [50.0, 50.0, 100.0][opt.channel] # ms
        params_cam.gain = [0.0, 0.0, 0.0][opt.channel]
        params_cam.flipud = True
        params_cam.fliplr = False
        opt.params_cam = params_cam

        # slm
        opt.monitor_num = 1

        # homography
        params_calib = PMap()
        params_calib.show_preview = opt.show_preview
        params_calib.num_circles = (13, 21)
        params_calib.spacing_size = [int(roi / (num_circs - 1))
                                    for roi, num_circs in zip(opt.roi_res, params_calib.num_circles)]
        params_calib.pad_pixels = [int(slm - roi) // 2 for slm, roi in zip(opt.slm_res, opt.roi_res)]
        params_calib.quadratic = False
        params_calib.phase_path = opt.calib_phase_path
        opt.params_calib = params_calib


    # TODO: other hardware settings

    else:
        pass