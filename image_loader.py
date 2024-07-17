"""
Image Loaders

"""
import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset

##
import math
import skimage.io
from imageio import imread
from skimage.transform import resize
from torchvision.transforms.functional import resize as resize_tensor
import cv2
import utils
import torch
import prop_models
import Incoherent_focal_stack
import torch.nn.functional as F

def resize_keep_aspect(image, target_res, pad=False, lf=False, pytorch=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False, lf=lf)

    # switch to numpy channel dim convention, resize, switch back
    if lf or pytorch:
        image = resize_tensor(image, target_res)
        return image
    else:
        image = np.transpose(image, axes=(1, 2, 0))
        image = resize(image, target_res, mode='reflect')
        return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res, pytorch=False):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=pytorch, stacked_complex=False),
                            target_res, pytorch=pytorch, stacked_complex=False)


def get_listdir(data_path):
    listdir = [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]  # get folder names

    return listdir


def view2idx(view, num_views):
    return (view[0] - 1) * num_views[0] + (view[1] - 1)


def idx2view(idx, num_views):
    return ((idx // num_views[0]) + 1, (idx % num_views[1]) + 1)    


def get_image_filenames(dir, focuses=None):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif', 'exr', 'dpt', 'hdf5')
    if isinstance(dir, str):
        # dir is folder
        if os.path.isdir(dir):
            files = os.listdir(dir)
            exts = (os.path.splitext(f)[1] for f in files)
            if focuses is not None:
                images = [os.path.join(dir, f)
                        for e, f in zip(exts, files)
                        if e[1:] in image_types and int(os.path.splitext(f)[0].split('_')[-1]) in focuses]
            else:
                images = [os.path.join(dir, f)
                        for e, f in zip(exts, files)
                        if e[1:] in image_types]
        # dir is file
        elif os.path.isfile(dir):
            assert os.path.splitext(dir)[1][1:] in image_types
            images = [dir]
        # no match
        else:
            raise ValueError(f'{dir} is neither file nor directory')            
        return images
    
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            images_in_folder = [os.path.join(folder, f)
                                for e, f in zip(exts, files)
                                if e[1:] in image_types]
            images = [*images, *images_in_folder]

        return images


def get_lf_foldernames(data_path, ch):
    """
    Returns list of LF scene folders in the input directory    
    If color-dependent lightfields are vailable, they should be placed as:
        data_path
        |-scene1
            |- ch_0
                |- LF_0_0.png
                |- LF_0_1.png
                ...
            |- ch_1
            |- ch_2
        ...

        :param data_path: list of directories [scene1, scene2, ...] or str 'data_path' for multiple scenes
        :return folder_path: list of folders of scene names [scene1, scene2, ...]. If color-dependence is available, output would be [scene1/ch_1, scene2/ch_1, ...]
    """
    if isinstance(data_path, str):
        folders = get_listdir(data_path)        

        # data_path = single scene name
        if (len(folders) == 3 and all(fname[0:2] == 'ch' for fname in folders)) or (len(folders) == 0):
            folders = [data_path]
        # multiple scenes under data_path    
        else:
            folders = [os.path.join(data_path, folder) for folder in folders]
    
    # list of scene names
    elif isinstance(data_path, list):
        folders = data_path

    # color-dependent lightfields
    folder_path = []
    for folder in folders:
        subfolders = get_listdir(folder)

        if len(subfolders) == 3 and all(fname[0:2] == 'ch' for fname in subfolders):
            print('load color-dependent lf')
            folder = os.path.join(folder, f'ch_{ch}')
        
        folder_path.append(folder)

    return folder_path


def get_fs_foldernames(data_path):
    if isinstance(data_path, str):
        folders = get_listdir(data_path)

        # multiple scenes
        if len(folders) > 0:
            folder_path = [os.path.join(data_path, folder) for folder in folders]
        # single scene (data_path = scene folder)
        else:
            folder_path = [data_path]

    elif isinstance(data_path, list):
        folder_path = data_path
        
    return folder_path

def get_rgbd_filenames(data_path):
    """
    Get rgbd filenames inside data_path
    We expect rgbd files as 'path/scene_rgb.png, path/scene_depthmap.png'
    If data_path is
        list: Each str should be 'path/scene' (_rgb and _depthmap omitted)
        str: If dir, load all rgbd in data_path. Else it should be 'path/scene' (_rgb and _depthmap omitted)

    """
    # read all images in folder data_path
    if os.path.isdir(data_path):
        fnames = get_image_filenames(data_path)
        fnames.sort()
        rgb_path = fnames[1::2]
        depth_path = fnames[0::2]
    # data_path is list of scenes
    else:
        # str input is length 1 list
        if not isinstance(data_path, list):
            data_path = [data_path]
        rgb_path = [f'{d}_rgb.png' for d in data_path]
        depth_path = [f'{d}_depthmap.png' for d in data_path]
    return rgb_path, depth_path


def imread_srgb(path, crop_to_roi, gamma2lin=True, **opt):
    im = imread(path)

    if len(im.shape) < 3:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  # augment channels for gray images

    if opt['channel'] is None:
        im = im[..., :3]  # remove alpha channel, if any
    else:
        # select channel while keeping dims
        im = im[..., opt['channel'], np.newaxis]

    im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

    # linearize intensity and convert to amplitude
    if gamma2lin:
        im = utils.srgb_gamma2lin(im)
        im = np.sqrt(im)  # to amplitude

    # move channel dim to torch convention
    im = np.transpose(im, axes=(2, 0, 1))

    # normalize resolution
    if crop_to_roi:
        im = pad_crop_to_res(im, opt['roi_res'])
    else:
        im = resize_keep_aspect(im, opt['roi_res'])
    im = pad_crop_to_res(im, opt['image_res'])

    # path = os.path.splitext(self.target_names[filenum])[0]

    return torch.from_numpy(im).float().unsqueeze(0)

def gen_target_mask(target_amp, target_type, slm_mode, **opt):
    # generate LF target mask considering sideband filtering
    if target_type == '4d' and slm_mode.lower() in ('amp'): # 
        # angular resolution
        my, mx = target_amp.shape[-2:] 
        # sideband filter, but not shifted
        mag = 10
        tmp_filter = prop_models.compute_filter((my*mag, mx*mag), shape='rect', aperture=[opt['aperture'] / 2, opt['aperture']], sideband=False)
        tmp_filter = tmp_filter.to(target_amp.device)

        target_mask = torch.zeros_like(target_amp)
        for i in range(my):
            for j in range(mx):
                # integrate filter area
                target_mask[..., i,j] = torch.sum(tmp_filter[mag*i:mag*(i+1), mag*j:mag*(j+1)]) / (mag**2)

    # RGBD layer mask
    # NOTE: in this case, target_amp is actually target_depthmap
    elif target_type == '2.5d':
        target_mask = Incoherent_focal_stack.gen_depthmap_mask(target_amp, opt['division_dists'])

    # TODO: other target mask methods
    else:
        target_mask = None    
        
    return target_mask


class TargetLoader(Dataset):
    def __init__(self, data_path, target_type, crop_to_roi=False, flipud=False, shuffle=False, target_save=False, target_load = False,**opt):
        self.data_path = data_path
        if target_type == '2d':
            self.target_names = get_image_filenames(data_path)            
            # center view of lf dataset
            if 'lf' in data_path or 'olas' in data_path:
                self.target_names = [p for p in self.target_names if '13_13' in p] # 13_13 is an index of center view

        if target_type == '2.5d':
            self.target_names, self.depthmap_names = get_rgbd_filenames(data_path)

        if target_type == '3d':
            self.target_names = get_fs_foldernames(data_path)

        if target_type == '4d':
            self.target_names = get_lf_foldernames(data_path, opt['channel'])
            
        # select only few data
        if opt['data_size'] is not None:
            self.target_names = self.target_names[0:opt['data_size']]
    
        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.target_names)))
        self.order = list(self.order)
        self.target_type = target_type
        self.crop_to_roi = crop_to_roi  # if False, resize
        self.flipud = flipud
        self.opt = opt
        self.shuffle = shuffle
        self.target_save = target_save
        self.target_load = target_load

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.order)

        return self

    def __len__(self):
        return len(self.order)

    def __next__(self):
        if self.idx < len(self.order):
            idx = self.order[self.idx]
            self.idx += 1
            return self.__getitem__(*idx)
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if self.target_type == '2d':
            return self.load_image(idx)
        if self.target_type == '2.5d':
            target_amp = self.load_image(idx)
            depthmap = self.load_depthmap(idx)
            target_mask = gen_target_mask(depthmap, self.target_type, **self.opt)
            return target_amp, target_mask.unsqueeze(0)
        if self.target_type == '3d':
            return self.load_fs(idx)
        if self.target_type == '4d':
            target_amp = self.load_lf(idx) 
            target_mask = gen_target_mask(target_amp, self.target_type, **self.opt)
            return target_amp, target_mask

    def load_image(self, filenum, *augmentation_states):
        target_name = self.target_names[filenum]
        print(f'-- target: {target_name}')

        im = imread_srgb(target_name, self.crop_to_roi, **self.opt)
        return im
    
    def load_depthmap(self, filenum):
        fname = self.depthmap_names[filenum]
        print(f'-- target depthmap: {fname}')

        depthmap = imread(fname)[:,:,0] # grayscale
        depthmap = utils.im2float(depthmap, dtype=np.float64)  # convert to double, max 1

        if self.opt['close_is_0'] is False:
            depthmap = 1.0 - depthmap

        # normalize resolution
        if self.crop_to_roi:
            depthmap = pad_crop_to_res(depthmap, self.opt['roi_res'])
        else:
            depthmap = resize_keep_aspect(depthmap, self.opt['roi_res'])
        depthmap = pad_crop_to_res(depthmap, self.opt['image_res'])
        
        return torch.from_numpy(depthmap).float()
    
    def load_fs(self, idx):
        """
        Load focal stack under fs folder
            :return: shape of (1,D,H,W)
        """
        folder_path = self.target_names[idx]
        print(f'-- target: {folder_path}')
        fnames = get_image_filenames(folder_path)
        fnames.sort()

        fs = []
        for fname in fnames:
            im = imread_srgb(fname, self.crop_to_roi, **self.opt)
            fs.append(im)
        fs = torch.cat(fs, dim=1)

        return fs
        

    def load_lf(self, idx):        
        """
        Assume filename 'Camera_00_01_rgbd.png' for OLAS dataset and 'LF_0_1.png' for others
        (25,25) is recommend for total_num_lf_views (supports 1x1, 3x3, 5x5, 7x7, 9x9, 13x13 ang_res with strides)
            y: 0 -> total_num_lf_views[0], camera position bottom -> top. (OLAS dataset is in reverse order top -> bottom, so we reverse them here)
            x: 0 -> total_num_lf_views[1], camera position left -> right
        """
        folder_path = self.target_names[idx]
        print(f'-- target: {folder_path}')

        total_num_lf_views = self.opt['total_num_lf_views']
        ang_res = self.opt['ang_res']

        target_pt_filename = os.path.join(folder_path,'Target_LF.pt')

        # load LF file saved as pt before
        if os.path.exists(target_pt_filename):
            lf = torch.load(target_pt_filename)
            print("Target light field loaded from pt.")
        
        # load full LF from image and save as pt
        else:
            print("Target light field loaded individually.")
            lfs = []
            for v_y in range(total_num_lf_views[0]):
                for v_x in range(total_num_lf_views[1]):                
                    # file indexing
                    my, mx = v_y, v_x

                    # invert LF order
                    if self.opt['invert_lf'] is True:
                        my, mx = total_num_lf_views[0] - 1 - my, total_num_lf_views[1] - 1 - mx

                    # filename
                    if 'olas' in folder_path:
                        my = total_num_lf_views[0] - 1 - my # invert y axis order for olas dataset
                        view_path = os.path.join(folder_path, f'Camera_{my:02d}_{mx:02d}_rgbd.png')
                    else:
                        view_path = os.path.join(folder_path, f'LF_{my}_{mx}.png')

                    im = imread_srgb(view_path, self.crop_to_roi, **self.opt)
                    lf_view = torch.tensor(im, dtype=torch.float32).reshape(1, *im.shape[-2:])
                    lfs.append(lf_view)

            lf = torch.stack(lfs, -1)
            lf = lf.reshape(*lf.shape[:3], *total_num_lf_views)

            if self.flipud:
                lf = lf.flip(dims=[-2])

            lf = lf.unsqueeze(0) # shape of (N,1,H,W,U,V)

            # save loaded LF 
            torch.save(lf, os.path.join(folder_path, 'Target_LF.pt'))            
            print("Target light field saved as pt.")

        # compute strides
        assert (total_num_lf_views[0] - 1) % (ang_res[0] - 1) == 0 and (total_num_lf_views[1] - 1) % (ang_res[1] - 1) == 0, f'Stride issue: total lf {total_num_lf_views} / ang res {ang_res}'
        strides = [(total_num_lf_views[i] - 1) // (ang_res[i] - 1) for i in range(2)]

        # LF with selected angular res
        lf_selected = torch.zeros(*lf.shape[0:-2], *ang_res).to(lf.device)

        for v_y in range(ang_res[0]):
            for v_x in range(ang_res[1]):                
                my, mx = v_y * strides[0], v_x * strides[1]
                lf_selected[..., v_y, v_x] = lf[..., my, mx]

        return lf_selected
        
