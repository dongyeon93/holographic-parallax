"""
Quantization model
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def binarize(tensor):
    # naive binarization of input tensor
    return (torch.sign(tensor - 0.5) + 1) / 2

def qt_model(qt_method, **opt):
    if qt_method is None or qt_method.lower() == 'none':
        return None
    else:
        return Quantizations(qt_method.lower(), **opt)

def encode_binary_slm(input_slm):
    """
    Encode multiple binary SLMs into 8bit 1 channel image
    encoded slm is normalized to [0,1]
        :param input_slm: shape of (N,1,H,W)
        :return encoded_slm: shape of (N/8,1,H,W)
    """
    N, _, H, W = input_slm.shape
    assert N % 8 == 0 or N == 1 or (N < 8 and 8 % N == 0), f'Number of SLMs {N} not divide or divided by 8'

    if N==1:
        print('- single SLM, not encoded')
        encoded_slm = input_slm
    else:
        if N < 8 and 8 % N == 0:
            print(f'- repeat {N} SLMs {8//N} times in a single 8bit image')
            input_slm = input_slm.repeat(8//N, 1, 1, 1)
        else:
            print(f'- {N} SLMs, encoded to {N//8} images')
        # reshape slm
        slm = input_slm.view(8, -1, 1, H, W)

        # encoding vector
        encode_vec = torch.arange(0,8).to(slm.device).view(8,1,1,1,1)
        encode_vec = (2 ** encode_vec) / 255

        # encode slm
        encoded_slm = encode_vec * slm
        encoded_slm = torch.sum(encoded_slm, 0, keepdim=False)

    return encoded_slm

def decode_binary_slm(input_slm, num_frames):
    """
    Decode encoded 8bit binary SLMs into multiple binary SLMs
        :param input_slm: shape of (N,1,H,W), 8bit images normalized to [0,1]
        :param frames: int, number of time-multiplexed frames
        :return decoded_slm: shape of (frames,1,H,W), binary SLMs
    """
    if num_frames == 1:
        assert input_slm.shape[0] == 1
        decoded_slm = input_slm
    else:
        input_slm = (255 * input_slm).to(torch.uint8)
        num_slm = input_slm.shape[0]
        decoded_slm = torch.zeros(8*num_slm, *input_slm.shape[1:]).to(input_slm.device)

        for N in range(num_slm):
            # select single 8bit image
            slm = input_slm[N:N+1, ...]
            for i in range(8):
                # decode each bit
                decoded_slm[8*N+i, ...] = slm // (2 ** (7-i))
                slm = slm % (2 ** (7-i))

        decoded_slm = decoded_slm[0:num_frames, ...].to(torch.float32)

    return decoded_slm


def score_phase(phase, lut, s=1., func='sigmoid'):
    # Here s is kinda representing the steepness

    # wrapped_phase = (phase + math.pi) % (2 * math.pi) - math.pi

    diff = phase - lut
    # diff = (diff + math.pi) % (2*math.pi) - math.pi  # signed angular difference
    # diff /= math.pi  # normalize

    if func == 'sigmoid':
        z = s * diff
        scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
    elif func == 'log':
        scores = -torch.log(diff.abs() + 1e-20) * s
    elif func == 'poly':
        scores = (1 - torch.abs(diff)**s)
    elif func == 'sine':
        scores = torch.cos(math.pi * (s * diff).clamp(-1., 1.))
    elif func == 'chirp':
        scores = 1 - torch.cos(math.pi * (1-diff.abs())**s)

    return scores

    
class QtzClip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_field, *args, **kwargs):
        # Straight-through estimator of Lee et al. "High-contrast, speckle-free, true 3D holography via binary CGH optimization."
        # Binary 0 or 1 quantization
        out_field = (torch.sign(input_field - 0.5) + 1) / 2

        # Non masking out the gradient (just pass through)
        maskout_grad = torch.ones_like(out_field)
        # save out for backward
        ctx.save_for_backward(maskout_grad)

        return out_field
    
    def backward(ctx, grad_output):
        # alternatively mask out where gradients should be zeros
        maskout_grad, = ctx.saved_tensors
        return grad_output * maskout_grad, None
    

class QtzHardTanh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_field, *args, **kwargs):        
        out_field = F.hardtanh(input_field, 0.0, 1.0)

        # Option 1: Non masking out the gradient (just pass through)
        maskout_grad = torch.ones_like(out_field)

        # Option 2: Masking out where it is clipped - should be worse
        # cond1 = torch.where(input_field <= 1.0, 1.0, 0.0)
        # cond2 = torch.where(input_field >= 0.0, 1.0, 0.0)
        # maskout_grad = cond1 * cond2

        # save out for backward
        ctx.save_for_backward(maskout_grad)

        return out_field

    def backward(ctx, grad_output):
        # alternatively mask out where gradients should be zeros
        maskout_grad, = ctx.saved_tensors
        return grad_output * maskout_grad, None


# Basic function for NN-based quantization, customize it with various surrogate gradients!
class QtzNearestNeighborSearch(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, f, bit_depth, *args, **kwargs):
        f_raw = f.detach()  # assume the input is already in [0, 1]
        f_q = torch.clip(f_raw.abs(), min=0.0, max=1.0)
        f_q = (f_q * (2**bit_depth - 1)).round().float() / (2**bit_depth - 1)
        ctx.mark_non_differentiable(f)

        cond1 = torch.where(f <= 1.0, 1.0, 0.0)
        cond2 = torch.where(f >= 0.0, 1.0, 0.0)
        maskout_grad = (cond1 * cond2).float()

        # save out for backward
        ctx.save_for_backward(maskout_grad)
        return f_q

    def backward(ctx, grad_output):
        maskout_grad, = ctx.saved_tensors
        return grad_output * maskout_grad, None


# Softmax-based quantization, supporting Gumbel noises.
class SoftmaxBasedQuantization(nn.Module):
    def __init__(self, lut, gumbel=True, tau_max=0.1, c=3000., p=0.5, **kwargs):
        super(SoftmaxBasedQuantization, self).__init__()

        if not torch.is_tensor(lut):
            self.lut = torch.tensor(lut, dtype=torch.float32)
        else:
            self.lut = lut
        self.lut = self.lut.reshape(1, len(lut), 1, 1)
        self.c = c  # boost the score
        self.p = p  # power
        self.gumbel = gumbel
        self.tau_max = tau_max

    def forward(self, phase, tau=1.0, s=0.1, hard=False, *args, **kwargs):
        # phase_wrapped = (phase + math.pi) % (2*math.pi) - math.pi

        # phase to score
        scores = score_phase(phase, self.lut.to(phase.device), s) * self.c * (self.tau_max / tau) ** self.p

        # score to one-hot encoding
        if self.gumbel:  # (N, 1, H, W) -> (N, C, H, W)
            one_hot = F.gumbel_softmax(scores, tau=tau, hard=hard, dim=1)
        else:
            y_soft = F.softmax(scores/tau, dim=1)
            index = y_soft.max(1, keepdim=True)[1]
            one_hot_hard = torch.zeros_like(scores,
                                            memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            if hard:
                one_hot = one_hot_hard + y_soft - y_soft.detach()
            else:
                one_hot = y_soft

        # one-hot encoding to phase value
        q_phase = (one_hot * self.lut.to(one_hot.device))
        q_phase = q_phase.sum(1, keepdims=True)
        return q_phase


# Quantization wrapper
class Quantizations(nn.Module):
    def __init__(self, qt_method, **opt):
        super(Quantizations, self).__init__()
        self.qt_method = qt_method
        self.opt = opt

        if qt_method.lower() in ('htanh'):
            self.q = QtzHardTanh.apply
        elif qt_method.lower() in ('lee_ste'):
            self.q = QtzClip.apply
        elif qt_method.lower() in ('nn', 'nns'):
            self.q = QtzNearestNeighborSearch.apply
        elif self.qt_method.lower() in ('gumbel', 'sbq'):
            self.q = SoftmaxBasedQuantization([0, 1], **opt)
        else:
            raise ValueError(f'Unsupported quantization method: {qt_method}')

    def forward(self, input_amp, cur_iter, *args, **kwargs):
        if self.qt_method.lower() in ('htanh', 'lee_ste', 'nn', 'nns'):
            return self.q(input_amp)
        elif self.qt_method.lower() in ('gumbel', 'sbq'):
            t = (cur_iter + 1e-12) / (self.opt['num_iters'] - 1 + 1e-12)
            tau = max(self.opt['tau_min'], self.opt['tau_max'] * math.exp(-self.opt['r'] * t))  # tau decay
            s = self.opt['s_min'] + (self.opt['s_max'] - self.opt['s_min']) * t  # narrowing the width for GS
            return self.q(input_amp, s=s, tau=tau, **kwargs)
