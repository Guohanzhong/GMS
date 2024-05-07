import os
import time
import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sympy
import random
import datasets
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sympy import *
import copy
import json
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import warnings
from absl import app, flags
import torch
#from torchmin import minimize
#from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
from tqdm import tqdm
import logging
from model import UNet
from score.both import get_inception_and_fid_score
from libs.iddpm import UNetModel,UNetModel4Pretrained,UNetModel4Pretrained2,UNetModel4Pretrained3
from libs.iddpm import UNetModel,UNetModel4Pretrained,UNetModel4Pretrained2,UNetModel4Pretrained_three
from dpmsolver import *
from adan import Adan
from shampoo import Shampoo
#import models
import logging
#from models import utils as mutils
#from models import ncsnv2
#from models import ncsnpp
#from models import ddpm as ddpm_model
#from models import layerspp
#from models import layers
#from models import normalization
from utils import restore_checkpoint
FLAGS = flags.FLAGS
# Mode for this code
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_enum('exp_name', 'CIFAR10', ['CIFAR10','IMAGENET','LSUN'], help='name of experiment')
# Models Config
flags.DEFINE_integer('in_channel', 3, help='input channel of UNet')
flags.DEFINE_integer('out_channel', 3, help='output channel of UNet')
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_integer('num_res_blocks', 3, help='# resblock in each level')
flags.DEFINE_integer('num_heads', 4, help='Multi-Heads for attention')
flags.DEFINE_integer('dims', 2, help='1,2,3 dims')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [32 // 16, 32 // 8], help='add attention to these levels')
flags.DEFINE_float('dropout', 0.3, help='dropout rate of resblock')
flags.DEFINE_bool('use_scale_shift_norm', True, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_integer('head_out_channels', 3, help='the final layer of High order noise network')
flags.DEFINE_enum('mode', 'simple', ['simple','complex'], help='the mode for honn modeling')

# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion training noising steps')
flags.DEFINE_enum('sample_type', 'ddpm', ['ddpm', 'analyticdpm', 'gmddpm','ddim','dpmsolver'], help='sample type for sampling')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 1e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 500001, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_integer('noise_order', 1, help="the order of noise used to training")
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_string('pretrained_dir', './logs/iDDPM_CIFAR10_EPS/models/ckpt_1_450000.pt', help='log directory')
flags.DEFINE_enum('model_type', 'noise', ['noise', 'nll'], help='variance type')
flags.DEFINE_integer('save_step', 50000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')

# Sampling
flags.DEFINE_string('logdir', './logs/iDDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
flags.DEFINE_integer('sample_steps', 1000, help='Sampling steps for generation stage')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_bool('time_shift', False, help='whether the noised data is from t=1')
flags.DEFINE_bool('rescale_time', True, help='adjust the maxmimum time to input the network is 1000')
flags.DEFINE_bool('nll_training', False, help='training the model to fit the noise.pow(a)')
flags.DEFINE_enum('noise_schedule', 'linear', ['linear','cosine'], help='the mode for honn modeling')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_integer('clip_pixel', 2, "Var Clip for final steps")
flags.DEFINE_bool('merge_model', False, help='using merge model for saving computing time')
#flags.DEFINE_float('weight',1/5, help="the weight for the first gaussian")
# Model Dir
flags.DEFINE_string('eps1_dir', './logs/iDDPM_CIFAR10_EPS/models/ckpt_1_300000.pt', help='eps1 model log directory')
flags.DEFINE_string('eps2_dir', './logs/iDDPM_CIFAR10_EPS2/models/ckpt_2_300000.pt', help='eps2 model log directory')
flags.DEFINE_string('eps3_dir', './logs/iDDPM_CIFAR10_complex_EPS3/models/ckpt_3_300000.pt', help='eps3 model log directory')

device = torch.device('cuda:0')

def _rescale_timesteps_ratio(N, flag):
    if flag:
        return 1000.0 / float(N)
    return 1

def statistics2str(statistics):
    #for k,v in statistics.items():
    #    print(v)
    return str({k: f'{v:.6g}' for k, v in statistics.items()})

def report_statistics(s, t, statistics):
    logging.info(f'[(s, r): ({s:.6g}, {t:.6g})] [{statistics2str(statistics)}]')

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)

def solve_gmm(mean,cov,ske,gt,timestep,report_dict):
    device= mean.device
    x0 = torch.unsqueeze((mean),dim=0)
    x1 = torch.unsqueeze((mean-1e-3),dim=0)
    #beta2 = torch.unsqueeze(((torch.ones(size=mean.size()).to(device))*(cov/gt.mean().item())),dim=0)
    beta = torch.unsqueeze((torch.ones(size=mean.size()).to(device)*0.998),dim=0)
    #x     = torch.cat([x0,x1,beta1,beta2],axis=0)
    #x0,x1,beta = solve_analytic(mean,cov,ske)
    x     = torch.cat([x0,x1,beta],axis=0)
    cov_g = gt
    def loss_f(tensor):
        #if solve_type =='pi':
        x0, x1, beta = tensor[0,...], tensor[1,...],tensor[2,...]
        #x0, x1, beta1 = tensor[0,...], tensor[1,...],tensor[2,...]
        beta = torch.clamp(beta, 0.1, 1.2)
        #beta2 = 1
        pi = 1/3
        E0 = (pi*x0 + (1-pi)*x1 - mean).pow(2)
        E1 = (pi*(x0**2+cov_g*beta)+(1-pi)*(x1**2+cov_g*beta) - (mean**2+cov)).pow(2)
        E2 = (pi*(x0**3+3*x0*cov_g*beta)+(1-pi)*(x1**3+3*x1*cov_g*beta) - ske).pow(2)
        return ((E0+E1+E2)).mean(),E2.max()
    warm_up    = 18
    iterations = 70
    lr = 0.06
    # CIFAR10:Linear
    #lr     = max(-0.16*((1000-timestep)**2/1000**2)+0.16,0.12)
    #min_lr = 0.10
    # Imagenet
    #lr     = max(-0.10*((5000-timestep)**2/4000**2)+0.12,0.10)
    #min_lr = 0.07

    # LSUN
    #lr     = max(-0.25*((1200-timestep)**2/1000**2)+0.25,0.20)
    #min_lr = 0.15

    # Fixed learning rate
    #lr = 0.04

    warm_up_with_cosine_lr = lambda iter: (iter) / warm_up if iter <= warm_up \
        else max(0.5 * ( math.cos((iter - warm_up) /(iterations - warm_up) * math.pi) + 1), 
        min_lr / lr)

    with TemporaryGrad():
        #torch.optim.Adam, torch.optim.RMSprop, torch.optim.Adagrad, torch.optim.AdamW, Shampoo
        optimizer_solve = Adan([x],lr=lr,betas=(0.9,0.92,0.92))
        #optimizer_solve = torch.optim.Adam([x],lr=lr,betas=(0.9,0.99))
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_solve, warm_up_with_cosine_lr)
        pred_0,max_pre_E_2 = loss_f(x)
        for step in range(iterations):
            x.requires_grad = True
            pred,max_E_2 = loss_f(x)
            optimizer_solve.zero_grad()
            pred.backward()
            optimizer_solve.step()
            #scheduler.step()
    report_dict['mean optimize'] = pred/pred_0
    report_dict['3-max optimize'] = max_E_2/max_pre_E_2
    return x[0,...], x[1,...],torch.clamp(x[2,...], 0.1, 1.2),report_dict

def implicit_ddim(a_t,a_s,sigma_t,sigma_s,x_t,model_fn,s,report_dict):
    device= x_t.device
    x     = x_t
    def loss_f(tensor):
        #if solve_type =='pi':
        x0 = tensor
        E0 = (x0-a_s/a_t*x_t+sigma_s*model_fn(x0,s)).pow(2)
        return ((E0)).mean(),E0.max()
    warm_up    = 18
    iterations = 100
    lr = 0.06
    min_lr = 0.01
    warm_up_with_cosine_lr = lambda iter: (iter) / warm_up if iter <= warm_up \
        else max(0.5 * ( math.cos((iter - warm_up) /(iterations - warm_up) * math.pi) + 1), 
        min_lr / lr)
    with TemporaryGrad():
        optimizer_solve = Adan([x],lr=lr,betas=(0.9,0.92,0.92))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_solve, warm_up_with_cosine_lr)
        pred_0,max_pre_E_2 = loss_f(x)
        for step in range(iterations):
            x.requires_grad = True
            pred,max_E_2 = loss_f(x)
            optimizer_solve.zero_grad()
            pred.backward()
            optimizer_solve.step()
            scheduler.step()
    report_dict['mean optimize'] = pred/pred_0
    report_dict['max optimize'] = max_E_2/max_pre_E_2
    return x,report_dict

def extract(v, t, x_shape,ratio=None):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    #if ratio:
    #    out = torch.ones(size=(200,1)).squeeze()
    #    for ele in range(ratio):
    #        out *= torch.gather(v, index=t-ele, dim=0).float()
    out = torch.gather(v, index=t, dim=0).float()
    #print
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, eps1_model,eps2_model,eps3_model, beta_1, beta_T, sample_T,total_T = 4000,img_size=32,
                 sample_type='ddpm',time_shift=True,noise_schedule='linear',rescale_time=True,model_type='noise',clip_pixel=2,merge=False):
        assert sample_type in ['ddpm', 'analyticdpm', 'gmddpm','dpmsolver','ddim']
        super().__init__()
        self.model      = eps1_model
        self.cov_model  = eps2_model
        self.eps3_model = eps3_model
        self.T = sample_T
        self.total_T = total_T
        self.model_type = model_type
        self.rescale_ratio = _rescale_timesteps_ratio(total_T, rescale_time)
        self.clip_pixel = clip_pixel
        self.merge = merge
        logging.info('the scale ratio for timesteps is {0}'.format(self.rescale_ratio))

        if self.total_T % self.T  ==0:
            self.ratio = int(self.total_T/self.T)
        else:
            self.ratio = int(self.total_T/self.T)+1
        self.ratio_raw = self.total_T/self.T
        #self.ratio = 1
        self.t_list = [max(int(self.total_T-1-self.ratio_raw*x),0) for x in range(self.T)]
        if self.t_list[-1] != 0:
            self.t_list.append(0)

        # test dynamic timestep
        #self.t_list0 = [999]
        #self.t_list = [int(max(x/self.T * (self.total_T-1) ,0)) for x in reversed(range(self.T+1))]
        #if self.t_list[-1] != 0:
        #    self.t_list.append(0)
        #self.t_list  = [999,862, 772, 630,523, 407,272,177,100,0]
        #self.t_list = [998, 896, 799,680,561,498,449,403,363,326,293,261,229,197,159,100,50,0]
        #self.t_list = [999,933,960,801,726,651,577, 513, 464, 417, 376, 338, 304, 272, 240, 208, 173, 100,50,0]
        #self.t_list = [999, 880, 801, 726, 651, 577, 513, 464, 417, 376, 338, 304, 272, 240, 208, 173,104, 51, 0]
        logging.info(len(self.t_list))
        logging.info(self.t_list)

        self.img_size  = img_size
        self.sample_type = sample_type
        self.time_shift  = time_shift
        self.noise_schedule = noise_schedule
        if noise_schedule=='linear':
            self.register_buffer(
                'betas', torch.linspace(beta_1, beta_T, self.total_T).double())
            alphas = 1. - self.betas
            alphas_bar = torch.cumprod(alphas, dim=0)
            # calculations for diffusion q(x_t | x_{t-1}) and others
        else:
            logging.info(noise_schedule)
            g = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = [0.]
            for i in range(self.total_T):
                t1 = i / self.total_T
                t2 = (i + 1) / self.total_T
                betas.append(min(1 - g(t2) / g(t1), 0.999))
            betas = torch.tensor(np.array(betas))
            self.register_buffer(
                'betas', betas[1:])
            alphas= 1-betas
            alphas_bar = torch.cumprod(alphas[1:], dim=0)
            alphas = alphas[1:]
        
        if self.sample_type == 'dpmsolver':
            self.ns = NoiseScheduleVP('discrete', betas=self.betas)
            steps = len(self.t_list)-1
            K = steps  
            #logging.info(K)
            if steps % 3 == 0:
                self.orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                self.orders = [3,] * (K - 1) + [1]
            else:
                self.orders = [3,] * (K - 1) + [1]
            #self.orders = self.orders * 3
            #logging.info(len(self.orders))
            log_alphas = 0.5 * torch.log(1 - self.betas).cumsum(dim=0)
            self.log_alpha_array = log_alphas.reshape((1, -1,))

        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.total_T]
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))
        self.register_buffer(
            'sqrt_recip_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

        #logging.info(alphas_bar[:4])
        #logging.info(alphas_bar[-10:])

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    # use eps to estimate one order moment
    def predict_xpre_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        a_t = extract(self.sqrt_alphas_bar, t, x_t.shape)
        if (t-self.ratio)[0]>=0:
            a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)/extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
        else:
            a_s  = extract(self.sqrt_alphas_bar, t-t, x_t.shape)
            a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)/extract(self.sqrt_alphas_bar, t-t, x_t.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-t, x_t.shape))
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
        mean_x0 = (x_t - sigma_t * eps)/a_t
        self.statistics['xt_mean'] = x_t.mean().item()
        self.statistics['eps_mean'] = eps.mean().item()
        self.statistics['unclip mean_x0_mean'] = mean_x0.mean().item()
        mean_x0 = mean_x0.clamp(-1.,1.)
        self.statistics['clip mean_x0_mean'] = mean_x0.mean().item()
        mean_xs = a_ts*sigma_s.pow(2)/(sigma_t.pow(2)) * x_t + a_s*beta_ts/(sigma_t.pow(2)) * mean_x0
        mean_xs = mean_xs.clamp(-1000.,1000.)
        self.statistics['clip mean_xs_max'] = mean_xs.max().item()
        return mean_xs,mean_x0

    # use eps and eps2 to estimate one order moment
    def predict_xpre_cov_from_eps(self, x_t, t, eps,eps2):
        a_t  = extract(self.sqrt_alphas_bar, t, x_t.shape)
        if (t-self.ratio)[0]>=0:
            a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)/extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
        else:
            # \alpha_{t|s}
            a_s  = extract(self.sqrt_alphas_bar, t-t, x_t.shape)
            a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)/extract(self.sqrt_alphas_bar, t-t, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2

        sigma2_small = (sigma_s**2*beta_ts)/(sigma_t**2)
        if self.model_type == 'noise':
            cov_x0_pred = sigma_t.pow(2)/a_t.pow(2) * (eps2-eps.pow(2)) 
        else:
            #cov_x0_pred = sigma_t.pow(2)/a_t.pow(2) * (eps2) 
            cov_x0_pred = sigma_t.pow(2)/a_t.pow(2) * (eps2-eps.pow(2)) 
        self.statistics['noise1 mean'] = eps.mean().item()
        self.statistics['noise2 mean'] = eps2.mean().item()
        self.statistics['cov_x0_coeffi'] = (sigma_t.pow(2)/a_t.pow(2)).mean().item()
        self.statistics['unclip cov_x0_mean'] = cov_x0_pred.mean().item()
        cov_x0_pred = cov_x0_pred.clamp(0., 1.)
        self.statistics['clip cov_x0_mean'] = cov_x0_pred.mean().item()
        offset = a_s.pow(2)*beta_ts.pow(2)/sigma_t.pow(4) * cov_x0_pred
        self.statistics['offset'] = offset.mean().item()
        self.statistics['offset_max'] = offset.max().item()
        self.statistics['sigma2_small'] = sigma2_small.mean().item()
        model_var  = sigma2_small + offset
        model_var  = model_var.clamp(0., 1.)
        return model_var
    
    # use eps and eps2 and eps3 to estimate one order moment
    def predict_xpre_3moment_from_eps(self, x_t, t, eps, eps2, eps3):
        sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
        a_t     = extract(self.sqrt_alphas_bar, t, x_t.shape)
        if (t-self.ratio)[0]>=0:
            # \alpha_{t|s}
            a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)/extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
        else:
            # \alpha_{t|s}
            a_s  = extract(self.sqrt_alphas_bar, t-t, x_t.shape)
            a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)/extract(self.sqrt_alphas_bar, t-t, x_t.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2

        mean_x0 = (x_t - sigma_t * eps)/a_t
        twom_x0 = 1/(a_t.pow(2))*(x_t.pow(2)+sigma_t.pow(2)*eps2-2*x_t*sigma_t*eps)
        mean_x0 = mean_x0.clamp(-1., 1.)
        twom_x0 = twom_x0.clamp(0., 1.)

        if self.model_type == 'noise':
            skew_x0 = 1/(a_t.pow(3))*(x_t.pow(3) - sigma_t.pow(3)*eps3 - 3*x_t.pow(2)*sigma_t*eps + 3*x_t*sigma_t.pow(2)*eps2)
        else:
            skew_x0 = 1/(a_t.pow(3))*(x_t.pow(3) + eps3)
        self.statistics['unclip_x0_skew'] = skew_x0.mean().item()
        skew_x0 = torch.where(torch.abs(skew_x0)<=torch.abs(mean_x0),skew_x0,mean_x0)
        skew_x0 = skew_x0.clamp(-1., 1.)
        self.statistics['clip_x0_skew'] = skew_x0.mean().item()
        sigma2_small = (sigma_s**2*beta_ts)/(sigma_t**2)

        skew_xs_part1 = (a_ts*sigma_s.pow(2)/(sigma_t.pow(2)) * x_t).pow(3)+\
            3*(a_ts*sigma_s.pow(2)/(sigma_t.pow(2)) * x_t).pow(2)*(a_s*beta_ts/sigma_t.pow(2))*mean_x0 +\
            3*(a_ts*sigma_s.pow(2)/(sigma_t.pow(2)) * x_t)*(a_s*beta_ts/sigma_t.pow(2)).pow(2)*twom_x0 +\
            (a_s*beta_ts/sigma_t.pow(2)).pow(3)*skew_x0
        skew_xs_part2 = 3*sigma2_small*(a_ts*sigma_s.pow(2)/(sigma_t.pow(2)) * x_t + a_s*beta_ts/(sigma_t.pow(2)) * mean_x0)
        skew_xs  = skew_xs_part1+skew_xs_part2
        self.statistics['clip_xs_skew'] = skew_xs.mean().item()
        return skew_xs

    def ddpm_cov(self, x_t, t):
        sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
        if (t-self.ratio)[0]>=0:
            # \alpha_{t|s}
            a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)/extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
        else:
            # \alpha_{t|s}
            a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)/extract(self.sqrt_alphas_bar, t-t, x_t.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2

        model_var1 = (sigma_s**2*beta_ts)/(sigma_t**2)
        self.statistics['sigma2_small'] = model_var1.mean().item()
        return model_var1

    #@torch.no_grad()
    def p_mean_variance(self, x_t, t):
        if self.merge:
            eps,eps2,eps3 = self.model(x_t, (t)*self.rescale_ratio)

        else:
            if self.time_shift:
                eps  = self.model(x_t, (t+1)*self.rescale_ratio)
                if self.sample_type=='gmddpm' or self.sample_type=='analyticdpm':
                    eps2 = self.cov_model(x_t, (t+1)*self.rescale_ratio)
                    if self.sample_type=='gmddpm':
                        eps3 = self.eps3_model(x_t, (t+1)*self.rescale_ratio)
            else:
                eps  = self.model(x_t, t*self.rescale_ratio)
                if self.sample_type=='gmddpm' or self.sample_type=='analyticdpm':
                    eps2 = self.cov_model(x_t, (t)*self.rescale_ratio)
                    if self.sample_type=='gmddpm':
                        eps3 = self.eps3_model(x_t, t*self.rescale_ratio)

        if self.sample_type == 'ddpm':   # the model predicts epsilon
            model_mean,mean_x0 = self.predict_xpre_from_eps(x_t, t, eps=eps)
            model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
            }['fixedsmall']
            if self.ratio == 1:
                model_log_var = extract(model_log_var, t, x_t.shape)
                return model_mean, torch.exp(model_log_var)
            else:
                model_log_var = self.ddpm_cov(x_t,t)
                return model_mean,model_log_var

        elif self.sample_type == 'analyticdpm':
            assert self.cov_model is not None
            model_mean,mean_x0 = self.predict_xpre_from_eps(x_t, t, eps)
            model_var  = self.predict_xpre_cov_from_eps(x_t, t, eps, eps2)
            return model_mean, model_var
        elif self.sample_type == 'gmddpm':
            assert self.eps3_model is not None
            mean,mean_x0    = self.predict_xpre_from_eps(x_t, t, eps)
            cov     = self.predict_xpre_cov_from_eps(x_t, t, eps, eps2)
            skeness = self.predict_xpre_3moment_from_eps(x_t, t, eps, eps2, eps3)
            #sigma2_small  = self.ddpm_cov(x_t,t)
            return mean,cov,skeness,cov
        elif self.sample_type == 'ddim':
            ## DDIM only need eps network
            a_t = extract(self.sqrt_alphas_bar, t, x_t.shape)
            a_s = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            x0_t = (x_t - eps*sigma_t)/(a_t)
            x0_t = x0_t.clamp(-1.,1.)
            #model_mean,mean_x0 = self.predict_xpre_from_eps(x_t, t, eps=eps)
            eta = 0
            c_1 = eta * torch.sqrt((1-a_t.pow(2)/a_s.pow(2)) * (1-a_s.pow(2))/(1-a_t.pow(2)))
            c_2 = torch.sqrt((1-a_s.pow(2))-c_1.pow(2))
            #sqrt_one_minus_a_s = 1/extract(self.sqrt_recip_one_minus_alphas_bar, t-self.ratio, x_t.shape)
            mean = a_s * x0_t + c_2*eps + c_1 * torch.randn_like(x_t)
            self.statistics['eta'] = eta
            return mean
        else:
            raise NotImplementedError(self.sample_type)

    def forward(self, x_T):
        x_t = x_T
        for n_count1,time_step in enumerate(self.t_list):
            if n_count1 < len(self.t_list)-1:self.ratio = int(self.t_list[n_count1] - self.t_list[n_count1+1])
            self.statistics = {}
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * time_step
            if time_step > 0:
                noise = torch.randn_like(x_t).to(x_T.device)
            else:
                if self.merge:
                    eps,eps2,eps3 = self.model(x_t, (t)*self.rescale_ratio)
                else:
                    if self.time_shift:
                        eps = self.model(x_t, (t+1)*self.rescale_ratio)
                    else:
                        eps = self.model(x_t, t*self.rescale_ratio)

                a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)
                sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                beta_ts = (1-a_ts**2)
                x_0 = 1/a_ts*( x_t - eps * beta_ts/sigma_t)
                return torch.clip(x_0, -1, 1)

            # sample with mixture of Gaussian
            if self.sample_type == 'gmddpm':
                mean,cov,tmoment,pre_cov = self.p_mean_variance(x_t=x_t, t=t)
                self.statistics['moment error'] =  (torch.abs(tmoment-mean.pow(3)-3*mean*cov)).mean().item()
                if time_step-self.ratio <= 0:
                    var_threshold = (self.clip_pixel * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                    self.statistics['unclip var_mean'] = var.mean().item()
                    var = cov
                    var = var.clamp(0., var_threshold)
                    self.statistics['clip var_mean'] = var.mean().item()
                    self.statistics['threshold for var'] = var_threshold
                    x_t = mean + var**0.5 * noise
                    report_statistics(torch.tensor(max(time_step-self.ratio,0)), torch.tensor(time_step), self.statistics)
                    continue
                mean1,mean2,beta,self.statistics = solve_gmm(mean,cov,tmoment,pre_cov,time_step,self.statistics)
                var  = pre_cov*beta
                mean = torch.zeros(size=mean1.size()).to(mean1.device)
                for n_count in range(mean.size()[0]):
                    if (torch.rand(size=(1,1))<torch.tensor(1/3))[0][0]:
                        mean[n_count,...]= mean1[n_count,...]
                    else:
                        mean[n_count,...]= mean2[n_count,...]
                self.statistics['Gaussian_cov'] = pre_cov.mean().item()
                self.statistics['choosend_cov'] = var.mean().item()
                self.statistics['choosend_cov_min'] = var.min().item()
                x_t = mean + var**0.5 * noise
            elif self.sample_type == 'ddim':
                #mean  = self.p_mean_variance(x_t=x_t, t=t)
                a_t = extract(self.sqrt_alphas_bar, t, x_t.shape)
                a_s = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
                sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
                s = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * (time_step-self.ratio)
                x_t,self.statistics = implicit_ddim(a_t,a_s,sigma_t,sigma_s,x_t,self.model,s,self.statistics)
                #x_t   = mean
            # sample with DPMSolver (Lu et al. (2022))
            elif self.sample_type == 'dpmsolver':
                order_now = self.orders[n_count1]
                a_t  = extract(self.sqrt_alphas_bar, t, x_t.shape)
                a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
                sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
                lambda_t = torch.log(a_t) - torch.log(sigma_t) 
                lambda_s = torch.log(a_s) - torch.log(sigma_s)
                h_i =  lambda_s - lambda_t

                if order_now == 1:
                    phi_1 = torch.expm1(h_i)
                    model_s = self.model(x_t, t*self.rescale_ratio)
                    x_t = (
                        a_s/a_t * x_t
                        - (sigma_s * phi_1) * model_s
                    )

                elif order_now == 2:
                    r_i = 1/2
                    lambda_s1 = (lambda_t+lambda_s)*r_i
                    s1 = int((self.ns.inverse_lambda(lambda_s1)*self.total_T)[0])
                    s1 = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * s1
                    a_si = extract(self.sqrt_alphas_bar, s1, x_t.shape)
                    sigma_si = torch.sqrt(extract(self.one_minus_alphas_bar, s1, x_t.shape))
                    model_s = self.model(x_t, t*self.rescale_ratio)

                    x_s1 = a_si/a_t * x_t - sigma_si*(torch.exp(h_i*r_i)-1)*model_s
                    model_s1 = self.model(x_s1, s1)
                    phi_1 = torch.expm1(h_i)
                    x_t = (
                        a_s/a_t * x_t
                        - (sigma_s * phi_1) * model_s
                        - (0.5 / r_i) * (sigma_s * phi_1) * (model_s1 - model_s)
                    )

                elif order_now == 3:
                    r1,r2 = 1/3,2/3
                    phi_11 = torch.expm1(r1 * h_i)
                    phi_12 = torch.expm1(r2 * h_i)
                    phi_1 = torch.expm1(h_i)
                    phi_22 = torch.expm1(r2 * h_i) / (r2 * h_i) - 1.
                    phi_2 = phi_1 / h_i - 1.
                    phi_3 = phi_2 / h_i - 0.5

                    lambda_s1 = lambda_t + r1 * h_i
                    lambda_s2 = lambda_t + r2 * h_i

                    s1 = int((self.ns.inverse_lambda(lambda_s1)*self.total_T)[0])
                    s2 = int((self.ns.inverse_lambda(lambda_s2)*self.total_T)[0])
                    s1 = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * s1
                    s2 = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * s2
                
                    a_s1 = extract(self.sqrt_alphas_bar, s1, x_t.shape)
                    sigma_s1 = torch.sqrt(extract(self.one_minus_alphas_bar, s1, x_t.shape))
                    a_s2 = extract(self.sqrt_alphas_bar, s2, x_t.shape)
                    sigma_s2 = torch.sqrt(extract(self.one_minus_alphas_bar, s2, x_t.shape))
                    model_s = self.model(x_t, t*self.rescale_ratio)
                    x_s1 = ( (a_t/a_s1) * x_t
                        - (sigma_s1 * phi_11) * model_s )

                    model_s1 = self.model(x_s1, s1)
                    x_s2 = (
                        (a_s2/a_t) * x_t
                        - (sigma_s2 * phi_12) * model_s
                        - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
                    )
                    model_s2 = self.model(x_s2, s2)
                    x_t = (a_s/a_t * x_t
                    - (sigma_s * phi_1) * model_s
                    - (1. / r2) * (sigma_s * phi_2) * (model_s2 - model_s))
                self.statistics['Adjustment for the third order'] = ((1. / r2) * (sigma_s * phi_2) * (model_s2 - model_s)).mean().item()
                self.statistics['order'] = order_now
            # sample with DDPM/Imperfect Analytic-DPM (Bao et al. (2022))
            else:
                mean, var = self.p_mean_variance(x_t=x_t, t=t)
                #logging.info('var={}'.format(var))
                if time_step-self.ratio <= 0:
                    var_threshold = (self.clip_pixel * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                    self.statistics['unclip var_mean'] = var.mean().item()
                    var = var.clamp(0., var_threshold)
                    self.statistics['clip var_mean'] = var.mean().item()
                    self.statistics['threshold for var'] = var_threshold
                x_t = mean + var**0.5 * noise
            ### Report #####
            report_statistics(torch.tensor(max(time_step-self.ratio,0)), torch.tensor(time_step), self.statistics)

device = torch.device('cuda:0')

def Sample_parallel(net_sampler):
    save_file = './sample/' + str(FLAGS.exp_name) +'/'+str(FLAGS.sample_type)+str(FLAGS.sample_steps)+'/'
    images    = []
    #with torch.no_grad():
    for i in trange(0, FLAGS.num_images, FLAGS.batch_size):
        batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
        x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
        batch_images_g= net_sampler(x_T.to(device))
        batch_images = batch_images_g.cpu()
        images.append((batch_images + 1) / 2)
        #for kkk in range(batch_images.size()[0]):
        #    single_image = (batch_images[kkk,...]+1)/2
        #    try:
        #        save_image(single_image, save_file+str(i+kkk)+'.png')
        #    except:
        #        os.mkdir(save_file)
        #        save_image(single_image, save_file+str(i+kkk)+'.png')
        grid = (make_grid(batch_images[:64,...]) + 1) / 2
        path = os.path.join(
            save_file,'sample.png')
        try:
            save_image(grid, path)
        except:
            os.mkdir(save_file)
            save_image(grid, path)
    images = torch.cat(images, dim=0).numpy()
    print(images.shape)
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    print(IS)
    print(FID)

def eval():
    if FLAGS.exp_name != 'LSUN':
        eps1_model = UNetModel4Pretrained2(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
        channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=FLAGS.use_scale_shift_norm,
        head_out_channels=FLAGS.head_out_channels,mode=FLAGS.mode)
    else:
        from libs.ddpm import Model4Pretrained2
        eps1_model = Model4Pretrained2(in_channels=FLAGS.in_channel,ch=FLAGS.ch,out_ch=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attn_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
        ch_mult=tuple(FLAGS.ch_mult),resamp_with_conv=True,resolution=FLAGS.img_size,mode=FLAGS.mode,head_out_ch=3)     
    
    try:
        ckpt1 = torch.load(FLAGS.eps1_dir)['ema_model']
    except:
        ckpt1 = torch.load(FLAGS.eps1_dir)
    try:
        #logging.info('sucess loading')
        eps1_model.load_state_dict(ckpt1)
    except:
        eps1_model = UNetModel(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=FLAGS.use_scale_shift_norm,)
        eps1_model.load_state_dict(ckpt1)
    eps1_model.eval()    
    eps2_model = None
    eps3_model = None

    # Sampling for Extended Analytic DPM
    if FLAGS.sample_type == 'analyticdpm' or FLAGS.sample_type == 'gmddpm':
        print('Sample IS not using DDPM')
        if FLAGS.exp_name != 'LSUN':
            eps2_model = UNetModel4Pretrained(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=FLAGS.use_scale_shift_norm,
            head_out_channels=FLAGS.head_out_channels,mode='complex')
        else:
            from libs.ddpm import Model4Pretrained
            eps2_model = Model4Pretrained(in_channels=FLAGS.in_channel,ch=FLAGS.ch,out_ch=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attn_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            ch_mult=tuple(FLAGS.ch_mult),resamp_with_conv=True,resolution=FLAGS.img_size,mode=FLAGS.mode,head_out_ch=3)   
        try:
            ckpt2 = torch.load(FLAGS.eps2_dir)['ema_model']
        except:
            ckpt2 = torch.load(FLAGS.eps2_dir)
        eps2_model.load_state_dict(ckpt2)
        eps2_model.eval()

        if FLAGS.sample_type == 'gmddpm':
            if FLAGS.exp_name != 'LSUN':
                eps3_model = UNetModel4Pretrained(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
                    channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=FLAGS.use_scale_shift_norm,
                    head_out_channels=FLAGS.head_out_channels,mode='complex')
            else:
                eps3_model = Model4Pretrained(in_channels=FLAGS.in_channel,ch=FLAGS.ch,out_ch=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attn_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
                ch_mult=tuple(FLAGS.ch_mult),resamp_with_conv=True,resolution=FLAGS.img_size,mode=FLAGS.mode,head_out_ch=3)   
            try:
                ckpt3 = torch.load(FLAGS.eps3_dir)['ema_model']
            except:
                ckpt3 = torch.load(FLAGS.eps3_dir)
            eps3_model.load_state_dict(ckpt3)
            logging.info(FLAGS.eps3_dir)
            eps3_model.eval()

    ##### merge the model ##### ##### ##### #####
    if FLAGS.merge_model:
        if FLAGS.exp_name != 'LSUN':
            for name in eps3_model.state_dict():
                if name.split('.')[0] == 'out2':
                    name2 = name[4:]
                    name_changes = 'out3' + name2
                    net_dict[name_changes] = eps3_model.state_dict()[name]
                else:
                    net_dict[name] = eps3_model.state_dict()[name]
        else:
            for name in eps3_model.state_dict():
                if len(name.split('_'))>1:
                    if name.split('_')[1].split('.')[0] == 'out2':
                        name1 = name.split('_')[0] + '_'
                        name2 = name.split('_')[1][4:]
                        name_changes =  name1 + 'out3'+name2 
                        net_dict[name_changes] = eps3_model.state_dict()[name]
                    elif name.split('_')[1].split('.')[0] == 'out' and name.split('_')[0].split('.')[-1] != 'proj':
                        name1 = name.split('_')[0] + '_'
                        if len(name.split('_') )==2:
                            name2 = name.split('_')[1][3:]
                            name_changes =  name1 + 'out3'+name2 
                        else:
                            name2 = name.split('_')[1][3:]
                            name3 = name.split('_')[2]
                            name_changes =  name1 + 'out3'+name2 +'_'+name3
                        net_dict[name_changes] = eps3_model.state_dict()[name]
                    else:
                        net_dict[name] = eps3_model.state_dict()[name]
                else:
                    net_dict[name] = eps3_model.state_dict()[name]
        if FLAGS.exp_name != 'LSUN':
            model_final  = UNetModel4Pretrained_three(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
                channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=FLAGS.use_scale_shift_norm,
                head_out_channels=FLAGS.head_out_channels,mode=FLAGS.mode)
        else:
            from libs.ddpm import Model4Pretrained_three
            model_final = Model4Pretrained_three(in_channels=FLAGS.in_channel,ch=FLAGS.ch,out_ch=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attn_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            ch_mult=tuple(FLAGS.ch_mult),resamp_with_conv=True,resolution=FLAGS.img_size,mode=FLAGS.mode,head_out_ch=FLAGS.head_out_channels)   

        model_dict = model_final.state_dict()
        if FLAGS.exp_name != 'LSUN':
            for k,v in net_dict.items():
                if k.split('.')[0] != 'out2':
                    model_dict[k] = v
        else:
            for k,v in net_dict.items():
                if len(k.split('_')) >1:
                    if k.split('_')[1].split('.')[0] != 'out' or k.split('_')[1].split('.')[0] != 'out2':
                        model_dict[k] = v    
                else:
                    model_dict[k] = v    
        model_final.load_state_dict(model_dict)
        pretrained_dict = {k:v for k,v in ckpt2.items() if (k in model_dict)}
        model_final_dict = model_final.state_dict()
        model_final_dict.update(pretrained_dict)
        model_final.load_state_dict(model_final_dict)
        model_final.eval()
    ##### merge the model ##### ##### ##### ########## merge the model ##### ##### ##### #####
    if FLAGS.merge_model: eps1_model = model_final

    net_sampler = GaussianDiffusionSampler(
        eps1_model,eps2_model,eps3_model,
        FLAGS.beta_1, FLAGS.beta_T, FLAGS.sample_steps, FLAGS.T, FLAGS.img_size,
        FLAGS.sample_type,FLAGS.time_shift,FLAGS.noise_schedule,FLAGS.rescale_time,FLAGS.model_type,FLAGS.clip_pixel,merge=FLAGS.merge_model).to(device)

    if FLAGS.parallel:
        net_sampler = torch.nn.DataParallel(net_sampler)
    with torch.no_grad():
        Sample_parallel(net_sampler)

def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    eval()

app.run(main)