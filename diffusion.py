import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _rescale_timesteps_ratio(N, flag):
    if flag:
        return 1000.0 / float(N)
    return 1

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,noise_order=1,noise_schedule='linear',time_shift=False,rescale_time=True,nll_training=False):
        super().__init__()
        """
        T: total sample steps (training and sampling)
        """
        self.model = model
        self.T = T
        self.noise_order = int(noise_order)
        self.time_shift  = time_shift
        self.rescale_ratio = _rescale_timesteps_ratio(T, rescale_time)
        logging.info('the scale ratio for timesteps is {0}'.format(self.rescale_ratio))
        self.nll_training = nll_training

        """
        linear schedule and cosine schedule
        """
        if noise_schedule=='linear':
            self.register_buffer(
                'betas', torch.linspace(beta_1, beta_T, T).double())
            alphas = 1. - self.betas
            alphas_bar = torch.cumprod(alphas, dim=0)
            # calculations for diffusion q(x_t | x_{t-1}) and others
        else:
            logging.info(noise_schedule)
            g = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = [0.]
            for i in range(self.T):
                t1 = i / self.T
                t2 = (i + 1) / self.T
                betas.append(min(1 - g(t2) / g(t1), 0.999))
            betas = torch.tensor(np.array(betas))
            self.register_buffer(
                'betas', betas[1:])
            alphas= 1-betas
            alphas_bar = torch.cumprod(alphas[1:], dim=0)
            alphas = alphas[1:]
            #logging.info(alphas_bar)
            logging.info(alphas_bar.size())
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))

    def forward(self, x_0,mean_predict=False,nll_training=False):
        """
        Algorithm for training using noise network or nll network
        """

        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        # When the model start with t=-1, time_shift = False
        if self.time_shift:
            output_model = self.model(x_t, (t+1)*self.rescale_ratio)
        else:
            output_model = self.model(x_t, t*self.rescale_ratio)


        if self.noise_order==1:
            loss = F.mse_loss(output_model, noise, reduction='none')
        else:
            loss = F.mse_loss(output_model, noise.pow(self.noise_order), reduction='none')
            if self.nll_training:
                sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                error_three = - sigma_t.pow(3)*noise.pow(3) - 3*x_t.pow(2)*sigma_t*noise + 3*x_t*sigma_t.pow(2)*noise.pow(2)
                loss = F.mse_loss(output_model, error_three, reduction='none')
        """
        else:
            t = torch.randint(self.T-1, size=(x_0.shape[0], ), device=x_0.device)
            noise = torch.randn_like(x_0)
            x_tminus = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
            noise = torch.randn_like(x_0)
            a_ts = extract(self.sqrt_alphas_bar, t+1, x_tminus.shape)/extract(self.sqrt_alphas_bar, t, x_tminus.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_tminus.shape))
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t+1, x_tminus.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
            x_t = (
                a_ts * x_tminus + beta_ts**0.5 * noise)
            loss = F.mse_loss(self.model(x_t, t), x_tminus, reduction='none')
        """
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', var_type='fixedlarge',noise_schedule='linear',time_shift=False,rescale_time=True):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.time_shift = time_shift
        self.rescale_ratio = _rescale_timesteps_ratio(T, rescale_time)
        logging.info('the scale ratio for timesteps is {0}'.format(self.rescale_ratio))

        if noise_schedule=='linear':
            self.register_buffer(
                'betas', torch.linspace(beta_1, beta_T, T).double())
            alphas = 1. - self.betas
            alphas_bar = torch.cumprod(alphas, dim=0)
            # calculations for diffusion q(x_t | x_{t-1}) and others
        else:
            logging.info(noise_schedule)
            g = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = [0.]
            for i in range(self.T):
                t1 = i / self.T
                t2 = (i + 1) / self.T
                betas.append(min(1 - g(t2) / g(t1), 0.999))
            betas = torch.tensor(np.array(betas))
            self.register_buffer(
                'betas', betas[1:])
            alphas= 1-betas
            alphas_bar = torch.cumprod(alphas[1:], dim=0)
            alphas = alphas[1:]
            #logging.info(alphas_bar)
            logging.info(alphas_bar.size())
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))
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

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            if self.time_shift:
                eps = self.model(x_t, (t+1)*self.rescale_ratio)
            else:
                eps = self.model(x_t, t*self.rescale_ratio)
            eps = self.model(x_t, t*self.rescale_ratio)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            x_0 = torch.clip(x_0, -1., 1.)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            if time_step == 0:
                if self.time_shift:
                    eps = self.model(x_t, (t+1)*self.rescale_ratio)
                else:
                    eps = self.model(x_t, t*self.rescale_ratio)
                a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)
                sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                beta_ts = (1-a_ts**2)
                x_0 = 1/a_ts*( x_t - eps * beta_ts/sigma_t)
                #x_0 = x_t
                return torch.clip(x_0, -1, 1)
            #print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

class GaussianDiffusionSamplergm(nn.Module):
    def __init__(self, eps1_model, beta_1, beta_T, T,img_size=32,
                 sample_type='eps',eps2_model=None,eps3_model=None,eps4_model=None):
        assert sample_type in ['ddpm', 'analyticdpm', 'gmddpm']
        super().__init__()
        self.model      = eps1_model
        self.cov_model  = eps2_model
        self.eps3_model = eps3_model
        self.eps4_model = eps4_model
        self.T = T
        self.total_T = 1000
        if self.total_T % self.T  ==0:
            self.ratio = int(self.total_T/self.T)
        else:
            self.ratio = int(self.total_T/self.T)+1
        self.t_list = [max(self.total_T-1-self.ratio*x,1) for x in range(T)]
        print(self.t_list)
        self.img_size  = img_size
        self.sample_type = sample_type
        #self.device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, self.total_T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.total_T]
        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))

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
        if (t-self.ratio)[0]>=0:
            a_ts = extract(self.sqrt_recip_alphas_bar, t-self.ratio, x_t.shape)/extract(self.sqrt_recip_alphas_bar, t, x_t.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
        else:
            a_ts = extract(self.sqrt_recip_alphas_bar, t-t, x_t.shape)/extract(self.sqrt_recip_alphas_bar, t, x_t.shape)
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-t, x_t.shape))
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2

        return 1/a_ts*( x_t - eps * beta_ts/sigma_t)

    # use eps and eps2 to estimate one order moment
    def predict_xpre_cov_from_eps(self, x_t, t, eps):
        eps2 = self.cov_model(x_t, t)
        if (t-self.ratio)[0]>=0:
            beta_ts = extract(self.one_minus_alphas_bar, t, x_t.shape)-(extract(self.sqrt_recip_alphas_bar, t-self.ratio, x_t.shape)/extract(self.sqrt_recip_alphas_bar, t, x_t.shape))**2*(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            model_log_var1 = extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape)*beta_ts/extract(self.one_minus_alphas_bar, t, x_t.shape)
            model_log_var2 = beta_ts**2/(extract(self.one_minus_alphas_bar, t, x_t.shape) * extract(self.sqrt_recip_alphas_bar, t-self.ratio, x_t.shape)**2/extract(self.sqrt_recip_alphas_bar, t, x_t.shape)**2)
            model_log_var  = model_log_var1 + model_log_var2 * (eps2-eps**2)
        else:
            beta_ts = extract(self.one_minus_alphas_bar, t, x_t.shape)-(extract(self.sqrt_recip_alphas_bar, t-t, x_t.shape)/extract(self.sqrt_recip_alphas_bar, t, x_t.shape))**2*(extract(self.one_minus_alphas_bar, t-t, x_t.shape))
            model_log_var1 = extract(self.one_minus_alphas_bar, t-t, x_t.shape)*beta_ts/extract(self.one_minus_alphas_bar, t, x_t.shape)
            model_log_var2 = beta_ts**2/(extract(self.one_minus_alphas_bar, t, x_t.shape) * extract(self.sqrt_recip_alphas_bar, t-t, x_t.shape)**2/extract(self.sqrt_recip_alphas_bar, t, x_t.shape)**2)
            model_log_var  = model_log_var1 + model_log_var2 * (eps2-eps**2)
        return model_log_var,eps2

    # use eps and eps2 and eps3 to estimate one order moment
    def predict_xpre_3moment_from_eps(self, x_t, t, eps,eps2):
        eps3 = self.eps3_model(x_t, t)
        if (t-self.ratio)[0]>=0:
            a_ts = extract(self.sqrt_recip_alphas_bar, t-self.ratio, x_t.shape)/extract(self.sqrt_recip_alphas_bar, t, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
        else:
            a_ts = extract(self.sqrt_recip_alphas_bar, t-t, x_t.shape)/extract(self.sqrt_recip_alphas_bar, t, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2

        part1 = 1/(a_ts**3) * ((x_t**3) - 3*(x_t**2)*eps*(beta_ts/sigma_t)+3*(x_t)*eps2*(beta_ts**2/sigma_t**2)-(beta_ts/sigma_t)**3*eps3)
        part2 = 3*(sigma_s**2*beta_ts)/(sigma_t**2) * (1/a_ts) * (x_t-beta_ts/sigma_t*eps)
        third_moment = part1 + part2 
        return third_moment,eps3

    # use eps and eps2 and eps3 and eps4 to estimate one order moment
    def predict_xpre_4moment_from_eps(self, x_t, t, eps,eps2,eps3):
        eps4 = self.eps4_model(x_t, t)
        if (t-self.ratio)[0]>=0:
            # \alpha_{t|s}
            a_ts = extract(self.sqrt_recip_alphas_bar, t-self.ratio, x_t.shape)/extract(self.sqrt_recip_alphas_bar, t, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
        else:
            # \alpha_{t|s}
            a_ts = extract(self.sqrt_recip_alphas_bar, t-t, x_t.shape)/extract(self.sqrt_recip_alphas_bar, t, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-t, x_t.shape))
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2

        part1 = 1/(a_ts**4) * ((x_t**4)-4*(x_t**3)*(beta_ts/sigma_t)*eps+6*(x_t**2)*(beta_ts/sigma_t)**2*eps2-4*(x_t)*(beta_ts/sigma_t)**3*eps3+(beta_ts/sigma_t)**4*eps4)
        part2 = 6*1/(a_ts**2)*((x_t**2)-2*(x_t)*(beta_ts/sigma_t)*eps+(beta_ts/sigma_t)**2*eps2)*(sigma_s**2*beta_ts)/sigma_t**2
        part3 = 3*((sigma_s**2*beta_ts)/sigma_t**2)**2
        four_moment = part1 + part2 + part3
        return four_moment
        
    #@torch.no_grad()
    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations or Analytic-DPM
        # Mean parameterization
        if self.sample_type == 'ddpm':   # the model predicts epsilon
            eps = self.model(x_t, t)
            model_mean = self.predict_xpre_from_eps(x_t, t, eps=eps)
            #model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
            }['fixedsmall']
            model_log_var = extract(model_log_var, t, x_t.shape)
            return model_mean, torch.exp(model_log_var)

        elif self.sample_type == 'analyticdpm':
            assert self.cov_model is not None
            eps = self.model(x_t, t)
            x_0 = self.predict_xpre_from_eps(x_t, t, eps=eps)
            model_mean = x_0
            model_var,eps2 = self.predict_xpre_cov_from_eps(x_t, t, eps)
            return model_mean, model_var

        elif self.sample_type == 'gmddpm':
            assert self.eps3_model is not None
            assert self.eps4_model is not None
            eps  = self.model(x_t, t)
            eps2 = self.cov_model(x_t, t)
            eps3 = self.eps3_model(x_t, t)
            # mean function
            mean     = self.predict_xpre_from_eps(x_t, t, eps=eps)
            cov,eps2 = self.predict_xpre_cov_from_eps(x_t, t, eps)
            skeness,eps3  = self.predict_xpre_3moment_from_eps(x_t, t, eps,eps2)
            if self.eps4_model is not None:
                fmoment  = self.predict_xpre_4moment_from_eps(x_t, t, eps,eps2,eps3)
            else:
                fmoment  = None
            gt_var = torch.exp(extract(self.posterior_log_var_clipped, t, x_t.shape))
            return mean,cov,skeness,fmoment,gt_var
        else:
            raise NotImplementedError(self.sample_type)

    def forward(self, x_T):
        solve_type = 'pi'
        x_t = x_T
        for time_step in self.t_list:
            if time_step > 0:
                noise = torch.randn_like(x_t).to(x_T.device)
            else:
                noise = 0
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * time_step
            # sample with mixture of Gaussian
            if self.sample_type == 'gmddpm':
                mean,cov,tmoment,fmoment,gt_var = self.p_mean_variance(x_t=x_t, t=t)
                # clip the odd order moment
                cov = torch.clip(cov,1e-9,100)
                if fmoment is not None:
                    fmoment = torch.clip(fmoment,1e-9,100)
                pre_cov = gt_var

                random_matrics = torch.rand(size=mean.size()).to(mean.device)
                if solve_type == 'pi':
                    mean1,mean2,beta2,pi = solve_gmm(mean,cov,tmoment,fmoment,gt_var,solve_type)
                    gaussian1 = torch.tensor(torch.tensor(mean1)).to(x_T.device) + torch.sqrt(torch.tensor(pre_cov)).to(x_T.device) * noise
                    gaussian2 = torch.tensor(torch.tensor(mean2)).to(x_T.device) + torch.sqrt(torch.tensor(pre_cov*beta2)).to(x_T.device) * noise
                    x_t = torch.where(random_matrics<=pi,gaussian1,gaussian2)
                else:
                    mean1,mean2,beta1,beta2 = solve_gmm(mean,cov,tmoment,fmoment,gt_var,solve_type)
                    gaussian1 = torch.tensor(torch.tensor(mean1)).to(x_T.device) + torch.sqrt(torch.tensor(pre_cov*beta1)).to(x_T.device) * noise
                    gaussian2 = torch.tensor(torch.tensor(mean2)).to(x_T.device) + torch.sqrt(torch.tensor(pre_cov*beta2)).to(x_T.device) * noise
                    x_t = torch.where(random_matrics<=0.5,gaussian1,gaussian2)
            # sample with DDPM/Imperfect Analytic-DPM (Bao et al. (2022))
            else:
                mean, var = self.p_mean_variance(x_t=x_t, t=t)
                x_t = mean + torch.sqrt(var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
