import copy
import json
import os
import math
import warnings
from absl import app, flags
import logging
import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from data import ImageNet,LSUNBed
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from libs.iddpm import UNetModel,UNetModel4Pretrained,UNetModel4Pretrained3
from score.both import get_inception_and_fid_score
from adan import Adan

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_enum('exp_name', 'CIFAR10', ['CIFAR10','IMAGENET','LSUN'], help='name of experiment')
# UNet: IDDPM
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
flags.DEFINE_enum('mode', 'simple', ['simple','complex','complex2'], help='the mode for honn modeling')

# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion training noising steps')
flags.DEFINE_enum('sample_type', 'ddpm', ['ddpm', 'analyticdpm', 'gmddpm'], help='sample type for sampling')
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
flags.DEFINE_string('pretrained_dir', './logs/iDDPM_CIFAR10_EPS1/models/ckpt_1_800000.pt', help='log directory')
flags.DEFINE_bool('time_shift', False, help='whether the noised data is from t=1')
flags.DEFINE_bool('rescale_time', True, help='adjust the maxmimum time to input the network is 1000')
flags.DEFINE_bool('nll_training', False, help='training the model to fit the noise.pow(a)')
flags.DEFINE_enum('noise_schedule', 'linear', ['linear','cosine'], help='the mode for honn modeling')
flags.DEFINE_enum('model_type', 'noise', ['noise', 'nll'], help='variance type')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/iDDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
flags.DEFINE_integer('sample_steps', 1000, help='Sampling steps for generation stage')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_integer('clip_pixel', 2, "Var Clip for final steps")
flags.DEFINE_integer('save_step', 50000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')

# Model Dir
flags.DEFINE_string('eps1_dir', './logs/iDDPM_CIFAR10_EPS/models/ckpt_1_300000.pt', help='eps1 model log directory')
flags.DEFINE_string('eps2_dir', './logs/iDDPM_CIFAR10_EPS2/models/ckpt_2_300000.pt', help='eps2 model log directory')
flags.DEFINE_string('eps3_dir', './logs/iDDPM_CIFAR10_complex_EPS3/models/ckpt_3_300000.pt', help='eps3 model log directory')

device = torch.device('cuda:0')

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            print(i)
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def train():
    # dataset
    logging.info('train on {0}'.format(FLAGS.exp_name))
    if FLAGS.exp_name == 'CIFAR10':
        dataset = CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
            
    elif FLAGS.exp_name == 'IMAGENET':
        dataset = ImageNet()
    elif FLAGS.exp_name == 'LSUN':
        dataset = LSUNBed()
    else:
        pass
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # modeling up to the training order of noise
    if FLAGS.noise_order == 1:
        if FLAGS.exp_name != 'LSUN':
            net_model = UNetModel(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=FLAGS.use_scale_shift_norm,)
        else:
            from libs.ddpm import Model
            net_model = Model(in_channels=FLAGS.in_channel,ch=FLAGS.ch,out_ch=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attn_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            ch_mult=tuple(FLAGS.ch_mult),resamp_with_conv=True,resolution=FLAGS.img_size)   
            try:
                ckpt1 = torch.load(FLAGS.pretrained_dir)
                net_model.load_state_dict(ckpt1)
                logging.info('Sucess Configs')
            except:
                logging.info('Mistakes on Configs')
        #net_model = UNet(
        #        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=[1],
        #        num_res_blocks=2, dropout=0.1)
        
    elif FLAGS.noise_order == 2:
        logging.info('training nosie order is {0}-th'.format(FLAGS.noise_order))
        if FLAGS.exp_name != 'LSUN':
            net_model = UNetModel4Pretrained(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=FLAGS.use_scale_shift_norm,
            head_out_channels=FLAGS.head_out_channels,mode=FLAGS.mode)
        else:
            from libs.ddpm import Model4Pretrained
            #logging.info(FLAGS.ch_mult)
            net_model = Model4Pretrained(in_channels=FLAGS.in_channel,ch=FLAGS.ch,out_ch=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attn_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            ch_mult=tuple(FLAGS.ch_mult),resamp_with_conv=True,resolution=FLAGS.img_size,mode=FLAGS.mode,head_out_ch=3)     
        logging.info(FLAGS.pretrained_dir)
        try:
            pretrained_dict = torch.load(FLAGS.pretrained_dir)['ema_model']
        except:
            pretrained_dict = torch.load(FLAGS.pretrained_dir)
        model_dict = net_model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        net_model.load_state_dict(model_dict)

    elif FLAGS.noise_order == 3:
        logging.info('training nosie order is {0}-th'.format(FLAGS.noise_order))
        if FLAGS.exp_name != 'LSUN':
            net_model = UNetModel4Pretrained3(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=FLAGS.use_scale_shift_norm,
            head_out_channels=FLAGS.head_out_channels,mode=FLAGS.mode)
        else:
            from libs.ddpm import Model4Pretrained
            #logging.info(FLAGS.ch_mult)
            net_model = Model4Pretrained(in_channels=FLAGS.in_channel,ch=FLAGS.ch,out_ch=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attn_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
            ch_mult=tuple(FLAGS.ch_mult),resamp_with_conv=True,resolution=FLAGS.img_size,mode=FLAGS.mode,head_out_ch=3)     
        logging.info(FLAGS.pretrained_dir)

        try:
            pretrained_dict = torch.load(FLAGS.pretrained_dir)['ema_model']
        except:
            pretrained_dict = torch.load(FLAGS.pretrained_dir)
        model_dict = net_model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        net_model.load_state_dict(model_dict)
    logging.info('first beta for linear noise schedule is {0}'.format(FLAGS.beta_1))
    ema_model = copy.deepcopy(net_model)
    #optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)

    optim = Adan(net_model.parameters(),lr=FLAGS.lr,betas=(0.9,0.92,0.92))
    #warm_up = 1000
    #min_lr  = 1e-5
    #warm_up_with_cosine_lr = lambda iter: (iter) / warm_up if iter <= warm_up \
    #    else max(0.5 * ( math.cos((iter - warm_up) /(FLAGS.total_steps - warm_up) * math.pi) + 1), 
    #    min_lr / FLAGS.lr)

    #sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    #sched = torch.optim.lr_scheduler.LambdaLR(optim, warm_up_with_cosine_lr)
    
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T,FLAGS.noise_order,FLAGS.noise_schedule,FLAGS.time_shift,FLAGS.rescale_time,FLAGS.nll_training).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type,FLAGS.noise_schedule,FLAGS.time_shift,FLAGS.rescale_time).to(device)

    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type,FLAGS.noise_schedule,FLAGS.time_shift,FLAGS.rescale_time).to(device)

    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    sample_path = os.path.join(FLAGS.logdir, 'sample')
    #if not os.path.exits(sample_path):
    try:
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
        os.makedirs(os.path.join(FLAGS.logdir, 'models'))
    except:
        pass

    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device).float()
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            #sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)
            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    #'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                ckpt_path = 'models/ckpt_' + str(int(FLAGS.noise_order)) +'_'+ str(step) +'.pt'
                logging.info('model dir is {0}'.format(ckpt_path))
                torch.save(ckpt, os.path.join(FLAGS.logdir, ckpt_path))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()

"""
def eval():
    # model setup
    from libs.iddpm import UNetModel
    model = UNetModel(in_channels=FLAGS.in_channel,model_channels=FLAGS.ch,out_channels=FLAGS.out_channel,num_res_blocks=FLAGS.num_res_blocks,attention_resolutions=FLAGS.attn,dropout=FLAGS.dropout,
        channel_mult=FLAGS.ch_mult,conv_resample=True,dims=FLAGS.dims,num_classes=None,use_checkpoint=False,num_heads=FLAGS.num_heads,num_heads_upsample=-1,use_scale_shift_norm=False,)
    eps2_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)

    #sampler = GaussianDiffusionSampler(
    #    model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
    #    mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    
    net_sampler = GaussianDiffusionSamplergm(
        model, FLAGS.beta_1, FLAGS.beta_T, 1000, FLAGS.img_size,
        'analyticdpm',eps2_model).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    #ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    #model.load_state_dict(ckpt['net_model'])
    #(IS, IS_std), FID, samples = evaluate(sampler, model)
    #print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    #save_image(
    #    torch.tensor(samples[:256]),
    #    os.path.join(FLAGS.logdir, 'samples.png'),
    #    nrow=16)
    ckpt1 = torch.load('./logs/DDPM_CIFAR10_EPS/ckpt.pt')
    model.load_state_dict(ckpt1['ema_model'])

    ckpt2 = torch.load('./logs/DDPM_CIFAR10_EPS2/ckpt_2.pt')
    eps2_model.load_state_dict(ckpt2['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)
"""

def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
