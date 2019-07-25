"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import DGNet_Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy.random as random
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/latest.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='DGNet', help="DGNet")
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
opts = parser.parse_args()

str_ids = opts.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gpu_ids.append(int(str_id))
num_gpu = len(gpu_ids)

cudnn.benchmark = True

# Load experiment setting
if opts.resume:
    config = get_config('./outputs/'+opts.name+'/config.yaml')
else:
    config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'DGNet':
    trainer = DGNet_Trainer(config, gpu_ids)
    trainer.cuda()

if num_gpu>1:
    #torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8001',
    #                                     world_size = 1, rank = 0)
    #trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=gpu_ids)

    #trainer.teacher_model = torch.nn.DataParallel(trainer.teacher_model, gpu_ids)
    trainer.id_a = torch.nn.DataParallel(trainer.id_a, gpu_ids)
    trainer.gen_a.enc_content = torch.nn.DataParallel(trainer.gen_a.enc_content, gpu_ids)
    #trainer.gen_a.dec = torch.nn.DataParallel(trainer.gen_a.dec, gpu_ids)
    trainer.gen_a.mlp_w1 = torch.nn.DataParallel(trainer.gen_a.mlp_w1, gpu_ids)
    trainer.gen_a.mlp_w2 = torch.nn.DataParallel(trainer.gen_a.mlp_w2, gpu_ids)
    trainer.gen_a.mlp_w3 = torch.nn.DataParallel(trainer.gen_a.mlp_w3, gpu_ids)
    trainer.gen_a.mlp_w4 = torch.nn.DataParallel(trainer.gen_a.mlp_w4, gpu_ids)
    trainer.gen_a.mlp_b1 = torch.nn.DataParallel(trainer.gen_a.mlp_b1, gpu_ids)
    trainer.gen_a.mlp_b2 = torch.nn.DataParallel(trainer.gen_a.mlp_b2, gpu_ids)
    trainer.gen_a.mlp_b3 = torch.nn.DataParallel(trainer.gen_a.mlp_b3, gpu_ids)
    trainer.gen_a.mlp_b4 = torch.nn.DataParallel(trainer.gen_a.mlp_b4, gpu_ids)
    for dis_model in trainer.dis_a.cnns:
        dis_model = torch.nn.DataParallel(dis_model, gpu_ids)

random.seed(7) #fix random result
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
train_a_rand = random.permutation(train_loader_a.dataset.img_num)[0:display_size] 
train_b_rand = random.permutation(train_loader_b.dataset.img_num)[0:display_size] 
test_a_rand = random.permutation(test_loader_a.dataset.img_num)[0:display_size] 
test_b_rand = random.permutation(test_loader_b.dataset.img_num)[0:display_size] 

train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in train_a_rand]).cuda()
train_display_images_ap = torch.stack([train_loader_a.dataset[i][2] for i in train_a_rand]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in train_b_rand]).cuda()
train_display_images_bp = torch.stack([train_loader_b.dataset[i][2] for i in train_b_rand]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in test_a_rand]).cuda()
test_display_images_ap = torch.stack([test_loader_a.dataset[i][2] for i in test_a_rand]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in test_b_rand]).cuda()
test_display_images_bp = torch.stack([test_loader_b.dataset[i][2] for i in test_b_rand]).cuda()

# Setup logger and output folders
if not opts.resume:
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    shutil.copyfile('trainer.py', os.path.join(output_directory, 'trainer.py')) # copy file to output folder
    shutil.copyfile('reIDmodel.py', os.path.join(output_directory, 'reIDmodel.py')) # copy file to output folder
    shutil.copyfile('networks.py', os.path.join(output_directory, 'networks.py')) # copy file to output folder
else:
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", opts.name))
    output_directory = os.path.join(opts.output_path + "/outputs", opts.name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
config['epoch_iteration'] = round( train_loader_a.dataset.img_num  / config['batch_size'] )
print('Every epoch need %d iterations'%config['epoch_iteration'])
nepoch = 0 
    
print('Note that dataloader may hang with too much nworkers.')

while True:
    for it, ((images_a,labels_a, pos_a),  (images_b, labels_b, pos_b)) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        pos_a, pos_b = pos_a.cuda().detach(), pos_b.cuda().detach()
        labels_a, labels_b = labels_a.cuda().detach(), labels_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, labels_a, pos_a, images_b, labels_b, pos_b, config, iterations)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("\033[1m Epoch: %02d Iteration: %08d/%08d \033[0m" % (nepoch, iterations + 1, max_iter), end=" ")
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            del test_image_outputs

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            del image_outputs
        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

    # Save network weights by epoch number
    nepoch = nepoch+1
    if(nepoch + 1) % 10 == 0:
        trainer.save(checkpoint_directory, iterations)

