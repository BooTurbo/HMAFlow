import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hmaflow import HMAFlow
import evaluate
import datasets

torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exclude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    # We should put the declaration of optimizer after freezing parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
     
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        # ['1px', '3px', '5px', 'epe']
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter() 

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]
        
        # 99 for first call 
        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter() 

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    model = nn.DataParallel(HMAFlow(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        # model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        model.load_state_dict(torch.load(args.restore_ckpt)['params'], strict=False)
        print('loading restore checkpoint done!')
    
    if args.resume_point is not None:
        load_ckpt = torch.load(args.resume_point)
        model.load_state_dict(load_ckpt['params'], strict=False)
        print('loading resume point done!')

    model.cuda()
    model.train()
    
    if args.stage != 'chairs':
        model.module.freeze_bn()  
    
    # Init DataLoader(), train_dataset
    train_loader = datasets.fetch_dataloader(args) 
    optimizer, scheduler = fetch_optimizer(args, model)
    
    # if resume
    if args.resume_point is not None:
        optimizer.load_state_dict(load_ckpt['optimizer'])
        scheduler.load_state_dict(load_ckpt['scheduler'])
        print("=> loading last total_steps: {}".format(load_ckpt['total_steps']))
        print("last lr is: {:10.7f}".format(scheduler.get_last_lr()[0]))
        total_steps = 1 + load_ckpt['total_steps']
    else:
        total_steps = 0
            
    logger = Logger(model, scheduler)

    if args.resume_point is not None:
        logger.total_steps = total_steps


    VAL_FREQ = 5000

    should_keep_training = True
    while should_keep_training:
        # iter=2223, epochs=(100001/2223):45
        for i_batch, data_blob in enumerate(train_loader):    
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)  
            
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            loss.backward()                
            
            optimizer.step()
            scheduler.step()

            logger.push(metrics)  
            

            if total_steps % VAL_FREQ == VAL_FREQ - 1:    # 4999     
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                # Save checkpoint
                if args.save_resume_point:    
                    checkpoint = {'params': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'scheduler': scheduler.state_dict(),
                                  'total_steps': total_steps}    # 4999 
                    torch.save(checkpoint, PATH)
                else: 
                    torch.save(model.state_dict(), PATH)


                results, validate_results = {}, {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        ret1 = evaluate.validate_chairs(model.module)
                        results.update(ret1)
                        
                        iters_epe = 'Iteration {} EPE'.format(total_steps + 1)
                        validate_results[iters_epe] = ret1['chairs']
                    elif val_dataset == 'sintel':
                        ret1, ret2 = evaluate.validate_sintel(model.module)
                        results.update(ret1)

                        iters_epe = 'Iteration {} clean'.format(total_steps + 1)
                        validate_results[iters_epe] = ret2['clean']
                        iters_epe = 'Iteration {} final'.format(total_steps + 1)
                        validate_results[iters_epe] = ret2['final']
                    elif val_dataset == 'kitti':
                        ret1 = evaluate.validate_kitti(model.module)
                        results.update(ret1)

                        iters_kitti = 'Iteration {} KITTI'.format(total_steps + 1)                    
                        value_epe_f1 = 'EPE {}, F1 {}'.format(ret1['kitti-epe'], ret1['kitti-f1'])
                        validate_results[iters_kitti] = value_epe_f1
                        

                with open('validation_epe.txt', 'a') as f:
                    for k, v in sorted(validate_results.items()):
                        f.write(str(k) + ': ' + str(v) +'\n')
                

                logger.write_dict(results)
                
                model.train()

                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            # Actually training 100001 times
            if total_steps > args.num_steps:
                should_keep_training = False
                break
    
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='hmaflow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--save_resume_point', type=bool, default=True)
    parser.add_argument('--resume_point', help='path to resume checkpoint')
    
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)

