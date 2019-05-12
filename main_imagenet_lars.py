# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import convert_secs2time, time_string, time_file_str
#from models import print_log
import models
from lars_optimizer import LARSOptimizer


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='n', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning_rate', default=0.128, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')

parser.add_argument('--warmup', type=int, default=5, help='Number of epochs for warmup.')
parser.add_argument('--steps', type=int, default=1, help='Number of steps before applying gradient.')
parser.add_argument('--lars', action='store_true', default=False, help='lars optimizer')
parser.add_argument('--eta', type=float, default=0.001, help='eta for lars optimizer')
parser.add_argument('--lw', action='store_true', default=False, help='layerwise learning rate')
parser.add_argument('--M_iters', type=int, default=64, help='approximate M')
parser.add_argument('--lw_epochs', type=int, default=200, help='epochs computing M')
parser.add_argument('--lw_eta', type=float, default=0.01, help='eta for lars optimizer')

args = parser.parse_args()
args.prefix = time_file_str()

def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
      os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch,args.prefix)), 'w')

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](1000)
    print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
      model.features = torch.nn.DataParallel(model.features)
      model.cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=[x for x in range(args.ngpu)]).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    """
    param_lrs = []
    params = []
    names = []
    layers = [3,] + [12,]+[9,]*2+[12,]+[9,]*3+[12,]+[9,]*5+[12,]+[9,]*2+[2,]
    for i, (name, param) in enumerate(model.named_parameters()):
        params.append(param)
        names.append(name)
        if len(params) == layers[0]:
            param_dict = {'params': params, 'lr':args.learning_rate}
            param_lrs.append(param_dict)
            print(names)
            params = []
            names = []
            layers.pop(0)
    """
    skip_lists = ['bn', 'bias']
    skip_idx = []
    for idx, (name, param) in enumerate(net.named_parameters()):
        if any(skip_name in name for skip_name in skip_lists):
            skip_idx.append(idx)

    param_lrs = model.parameters()
    if args.lars:  
        optimizer = LARSOptimizer(param_lrs, args.learning_rate, momentum=args.momentum,
                weight_decay=args.weight_decay, nesterov=False, steps=args.steps, eta=args.eta)
    else:
        optimizer = optim.SGD(param_lrs, state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    filename = os.path.join(args.save_dir, 'checkpoint.{}.{}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(args.save_dir, 'best.{}.{}.pth.tar'.format(args.arch, args.prefix))


    print_log('Epoch  Train_Prec@1  Train_Prec@5  Train_Loss  Test_Prec@1  Test_Prec@5  Test_Loss  Best_Prec@1  Time', log)
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        start_time = time.time()
        train_top1, train_top5, train_loss = train(train_loader, model, criterion, optimizer, epoch, log)
        training_time = time.time() - start_time

        # evaluate on validation set
        val_top1, val_top5, val_loss = validate(val_loader, model, criterion, log)

        # remember best prec@1 and save checkpoint
        is_best = val_top1 > best_prec1
        best_prec1 = max(val_top1, best_prec1)

        print('{epoch:d}        {train_top1:.3f}      {train_top5:.3f}     {train_loss:.3f}      {test_top1:.3f}      {test_top5:.3f}    {test_loss:.3f}    {best_top1:.3f}      {time:.3f} '.format(epoch=epoch, time=training_time, train_top1=train_top1, train_top5=train_top5, train_loss=train_loss, test_top1=val_top1, test_top5=val_top5, test_loss=val_loss, best_top1=best_prec1))
		
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename, bestname)
        # measure elapsed time

    log.close()


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    num_batches = len(train_loader) // args.steps
    warmup_steps = num_batches * args.warmup
    total_steps = num_batches * args.epochs

    if epoch == 0:
        if args.lw:
            compute_M(model, criterion, optimizer, M_loader, avg_norm, iters=args.M_iters)
        optimizer.zero_grad()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if (epoch*len(train_loader)+i) % args.steps == 0:
            poly_lr_rate((epoch*len(train_loader)+i)//args.steps, warmup_steps, total_steps, optimizer, args.learning_rate*args.batch_size*args.steps/256, args.lw) 
		
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if (epoch*len(train_loader)+i+1) % args.steps == 0:
            if args.lars:
                optimizer.step(avg_norm=avg_norm)
            else:
                optimizer.step()

            if args.lw and epoch < args.lw_epochs:
                compute_M(model, criterion, optimizer, M_loader, avg_norm, iters=args.M_iters)
                optimizer.eta = args.lw_eta 
            else:
                avg_norm = []
                optimizer.eta = args.eta

            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return top1.avg, top5.avg, losses.avg



def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, top5.avg, losses.avg


def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_lr_rate(current_steps, warmup_steps, total_steps, optimizer, lr, lw=False):

  # poly + warmup
  if current_steps < warmup_steps:
    current_lr = lr * current_steps / warmup_steps
  else:
    decay_steps = max(current_steps-warmup_steps, 1)
    current_lr = polynomial_decay(lr, decay_steps, total_steps-warmup_steps+1, power=2.0)

  if not lw:
    for param_group in optimizer.param_groups:
      param_group['lr'] = current_lr
  else:
    num_params = len(optimizer.param_groups)
    if current_steps < warmup_steps:
      min_lr = current_lr / 2
      for j, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = min((num_params-j)/num_params*(current_lr-min_lr)+min_lr, lr)
    else:
      min_lr = current_lr / 2
      for j, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = min((j+1)/num_params * min_lr+min_lr, lr)


def polynomial_decay(lr, global_step, decay_steps, end_lr=0.0001,  power=1.0):
  global_step = min(global_step, decay_steps)
  return (lr-end_lr) * pow(1-global_step/decay_steps, power) + end_lr
  

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_M(model, criterion, optimizer, M_loader, avg_norm, iters=8):
  for i in range(len(avg_norm)):
    avg_norm[i] = 0

  for i, (inputs, target) in  enumerate(M_loader):
    if args.use_cuda:
      inputs = inputs.cuda(async=True)
      target = target.cuda()

    input_var = torch.autograd.Variable(inputs)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()

    for j, param in enumerate(model.parameters()):
      avg_norm[j] += torch.norm(param.grad.data) / iters

    if i >= iters-1:
      break

if __name__ == '__main__':
    main()
