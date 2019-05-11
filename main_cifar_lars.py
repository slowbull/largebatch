from __future__ import division

import os, sys, shutil, time, random, math
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models

import numpy as np

from lars_optimizer import LARSOptimizer


def main(args):
  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("torch  version : {}".format(torch.__version__), log)
  print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

  # Init dataset
  if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

  if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  elif args.dataset == 'imagenet32x32':
    mean = [x / 255 for x in [122.7, 116.7, 104.0]] 
    std = [x / 255 for x in [66.4, 64.6, 68.4]]
  elif args.dataset == 'svhn':
    pass
  else:
    assert False, "Unknow dataset : {}".format(args.dataset)

  if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet32x32':
    train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
      transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize(mean, std)])

  if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100
  elif args.dataset == 'svhn':
    def target_transform(target):
      return int(target[0])-1
    train_data = dset.SVHN(args.data_path, split='train', transform=transforms.Compose(
        [transforms.ToTensor(),]), download=True, target_transform=target_transform)
    extra_data = dset.SVHN(args.data_path, split='extra', transform=transforms.Compose(
        [transforms.ToTensor(),]), download=True, target_transform=target_transform)
    train_data.data = np.concatenate([train_data.data, extra_data.data])
    train_data.labels = np.concatenate([train_data.labels, extra_data.labels])
    print(train_data.data.shape, train_data.labels.shape)
    test_data = dset.SVHN(args.data_path, split='test', transform=transforms.Compose([transforms.ToTensor(),]), download=True, target_transform=target_transform)
    num_classes = 10
  elif args.dataset == 'imagenet32x32':
    train_data = IMAGENET32X32(args.data_path, train=True, transform=train_transform, download=True)
    test_data = IMAGENET32X32(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 1000
  else:
    assert False, 'Do not support dataset : {}'.format(args.dataset)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

  print_log("=> creating model '{}'".format(args.arch), log)
  # Init model, criterion, and optimizer
  net = models.__dict__[args.arch](num_classes=num_classes)

  #print_log("=> network:\n {}".format(net), log)

  net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

  # define loss function (criterion) and optimizer
  criterion = torch.nn.CrossEntropyLoss()

  """
  params_skip = []
  params_noskip = []
  skip_lists = ['bn', 'bias']
  for name, param in net.named_parameters():
    if any(name in skip_name for skip_name in skip_lists):
      params_skip.append(param)
    else:
      params_noskip.append(param)
  param_lrs = [{'params':params_skip, 'lr':state['learning_rate']},
		{'params':params_noskip, 'lr':state['learning_rate']}]
  """ 
  param_lrs = []
  params = []
  names = []
  layers = [3,] + [54,]*3 + [2,]
  for i, (name, param) in enumerate(net.named_parameters()):
    params.append(param)
    names.append(name)
    if len(params) == layers[0]:
      param_dict = {'params': params, 'lr':state['learning_rate']}
      param_lrs.append(param_dict)
      params = []
      names = []
      layers.pop(0)
      
  optimizer = LARSOptimizer(param_lrs, state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=False, steps=state['steps'], eta=state['eta'])

  if args.use_cuda:
    net.cuda()
    criterion.cuda()

  recorder = RecorderMeter(args.epochs)
  # optionally resume from a checkpoint

  # Main loop
  print_log('Epoch  Train_Prec@1  Train_Prec@5  Train_Loss  Test_Prec@1  Test_Prec@5  Test_Loss  Best_Prec@1  Time', log)
  for epoch in range(args.start_epoch, args.epochs):

    # train for one epoch
    start_time = time.time()
    train_top1, train_top5, train_loss = train(train_loader, net, criterion, optimizer, epoch, log, args)
    training_time = time.time() - start_time

    # evaluate on validation set
    val_top1, val_top5, val_loss = validate(test_loader, net, criterion, log, args)
    recorder.update(epoch, train_loss, train_top1, val_loss, val_top1)

    print('{epoch:d}        {train_top1:.3f}      {train_top5:.3f}     {train_loss:.3f}      {test_top1:.3f}      {test_top5:.3f}    {test_loss:.3f}    {best_top1:.3f}      {time:.3f} '.format(epoch=epoch, time=training_time, train_top1=train_top1, train_top5=train_top5, train_loss=train_loss, test_top1=val_top1, test_top5=val_top5, test_loss=val_loss, best_top1=recorder.max_accuracy(False)))


  log.close()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, args):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  # switch to train mode
  model.train()

  num_batches = len(train_loader) // args.steps
  warmup_steps = num_batches * args.warmup
  total_steps = num_batches * args.epochs
  f_time = 0
  b_time = 0
  d_time = 0
  start_time = time.time()
  if epoch == 0:
    optimizer.zero_grad()
  for i, (inputs, target) in enumerate(train_loader):
    if (epoch*len(train_loader)+i) % args.steps == 0:
      poly_lr_rate((epoch*len(train_loader)+i)//args.steps, warmup_steps, total_steps, optimizer, args.learning_rate*args.batch_size*args.steps/128, args.lw) 

    if args.use_cuda:
      inputs = inputs.cuda(async=True)
      target = target.cuda()

    input_var = torch.autograd.Variable(inputs)
    target_var = torch.autograd.Variable(target)
    d_time += time.time() - start_time

    # compute output
    start_time = time.time()
    output = model(input_var)
    loss = criterion(output, target_var)


    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.item(), inputs.size(0))
    top1.update(prec1.item(), inputs.size(0))
    top5.update(prec5.item(), inputs.size(0))
    f_time += time.time() - start_time

    # compute gradient and do SGD step
    start_time = time.time()
    loss.backward()

    if (epoch*len(train_loader)+i+1) % args.steps == 0:
      optimizer.step()
      optimizer.zero_grad()
    b_time += time.time() - start_time

    start_time = time.time()
  #print('trainingtime : f_time : {} b_time: {} d_time: {}'.format(f_time, b_time, d_time))
  return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, log, args):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():

    for i, (inputs, target) in enumerate(val_loader):
      if args.use_cuda:
        inputs = inputs.cuda()
        target = target.cuda()
      input_var = torch.autograd.Variable(inputs)
      target_var = torch.autograd.Variable(target)

      # compute output
      output = model(input_var)
      loss = criterion(output, target_var)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
      losses.update(loss.data.item(), inputs.size(0))
      top1.update(prec1.item(), inputs.size(0))
      top5.update(prec5.item(), inputs.size(0))

  return top1.avg, top5.avg, losses.avg

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

def save_checkpoint(state, is_best, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best.pth.tar')
    shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule, lr):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  #lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr

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
        param_group['lr'] = min(pow((num_params-j)/num_params, 1-(current_steps)/warmup_steps)*(current_lr-min_lr)+min_lr, lr)
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
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

if __name__ == '__main__':
  model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

  parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_path', type=str, help='Path to dataset')
  parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'svhn', 'imagenet32x32'], help='Choose between Cifar10/100 and ImageNet.')
  parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
  # Optimization options
  parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
  parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
  parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
  parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
  parser.add_argument('--decay', type=float, default=0.0002, help='Weight decay (L2 penalty).')
  parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
  parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
  # warmup
  parser.add_argument('--warmup', type=int, default=5, help='Number of epochs for warmup.')
  parser.add_argument('--steps', type=int, default=1, help='Number of steps before applying gradient.')
  parser.add_argument('--lw', action='store_true', default=False, help='layerwise learning rate')
  parser.add_argument('--eta', type=float, default=0.001, help='eta for lars optimizer')
  # Checkpoints
  parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
  parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
  parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
  parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
  parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
  # Acceleration
  parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
  parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
  # random seed
  parser.add_argument('--manualSeed', type=int, help='manual seed')
  args = parser.parse_args()
  args.use_cuda = args.ngpu>0 and torch.cuda.is_available()


  if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
  torch.manual_seed(args.manualSeed)
  cudnn.benchmark = True # find the fastest cudnn conv algorithm

  main(args)

