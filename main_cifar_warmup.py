from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import AverageMeter, RecorderMeter, accuracy
from model.resnet import *


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

    return losses.avg, top1.avg, top5.avg

def test(args, model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.data.item(), data.size(0))

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))
    return losses.avg, top1.avg, top5.avg

def compute_M(args, model, criterion, device, train_loader, optimizer):
    model.train()
    grad_norm = []
    avg_grad = []
    for param in model.parameters():
        grad_norm.append(0)
        avg_grad.append(torch.zeros_like(param.data))
    N = len(train_loader)
   
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        for i, param in enumerate(model.parameters()):
            grad_norm[i] += torch.sum(param.grad.data.pow(2)) / N
            avg_grad[i] += param.grad.data / N

    sum_grad_norm = 0.
    sum_avg_grad_norm = 0.
    Ms = []
    for i in  range(len(grad_norm)):
        sum_grad_norm += grad_norm[i]
        sum_avg_grad_norm += torch.sum(avg_grad[i].pow(2))
        avg_grad[i] = torch.sum(avg_grad[i].pow(2)).item()
        grad_norm[i] = grad_norm[i].item()
        Ms.append(grad_norm[i] / avg_grad[i])
        
    return sum_grad_norm, sum_avg_grad_norm, grad_norm, avg_grad, Ms



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--ll', action='store_true', default=False,
                        help='layerwise learning rate')
    parser.add_argument('--cnn', action='store_true', default=False,
                        help='CNN networks')
    parser.add_argument('--no_bn', action='store_true', default=False,
                        help='batch norm in each layer')
    parser.add_argument('--dim_p', type=int, default=1,
                        help='dimmension product (default: 1)')
    parser.add_argument('--ll_p', type=float, default=1.,
                        help='learning rate discounts')
    parser.add_argument('--layer', type=int, default=1,
                        help='layer with multiplying lr')
    parser.add_argument('--M', action='store_true', default=False,
                        help='compute M for warmup')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
      transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize(mean, std)])

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    data_path = "../ResNeXt-DenseNet/data/cifar.python"
    train_data = datasets.CIFAR10(data_path, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        **kwargs)
    if args.M:
        M_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=False,
                        **kwargs)

    criterion = nn.CrossEntropyLoss()
    model = resnet8(dim_p=args.dim_p, no_bn=args.no_bn).to(device)

    if not args.ll:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    else:
        nlr_params = []
        lr_params = []
        cnt = 0
        for name, param in model.named_parameters():
            if 'conv' in name:
                cnt += 1
            if 'conv' in name and cnt-1 == args.layer:
                nlr_params.append(param)
            else:
                lr_params.append(param)

        params = [
		{'params':lr_params, 'lr':args.lr},
		{'params':nlr_params, 'lr':args.lr*args.ll_p}]
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

    names = []
    for name, _ in model.named_parameters():
        names.append(name)
    print(names)
    print('Epoch   train_loss  train_acc_top1  train_acc_top5   test_loss   test_acc_top1  test_acc_top5  avg_norm norm_avg M')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1, train_top5 = train(args, model, criterion, device, train_loader, optimizer, epoch)
        test_loss, test_top1, test_top5 = test(args, model, criterion, device, test_loader)
        if not args.M:
            avg_norm, norm_avg, M, ll, lr, Ms = 0, 0, 0, [], [], []
        else:
            avg_norm, norm_avg, ll, lr, Ms = compute_M(args, model, criterion, device, M_loader, optimizer)
            M = avg_norm / norm_avg
        print("{:d}   {:.4f}   {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}".format(epoch, train_loss, train_top1, train_top5, test_loss, test_top1, test_top5, avg_norm, norm_avg, M))
        print(ll)
        print(lr)
        print(Ms)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
