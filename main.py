from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import AverageMeter, RecorderMeter, accuracy

class FCBNNet(nn.Module):
    def __init__(self, p=1):
        super(FCBNNet, self).__init__()
        dim = int(p*200)
        self.fc1 = nn.Linear(28*28, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc3 = nn.Linear(dim, dim)
        self.bn3 = nn.BatchNorm1d(dim)
        self.fc4 = nn.Linear(dim, 200)
        self.bn4 = nn.BatchNorm1d(200)
        self.fc5 = nn.Linear(200, 10)

    def forward(self, x):
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        '''
        x = self.bn1(torch.sigmoid(self.fc1(x)))
        x = self.bn2(torch.sigmoid(self.fc2(x)))
        x = self.bn3(torch.sigmoid(self.fc3(x)))
        x = self.bn4(torch.sigmoid(self.fc4(x)))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)
    

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 10)
        
    def forward(self, x):
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        '''
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 4, 2) # 28x28 -> 13x13
        self.conv2 = nn.Conv2d(20, 20, 5, 1) # 13x13 -> 9x9
        self.conv3 = nn.Conv2d(20, 20, 5, 1) # 9x9 -> 5x5
        self.conv4 = nn.Conv2d(20, 20, 3, 1) # 5x5 -> 3x3
        self.conv5 = nn.Conv2d(20, 10, 3, 1) # 3x3 -> 1x1

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

class BNNet(nn.Module):
    def __init__(self, p=1):
        super(BNNet, self).__init__()
        dim = int(20*p)
        self.conv1 = nn.Conv2d(1, dim, 4, 2) # 28x28 -> 13x13
        self.conv2 = nn.Conv2d(dim, dim, 5, 1) # 13x13 -> 9x9
        self.conv3 = nn.Conv2d(dim, dim, 5, 1) # 9x9 -> 5x5
        self.conv4 = nn.Conv2d(dim, 20, 3, 1) # 5x5 -> 3x3
        self.conv5 = nn.Conv2d(20, 10, 3, 1) # 3x3 -> 1x1
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.bn4 = nn.BatchNorm2d(20)

    def forward(self, x):
        x = self.bn1(torch.sigmoid(self.conv1(x)))
        x = self.bn2(torch.sigmoid(self.conv2(x)))
        x = self.bn3(torch.sigmoid(self.conv3(x)))
        x = self.bn4(torch.sigmoid(self.conv4(x)))
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if not args.cnn:
           data = data.view(-1, 28*28)
        else:
            data = data
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

    return losses.avg, top1.avg, top5.avg

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if not args.cnn:
                data = data.view(-1, 28*28)
            else:
                data = data
            output = model(data)
            loss = F.nll_loss(output, target)

            losses.update(loss.data.item(), data.size(0))

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))
    #test_loss /= len(test_loader.dataset)
    return losses.avg, top1.avg, top5.avg


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
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
    parser.add_argument('--bn', action='store_true', default=False,
                        help='batch norm in each layer')
    parser.add_argument('--dim_p', type=float, default=1.,
                        help='dimmension product (default: 1)')
    parser.add_argument('--ll_p', type=float, default=0.1,
                        help='learning rate discounts')
    parser.add_argument('--layer', type=int, default=1,
                        help='layer with multiplying lr')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    M_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10, shuffle=False, **kwargs)

    if not args.cnn:
        if not args.bn:
            model = FCNet().to(device)
        else:
            model = FCBNNet(p=args.dim_p).to(device)
    else:
        if not args.bn:
           model = Net().to(device)
        else:
            model = BNNet(p=args.dim_p).to(device)
    model_params = []
    if not args.cnn:
        model_params.append(model.fc1.parameters())
        model_params.append(model.fc2.parameters())
        model_params.append(model.fc3.parameters())
        model_params.append(model.fc4.parameters())
        model_params.append(model.fc5.parameters())
    else:
        model_params.append(model.conv1.parameters())
        model_params.append(model.conv2.parameters())
        model_params.append(model.conv3.parameters())
        model_params.append(model.conv4.parameters())
        model_params.append(model.conv5.parameters())

    nlr_params = model_params[args.layer-1]
    lr_params = []
    for i in range(5):
        if i+1 != args.layer:
            lr_params += list(model_params[i])

    if not args.ll:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        print('layerwise learning rate!')
        params = [
                {'params': lr_params, 'lr':args.lr}, 
                {'params': nlr_params, 'lr':args.lr*args.ll_p}, 
                ]
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)

    print('Epoch   train_loss  train_acc_top1  train_acc_top5   test_loss   test_acc_top1  test_acc_top5 ')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1, train_top5 = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_top1, test_top5 = test(args, model, device, test_loader)

        print("{:d}  {:.4f}  {:.4f}   {:.4f}  {:.4f}  {:.4f}  {:.4f}".format(epoch, train_loss, train_top1, train_top5, test_loss, test_top1, test_top5))

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
