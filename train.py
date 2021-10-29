import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Datasets
parser.add_argument('-d', '--data', default='', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[75, 150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 1000)')
# Network
parser.add_argument('--depth', default=50, type=int, help='Model Depth')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Cut-Thumbnail parameter
parser.add_argument('--mode', type=str, default='mst', help='mixed single thumbnail or self thumbnail, [mst, st]')
parser.add_argument('--size', type=int, default=112, help='size of thumbnail')
parser.add_argument('--lam', type=float, default=0.25, help='weight of target for thumbnail' )
parser.add_argument('--participation_rate', type=float, default=0.8,
                    help='rate of batches applying Cut-Thumbnail' )


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0                    # best test accuracy
best_t5 = 0                     # top-5 accuracy
best_epoch = 0                  # best epoch


def main():
    global best_acc, best_t5, best_epoch
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if os.path.exists(args.data):
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        print('==> Preparing dataset in %s' % args.data)
    else:
        print('Error: no dataset directory found in %s!' % args.data)
        exit()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.depth == 18:
        model = resnet18(pretrained=False)
        print('===> Preparing ResNet-18')
    elif args.depth == 50:
        model = resnet50(pretrained=False)
        print('===> Preparing ResNet-50')

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, top_5 = test(val_loader, model, criterion, epoch, use_cuda)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best is True:
            best_epoch = epoch + 1
            best_t5 = top_5

    print('Best acc:')
    print(best_acc)
    print('Best Top_5:')
    print(best_t5)
    print('Best Epoch:')
    print(best_epoch)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        if np.random.rand(1) < args.participation_rate:
            # generate mixed sample with thumbnail
            if args.mode == 'mst':
                # mixed single thumbnail
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(2), args.size)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = F.interpolate(inputs[rand_index, :, :, :], (args.size, args.size))
            elif args.mode == 'st':
                # single thumbnail
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(2), args.size)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = F.interpolate(inputs, (args.size, args.size))
            else:
                print('Wrong prarameter of args.mode')
                exit()
            # compute output
            outputs = model(inputs)
            if args.mode == 'mst':
                loss = criterion(outputs, target_a) * (1.0 - args.lam) + criterion(outputs, target_b) * args.lam
            elif args.mode == 'st':
                loss = criterion(outputs, targets)
        else:
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % args.print_freq == 0:
            print('\rTraining | Epoch:{}/{}| Batch: {}/{}| Losses:{:.4f} | Top-1:{:.2f} | Top-5:{:.2f}'.format(
                epoch + 1, args.epochs, batch_idx + 1, len(train_loader), losses.avg, top1.avg, top5.avg),
                end='', flush=True)

    print('\rTraining | Epoch:{}/{}| Batch: {}/{}| Losses:{:.4f} | Top-1:{:.2f} | Top-5:{:.2f}'.format(
        epoch + 1, args.epochs, batch_idx + 1, len(train_loader), losses.avg, top1.avg, top5.avg), end='\n')
    return losses.avg, top1.avg


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            if (batch_idx+1) % args.print_freq == 0:
                print('Testing | Epoch:{}/{}| Batch: {}/{} | Losses:{:.4f} | Top-1:{:.2f} | Top-5:{:.2f}'.format(
                    epoch + 1, args.epochs, batch_idx + 1, len(val_loader), losses.avg, top1.avg, top5.avg),
                end='', flush=True)

    print('Testing | Epoch:{}/{}| Batch: {}/{} | Losses:{:.4f} | Top-1:{:.2f} | Top-5:{:.2f}'.format(
        epoch + 1, args.epochs, batch_idx + 1, len(val_loader), losses.avg, top1.avg, top5.avg), end='\n')
    return losses.avg, top1.avg, top5.avg


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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


def rand_bbox(inputs_size, dst_size):
    x = random.randint(0, inputs_size-dst_size)
    y = random.randint(0, inputs_size-dst_size)
    return x, y, x+dst_size, y+dst_size


if __name__ == '__main__':
    a = time.time()
    main()
    b = time.time()
    print('Total time: {}day {}hour'.format(int((b - a) / 86400), int((b - a) % 86400 / 3600)))
