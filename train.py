from __future__ import print_function
import os
import argparse
import torch
import data_loader
import models

from torchvision import transforms
from lib.reduce_lr_on_plateau import ReduceLROnPlateau
from lib.transforms import Pad, Crop
from lib.average_meter import AverageMeter


parser = argparse.ArgumentParser(description='PyTorch code: New Adversarial Image Detection Based on Sentiment Analysis')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='/home/datasets/', help='path to dataset')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='resnet', help='resnet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)


def adjust_learning_rate(optimizer, reduce_lr_on_plateau):
    lr = reduce_lr_on_plateau.get_curr_lr()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(dnn, train_loader, criterion, optimizer, i, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    dnn.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = dnn(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).float().sum(0)
        acc = correct.mul_(100.0 / data.size(0))

        losses.update(loss.item(), data.size(0))
        accs.update(acc.item(), data.size(0))

    print('Train Epoch: [{}/{}]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
        i+1, epoch, losses.avg, int(accs.sum / 100.0), accs.count, accs.avg))

def test(dnn, test_loader):
    accs = AverageMeter()
    dnn.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = dnn(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).float().sum(0)
            acc = correct.mul_(100.0 / data.size(0))
            accs.update(acc.item(), data.size(0))

    print('\nTest total set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(accs.sum / 100.0), accs.count, accs.avg))
    return accs.avg

def main():
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    if args.dataset == 'cifar100':
        args.num_classes = 100
        label_smooth = 0.01
    else:
        label_smooth = 0.1

    if not os.path.exists('./pre_trained'):
        os.mkdir('./pre_trained')
    pre_trained_net = "./pre_trained/" + args.net_type + '_' + args.dataset + '.pth'

    # load networks
    model = models.ResNet34(num_c=args.num_classes)
    in_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), Pad(),
                                        Crop(crop_type='random', crop_frac=0.8), ])
    model.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, val_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    train_args = {
        'learning_rate': 0.1,
        'lr_factor': 0.9,
        'lr_patience': 3,
        'lr_cooldown': 2,
    }
    reduce_lr_on_plateau = ReduceLROnPlateau(factor=train_args['lr_factor'], patience=train_args['lr_patience'],
                                             cooldown=train_args['lr_cooldown'], init_lr=train_args['learning_rate'])

    loss = models.LabelSmoothingCrossEntropy(eps=label_smooth)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0004, nesterov=True)

    epoch = 200
    print("set optimizer. curr_lr={}".format(reduce_lr_on_plateau.get_curr_lr()))
    reduce_lr_on_plateau.on_train_begin()
    for i in range(epoch):
        adjust_learning_rate(optimizer, reduce_lr_on_plateau)
        train(model, train_loader, loss, optimizer, i, epoch)
        acc = test(model, val_loader)
        reduce_lr_on_plateau.on_epoch_end(i, acc)
        if reduce_lr_on_plateau.was_improvement():
            print('saveing new best model ckpt for epoch #{}'.format(i + 1))
            torch.save(model.state_dict(), pre_trained_net)

    model.load_state_dict(torch.load(pre_trained_net, map_location="cpu"))
    test(model, test_loader)


if __name__ == '__main__':
    main()
