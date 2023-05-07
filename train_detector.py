from __future__ import print_function
import argparse
import torch

import models
import os

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from models.resnet_detector import Text_CNN_ResNet
from models.inception_detector import Text_CNN_Inception
from sklearn.metrics import roc_auc_score, roc_curve
from lib.average_meter import AverageMeter


parser = argparse.ArgumentParser(description='PyTorch code: New Adversarial Image Detection Based on Sentiment Analysis')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='/home/datasets/', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='resnet', help='resnet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | JSMA | EAD | DeepFool | CW | PGD')
parser.add_argument('--det_type', default='text-cnn', help='text-cnn')
args = parser.parse_args()
print(args)


def train(classifier, detector, train_loader, criterion, optimizer, i, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    classifier.eval()
    detector.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        feature = classifier.feature_list(data)[1]
        optimizer.zero_grad()
        output = detector(feature)
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

def evaluate(classifier, detector, val_loader):
    accs = AverageMeter()
    classifier.eval()
    detector.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            feature = classifier.feature_list(data)[1]
            output = detector(feature)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).float().sum(0)
            acc = correct.mul_(100.0 / data.size(0))

            accs.update(acc.item(), data.size(0))
    print('\nEvaluate val set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(accs.sum / 100.0), accs.count, accs.avg))
    return accs.avg

def compute_auc(classifier, detector, test_loader, plot=False):
    classifier.eval()
    detector.eval()
    labels, preds = 0, 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            feature = classifier.feature_list(data)[1]
            output = detector(feature)[:, 1]
            if batch_idx == 0:
                labels = target.clone().data.cpu()
                preds = output.clone().data.cpu()
            else:
                labels = torch.cat((labels, target.clone().data.cpu()), 0)
                preds = torch.cat((preds, output.clone().data.cpu()), 0)

    auc_score = roc_auc_score(labels.numpy(), preds.numpy())
    if plot:
        fpr, tpr, _ = roc_curve(labels, preds.numpy(), pos_label=1)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    print('\nTest set: AUC: {:.4f}\n'.format(auc_score))

def get_loader():
    print('load target data: ', args.dataset)
    list_name = ['Train', 'Val', 'Test']
    total_loader = []
    for i in range(len(list_name)):
        if not list_name[i] == 'Test':
            shuffle_flag = True
            path = args.outf + 'train/'

            clean_data = torch.load(
            path + '%s_clean_data_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            adv_data = torch.load(
                path + '%s_adv_data_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            label = torch.load(
                path + '%s_label_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))

            X = torch.cat((clean_data, adv_data), 0)
            Y = torch.cat((torch.zeros(len(label)), torch.ones(len(label))), 0)
            dataset = TensorDataset(X, Y)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle_flag, num_workers=4)

            total_loader.append(loader)
        else:
            shuffle_flag = False
            path = args.outf + 'test/'

            clean_data = torch.load(
            path + '%s_clean_data_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            adv_data = torch.load(
                path + '%s_adv_data_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            label = torch.load(
                path + '%s_label_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            
            X = torch.cat((clean_data, adv_data), 0)
            Y = torch.cat((torch.zeros(len(label)), torch.ones(len(label))), 0)
            dataset = TensorDataset(X, Y)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle_flag, num_workers=4)

            total_loader.append(loader)

    return total_loader     

def main():
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf  + '/' + args.net_type + '_' + args.dataset + '/'
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    if args.dataset == 'cifar100':
        args.num_classes = 100

    # load networks
    if args.net_type == 'resnet':
        classifier = models.ResNet34(num_c=args.num_classes)
        detector = Text_CNN_ResNet().cuda()
    else:
        classifier = models.inceptionv3()
        detector = Text_CNN_Inception().cuda()
    classifier.load_state_dict(torch.load(pre_trained_net, map_location="cpu"))
    classifier.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    train_loader, val_loader, test_loader = get_loader()

    # Training Detector
    if not os.path.isdir("./trained_detector/"):
        os.mkdir("./trained_detector/")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=0.0001)
    epoch = 10
    max_acc = 0

    print('Training Detector...')
    for i in range(epoch):
        train(classifier, detector, train_loader, criterion, optimizer, i, epoch)
        acc = evaluate(classifier, detector, val_loader)
        if acc > max_acc:
            print('saveing new best model ckpt for epoch #{}'.format(i + 1))
            torch.save(detector.state_dict(), "./trained_detector/%s_%s_%s_%s.pt" % (args.det_type, args.net_type, args.dataset, args.adv_type))
            max_acc = acc

    # Evaluating Detector
    print('Evaluating Detector...')
    detector.load_state_dict(torch.load("./trained_detector/%s_%s_%s_%s.pt" % (args.det_type, args.net_type, args.dataset, args.adv_type),
                                    map_location='cpu'))
    compute_auc(classifier, detector, test_loader, plot=False)

if __name__ == '__main__':
    main()