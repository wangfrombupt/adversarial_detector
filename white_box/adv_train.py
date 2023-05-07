from __future__ import print_function
import argparse
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import data_loader

import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
from models.resnet_detector import Text_CNN
from lib.adversary import PGD
from lib.average_meter import AverageMeter


parser = argparse.ArgumentParser(description='PyTorch code: New Adversarial Image Detection Based on Sentiment Analysis')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='/home/datasets/', help='path to dataset')
parser.add_argument('--outf', default='../adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='resnet', help='resnet')
parser.add_argument('--det_type', default='text-cnn', help='text-cnn')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | JSMA | EAD | DeepFool | CW | PGD | PGD-20')
args = parser.parse_args()
print(args)

def get_loader():
    print('load origin data: %s , adv_type: %s' % (args.dataset, args.adv_type))
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

def evaluate_pgd(classifier, detector, attacker, test_loader):
    print('\nEvaluating detector on PGD %s white box attack.' % attacker.mode)
    if attacker.mode == 'combine':
        print('sigma : ', attacker.sigma)

    atks_clf = AverageMeter()
    accs_det = AverageMeter()

    classifier.eval()
    detector.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()

        adv_data = attacker(data, target)
        output_clf = classifier(adv_data)
        pred_clf = output_clf.argmax(dim=1, keepdim=True).squeeze()

        selected_list = (pred_clf != target.view_as(pred_clf)).nonzero(as_tuple=False)
        adv_data = torch.index_select(adv_data, 0, selected_list.squeeze())

        atk_clf = selected_list.size(0) * 100.0 / data.size(0)
        atks_clf.update(atk_clf, data.size(0))

        features = classifier.feature_list(adv_data)[1]
        output_det = detector(features)
        pred_det = output_det.argmax(dim=1, keepdim=True)

        label = torch.ones(adv_data.size(0)).long().cuda()
        correct_det = pred_det.eq(label.view_as(pred_det)).float().sum(0)
        acc_det = correct_det.mul_(100.0 / adv_data.size(0))
        accs_det.update(acc_det.item(), adv_data.size(0))

    print('Attack Success Rate: {}/{} ({:.2f}%)\n'.format(int(atks_clf.sum / 100.0), atks_clf.count, atks_clf.avg))
    print('Detector Final Accuracy: {}/{} ({:.2f}%)\n'.format(int(accs_det.sum / 100.0), accs_det.count, accs_det.avg))

def evaluate_standard(classifier, detector, test_loader):
    print('Evaluating detector on original test set.')
    accs = AverageMeter()
    classifier.eval()
    detector.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            features = classifier.feature_list(data)[1]
            output = detector(features)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).float().sum(0)
            acc = correct.mul_(100.0 / data.size(0))

            accs.update(acc.item(), data.size(0))
    print('\nEvaluate original test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(accs.sum / 100.0), accs.count, accs.avg))
    return accs.avg

def train(classifier, detector, static_detector, attacker, clean_train_loader, origin_train_loader, criterion, optimizer, i, epoch):
    classifier.eval()
    detector.train()
    static_detector.eval()

    losses_origin = AverageMeter()
    losses_wb = AverageMeter()
    accs_origin = AverageMeter()
    accs_wb = AverageMeter()
    for batch_idx, data in enumerate(zip(cycle(clean_train_loader), origin_train_loader)):
        clean_data, clean_target, origin_data, origin_target = data[0][0], data[0][1], data[1][0], data[1][1]
        origin_data, origin_target = origin_data.cuda(), origin_target.cuda()
        features_origin = classifier.feature_list(origin_data)[1]

        # train on original dataset
        optimizer.zero_grad()
        output_origin = detector(features_origin)
        loss_origin = criterion(output_origin, origin_target.long())
        loss_origin.backward()
        losses_origin.update(loss_origin.item(), origin_data.size(0))
        optimizer.step()

        pred_origin = output_origin.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_origin = pred_origin.eq(origin_target.view_as(pred_origin)).float().sum(0)
        acc_origin = correct_origin.mul_(100.0 / origin_data.size(0))
        accs_origin.update(acc_origin.item(), origin_data.size(0))

        # generate adv images by white box attack
        wb_data = attacker(clean_data, clean_target)
        output_wb = classifier(wb_data)
        pred_wb = output_wb.argmax(dim=1, keepdim=True).squeeze()

        # select the adv samples that are attacked successfully
        selected_list = (pred_wb != clean_target.view_as(pred_wb).cuda()).nonzero(as_tuple=False)
        wb_data = torch.index_select(wb_data, 0, selected_list.squeeze())
        features_wb = classifier.feature_list(wb_data)[1]

        # train on white box attack samples
        optimizer.zero_grad()
        output_wb = detector(features_wb)
        label = torch.ones(wb_data.size(0)).long().cuda()
        loss_wb = criterion(output_wb, label)
        loss_wb.backward()
        losses_wb.update(loss_wb.item(), wb_data.size(0))
        optimizer.step()

        pred_wb = output_wb.argmax(dim=1, keepdim=True)
        corrcet_wb = pred_wb.eq(label.view_as(pred_wb)).float().sum(0)
        acc_wb = corrcet_wb.mul_(100.0 / wb_data.size(0))
        accs_wb.update(acc_wb.item(), wb_data.size(0))

        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{}]\nWhite Box Loss: {:.6f}\tWhite Box Accuracy: {}/{} ({:.2f}%)   \
        #           \nOrigin Loss : {:.6f}\t\tOrigin Accuracy: {}/{} ({:.2f}%)'.format(
        #           i+1, i+1, epoch, losses_wb.avg, int(accs_wb.sum / 100.0), accs_wb.count, accs_wb.avg,
        #           losses_origin.avg, int(accs_origin.sum / 100.0), accs_origin.count, accs_origin.avg))
    
    print('Train Epoch: {} [{}/{}]\nWhite Box Loss: {:.6f}\tWhite Box Accuracy: {}/{} ({:.2f}%)   \
          \nOrigin Loss : {:.6f}\t\tOrigin Accuracy: {}/{} ({:.2f}%)'.format(
          i+1, i+1, epoch, losses_wb.avg, int(accs_wb.sum / 100.0), accs_wb.count, accs_wb.avg,
          losses_origin.avg, int(accs_origin.sum / 100.0), accs_origin.count, accs_origin.avg))


def main():
    # set the path to pre-trained model and output
    pre_trained_net = '../pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100
        label_smooth = 0.01
    else:
        label_smooth = 0.1

    # load networks
    classifier = models.ResNet34(num_c=args.num_classes)
    classifier.load_state_dict(torch.load(pre_trained_net, map_location='cpu'))
    in_transform = transforms.Compose([transforms.ToTensor()])

    classifier.eval()
    classifier.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    print('load target data: ', args.dataset)
    clean_train_loader, clean_val_loader, clean_test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform,
                                                                         args.dataroot)
    origin_train_loader, origin_val_loader, origin_test_loader = get_loader()

    # Training Detector
    path = './adv_trained_detector'
    if not os.path.isdir(path):
        os.mkdir(path)

    detector = Text_CNN().cuda()
    static_detector = Text_CNN().cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(detector.parameters(), lr=0.0001)

    epoch = 5
    pgd_iter_20 = PGD(classifier, detector, label_smooth, None, mode='iter', eps=8/255, steps=20)
    for i in range(epoch):
        if i == 0:
            detector.load_state_dict(torch.load("../trained_detector/%s_%s_%s_%s.pt" % (args.det_type, args.net_type, args.dataset, args.adv_type)))
            static_detector.load_state_dict(torch.load("../trained_detector/%s_%s_%s_%s.pt" % (args.det_type, args.net_type, args.dataset, args.adv_type)))
            evaluate_standard(classifier, detector, origin_test_loader)
            evaluate_pgd(classifier, detector, pgd_iter_20, clean_test_loader)
        else:
            print('Update static detector weights.')
            static_detector.load_state_dict(torch.load("%s/%s_%s_%s_%s.pt" % (path, args.det_type, args.net_type, args.dataset, str(i))))

        pgd_iter_2 = PGD(classifier, static_detector, label_smooth, None, mode='iter', eps=8/255, steps=2)
        train(classifier, detector, static_detector, pgd_iter_2, clean_train_loader, origin_train_loader, criterion, optimizer, i, epoch)
        evaluate_standard(classifier, detector, origin_val_loader)
        evaluate_pgd(classifier, detector, pgd_iter_20, clean_val_loader)

        print('saveing new model ckpt for epoch #{}'.format(i + 1))
        torch.save(detector.state_dict(),
                "%s/%s_%s_%s_%s.pt" % (path, args.det_type, args.net_type, args.dataset, str(i+1)))

if __name__ == '__main__':
    main()