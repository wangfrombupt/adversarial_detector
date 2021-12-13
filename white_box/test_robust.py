from __future__ import print_function
import argparse
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import data_loader

from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from models.text_cnn import Text_CNN
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
parser.add_argument('--resume', type=int, required=True, help='resume epoch')
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
    _, _, clean_test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform,
                                                                         args.dataroot)
    _, _, origin_test_loader = get_loader()

    # Training Detector
    path = './adv_trained_detector'

    detector = Text_CNN().cuda()
    detector.load_state_dict(torch.load("%s/%s_%s_%s_%s.pt" % (path, args.det_type, args.net_type, args.dataset, str(args.resume))))

    pgd_iter_20 = PGD(classifier, detector, label_smooth, None, mode='iter', eps=8/255, steps=20)
    pgd_comb_20 = PGD(classifier, detector, label_smooth, sigma=0.3, mode='combine', eps=8/255, steps=20)
    
    evaluate_standard(classifier, detector, origin_test_loader)           
    evaluate_pgd(classifier, detector, pgd_iter_20, clean_test_loader)
    evaluate_pgd(classifier, detector, pgd_comb_20, clean_test_loader)

if __name__ == '__main__':
    main()