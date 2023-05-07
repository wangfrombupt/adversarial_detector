from __future__ import print_function
import argparse
from typing import Sequence
import torch
import data_loader
import numpy as np
import calculate_log as callog
import models
import os
import lib_generation


from models.text_cnn import Text_CNN, Multi_Text_CNN
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Polygon
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='/home/ltx/datasets/', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='resnet', help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', default='PGD', help='FGSM | BIM | DeepFool | CWL2 | PGD')
parser.add_argument('--det_type', default='text-cnn', help='cnn | text-cnn | text-rcnn | text-rnn')
args = parser.parse_args()
print(args)

import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# Random state.
RS = 20150101

type = 'test1'

def get_total_loader(adv_type):
    print('load target data: {} , adv type: {}'.format(args.dataset, adv_type))
    if type == 'test':
        list_name = 'Val'
        path = args.outf + 'train/'
    else:
        list_name = 'Test'
        path = args.outf + 'test/'

    clean_data = torch.load(
        path + '%s_clean_data_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))
    adv_data = torch.load(
        path + '%s_adv_data_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))
    label = torch.load(
        path + '%s_label_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))

    clean_dataset = TensorDataset(clean_data, label)
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    adv_dataset = TensorDataset(adv_data, label)
    adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return clean_loader, adv_loader


def get_class_loader(adv_type, class_num):
    print('load target data: {} , class num: {}'.format(args.dataset, class_num))
    if type == 'test':
        list_name = 'Val'
        path = args.outf + 'train/'
    else:
        list_name = 'Test'
        path = args.outf + 'test/'

    selected_list = []

    clean_data = torch.load(
        path + '%s_clean_data_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))
    adv_data = torch.load(
        path + '%s_adv_data_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))
    label = torch.load(
        path + '%s_label_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))

    for i in range(len(label)):
        if label[i] == class_num:
            selected_list.append(i)

    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data, 0, selected_list)
    label_tot = torch.index_select(label, 0, selected_list)

    clean_dataset = TensorDataset(clean_data_tot, label_tot)
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return clean_loader


def get_adv_loader(model, adv_type, target_class, source_class=0):
    print('load target data: {} , adv class: {}'.format(args.dataset, target_class))
    if type == 'test':
        list_name = 'Val'
        path = args.outf + 'train/'
    else:
        list_name = 'Test'
        path = args.outf + 'test/'

    selected_list = []

    clean_data = torch.load(
        path + '%s_clean_data_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))
    adv_data = torch.load(
        path + '%s_adv_data_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))
    label = torch.load(
        path + '%s_label_%s_%s_%s.pth' % (list_name, args.net_type, args.dataset, adv_type))

    adv_dataset = TensorDataset(adv_data, label)
    adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    pred = 0
    for batch_idx, (data, _) in enumerate(adv_loader):
        data = data.cuda()
        output = model(data)
        if batch_idx == 0:
            pred = output.argmax(dim=1, keepdim=True)
        else:
            pred = torch.cat((pred, output.argmax(dim=1, keepdim=True)), 0)

    for i in range(len(pred)):
        # if pred[i] == target_class and label[i] == source_class:
        if pred[i] == target_class:
            selected_list.append(i)

    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data, 0, selected_list)
    label_tot = torch.index_select(label, 0, selected_list)

    clean_dataset = TensorDataset(clean_data_tot, label_tot)
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    adv_dataset = TensorDataset(adv_data_tot, label_tot)
    adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return clean_loader, adv_loader


def get_deep_representations(model, data_loader, layer_index):
    model.eval()
    feature = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        num_data = data.shape[0]
        if batch_idx == 0:
            feature = model.intermediate_forward(data, layer_index).detach().cpu().numpy().reshape(num_data, -1)
        else:
            feature = np.concatenate((feature, model.intermediate_forward(data, layer_index).detach().cpu().numpy().reshape(num_data, -1)), axis=0)

    return feature


def get_det_seq(model, detector, data_loader, layer_index):
    model.eval()
    seq = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        num_data = data.shape[0]
        if batch_idx == 0:
            feature = model.feature_list(data)[1]
            seq = detector.get_seq(feature).detach().cpu().numpy()[:, layer_index]
        else:
            feature = model.feature_list(data)[1]
            batch_seq = detector.get_seq(feature).detach().cpu().numpy()[:, layer_index]
            seq = np.concatenate((seq, batch_seq), axis=0)

    return seq


def scatter(x, colors, idx, adv_list):
    # We choose a color palette with seaborn.
    # label = [str(i) for i in range(args.num_classes)]
    # label = [' '] * args.num_classes
    # label.extend(adv_list)
    # label.append('adv')
    label = ['clean', args.adv_type]

    palette = np.array(sns.color_palette("hls", len(label)))
    palette[[0, -1], :] = palette[[-1, 0], :]

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []

    # if idx == 4:
    for i in range(len(label)):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(label[i]), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def draw_tsne(model, detector, class_loader, adv_list, idx):
    # X_features = get_deep_representations(model, class_loader[0], idx)
    X_features = get_det_seq(model, detector, class_loader[0], idx)
    Y = np.zeros(X_features.shape[0])
    for i in range(1, len(class_loader)):
        X_features_class = get_det_seq(model, detector, class_loader[i], idx)
        # if idx < 4:
        #     if i < 10:
        #         label = np.full(X_features_class.shape[0], 0)
        #     else:
        #         label = np.full(X_features_class.shape[0], 1)
        # else:
        label = np.full(X_features_class.shape[0], i)

        X_features = np.vstack((X_features, X_features_class))
        Y = np.hstack((Y, label))
    
    print(X_features.shape)
    print(Y.shape)
    fig_names = ['bn1', 'res1', 'res2', 'res3', 'res4']
    path = './figures/tsne/total/'
    # path = './figures/tsne/multi/'
    if not os.path.exists(path):
        os.mkdir(path)

    digits_proj = TSNE(random_state=RS, verbose=1).fit_transform(X_features)
    scatter(digits_proj, Y, idx, adv_list)

    save_path = path + args.adv_type + '_' + args.dataset + '_' + fig_names[idx] + '.pdf'
    plt.savefig(save_path, dpi=120)


def main():
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100

    # load networks
    if args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
 
    model.eval()
    model.cuda()
    print('load model: ' + args.net_type)

    # for i in range(100):
    #     _, adv_loader = get_adv_loader(model, adv_type=args.adv_type, target_class=i)
    #     print(i, len(adv_loader.dataset))

    detector = Text_CNN()
    detector.load_state_dict(torch.load("./trained_model/%s_%s_%s_%s.pt" % (args.det_type, args.net_type, args.dataset, args.adv_type), map_location="cuda:" + str(args.gpu)))
    detector.eval()
    detector.cuda()

    # load dataset
    # if args.dataset == 'cifar100':
    #     target_class = 16  # 42
    # elif args.dataset == 'svhn':
    #     target_class = 2
    # else:
    #     target_class = 2

    # class_loader = []
    # for class_num in range(0, args.num_classes):
    #     class_loader.append(get_class_loader(adv_type=args.adv_type, class_num=class_num))

    # # for adv in adv_list:
    # _, adv_loader = get_adv_loader(model, adv_type=args.adv_type, target_class=target_class)
    # class_loader.append(adv_loader)
    # print(args.adv_type, len(adv_loader.dataset))
    clean_loader, adv_loader = get_total_loader(args.adv_type)
    print(len(clean_loader.dataset), len(adv_loader.dataset))

    for i in range(0, 5):
        draw_tsne(model, detector, [clean_loader, adv_loader], args.adv_type, i)

if __name__ == '__main__':
    main()