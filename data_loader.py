import torch
from torchvision import datasets
import os
import numpy as np


def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    if not os.path.exists(data_root + '/val_indices.npy'):
        print('did not get val indices, therefore splitting train set.')
        indices = np.random.permutation(73257)
        cnt_train = 49000
        cnt_val = 1000
        train_indices = indices[0:cnt_train]
        val_indices = indices[cnt_train:cnt_train+cnt_val]
        np.save(data_root + '/train_indices.npy', train_indices)
        np.save(data_root + '/val_indices.npy', val_indices)
    if not os.path.exists(data_root + '/test_indices.npy'):
        print('did not get test indices, therefore splitting test set.')
        indices = np.random.permutation(26032)
        cnt_test = 10000
        test_indices = indices[0:cnt_test]
        np.save(data_root + '/test_indices.npy', test_indices)
    else:
        train_indices = np.load(data_root + '/train_indices.npy')
        val_indices = np.load(data_root + '/val_indices.npy')
        test_indices = np.load(data_root + '/test_indices.npy')

    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train = datasets.SVHN(root=data_root, split='train', download=True, transform=TF)
        train_dataset = torch.utils.data.Subset(train, train_indices)
        val_dataset = torch.utils.data.Subset(train, val_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
        ds.append(val_loader)
    if val:
        test = datasets.SVHN(root=data_root, split='test', download=True, transform=TF)
        test_dataset = torch.utils.data.Subset(test, test_indices)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    if not os.path.exists(data_root + '/val_indices.npy'):
        print('did not get val indices, therefore splitting train set.')
        indices = np.random.permutation(50000)
        cnt_train = 49000
        train_indices = indices[0:cnt_train]
        val_indices = indices[cnt_train:]
        np.save(data_root + '/train_indices.npy', train_indices)
        np.save(data_root + '/val_indices.npy', val_indices)
    else:
        train_indices = np.load(data_root + '/train_indices.npy')
        val_indices = np.load(data_root + '/val_indices.npy')

    kwargs.pop('input_size', None)
    ds = []
    if train:
        train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=TF)
        train_dataset = torch.utils.data.Subset(train, train_indices)
        val_dataset = torch.utils.data.Subset(train, val_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
        ds.append(val_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0:2] if len(ds) == 2 else ds
    return ds


def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    if not os.path.exists(data_root + '/val_indices.npy'):
        print('did not get val indices, therefore splitting train set.')
        indices = np.random.permutation(50000)
        cnt_train = 49000
        train_indices = indices[0:cnt_train]
        val_indices = indices[cnt_train:]
        np.save(data_root + '/train_indices.npy', train_indices)
        np.save(data_root + '/val_indices.npy', val_indices)
    else:
        train_indices = np.load(data_root + '/train_indices.npy')
        val_indices = np.load(data_root + '/val_indices.npy')

    kwargs.pop('input_size', None)
    ds = []
    if train:
        train = datasets.CIFAR100(root=data_root, train=True, download=True,transform=TF)
        train_dataset = torch.utils.data.Subset(train, train_indices)
        val_dataset = torch.utils.data.Subset(train, val_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
        ds.append(val_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, val_loader, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, val_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, val_loader, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)

    return train_loader, val_loader, test_loader
