import torch
import numpy as np
from sklearn.model_selection import train_test_split
from scDGD.classes import scDataset

def prepate_data(adata, label_column, train_fraction=0.8, include_test=True, scaling_type='max', batch_size=256, num_w=0):
    '''
    Prepares the pytorch data sets and loaders for training and testing

    For integrating a new data set, set the train_fraction to 1. Otherwise there should always be something left for validation.
    If include_test is True, the split will also include a held-out test set. Otherwise, it will only be train and validation.
    '''

    ###
    # first create a data split
    ###
    labels = adata.obs[label_column]

    train_mode = True
    if 'train_val_test' not in adata.obs.keys():
        if train_fraction < 1.0:
            if include_test:
                train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=(1.0-train_fraction)/2, stratify=labels)
                train_indices, val_indices = train_test_split(train_indices, test_size=(((1.0-train_fraction)/2)/(1.0-(1.0-train_fraction)/2)), stratify=labels[train_indices])
                # add the split to the anndata object
                train_val_test = [''] * len(labels)
                train_val_test = ['train' if i in train_indices else train_val_test[i] for i in range(len(labels))]
                train_val_test = ['validation' if i in val_indices else train_val_test[i] for i in range(len(labels))]
                train_val_test = ['test' if i in test_indices else train_val_test[i] for i in range(len(labels))]
            else:
                train_indices, val_indices = train_test_split(np.arange(len(labels)), test_size=(1.0-train_fraction), stratify=labels)
                train_val_test = [''] * len(labels)
                train_val_test = ['train' if i in train_indices else train_val_test[i] for i in range(len(labels))]
                train_val_test = ['validation' if i in val_indices else train_val_test[i] for i in range(len(labels))]
        else:
            train_mode = False
            train_val_test = 'test'
    else:
        if len(set(adata.obs['train_val_test'])) == 1:
            train_mode = False
    adata.obs['label'] = labels
    adata.obs['train_val_test'] = train_val_test
    # make sure to afterwards also return the adata object so that the data split can be re-used

    ###
    # then create the data sets and loaders
    ###

    if train_mode:
        trainset = scDataset(
        adata.X,
        adata.obs,
        scaling_type=scaling_type,
        subset=np.where(adata.obs['train_val_test']=='train')[0],
        label_type='label'
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_w)
        validationset = scDataset(
            adata.X,
            adata.obs,
            scaling_type=scaling_type,
            subset=np.where(adata.obs['train_val_test']=='validation')[0],
            label_type='label'
        )
        validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=num_w)
        if len(set(adata.obs['train_val_test'])) == 3:
            testset = scDataset(
                adata.X,
                adata.obs,
                scaling_type=scaling_type,
                subset=np.where(adata.obs['train_val_test']=='test')[0],
                label_type='label'
            )
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_w)
        else:
            testset, testloader = None, None
    else:
        testset = scDataset(
            adata.X,
            adata.obs,
            scaling_type=scaling_type,
            subset=np.where(adata.obs['train_val_test']=='test')[0],
            label_type='label'
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_w)
    
    return adata, trainloader, validationloader, testloader




    