import os
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import preprocessing

from scDGD.classes.representation import RepresentationLayer

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reshape_scaling_factor(x, out_dim):
    start_dim = len(x.shape)
    for i in range(out_dim - start_dim):
        x = x.unsqueeze(1)
    return x

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def gmm_clustering(gmm, rep):
    halfbeta = 0.5*torch.exp(gmm.logbeta)
    y = gmm.pi_term - (rep.unsqueeze(-2)-gmm.mean).square().mul(halfbeta).sum(-1) + gmm.betafactor*gmm.logbeta.sum(-1)
    # For each component multiply by mixture probs
    y += torch.log_softmax(gmm.weight,dim=0)
    return torch.exp(y)

def gmm_cluster_acc(r, gmm, labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    true_labels = le.transform(labels)
    clustering = gmm_clustering(gmm, r.z.detach())
    cluster_labels = torch.max(clustering, dim=-1).indices.cpu().detach()
    cm = confusion_matrix(true_labels, cluster_labels)
    indexes = linear_assignment(_make_cost_m(cm))
    cm2 = cm[:,indexes[1]]
    acc2 = np.trace(cm2) / np.sum(cm2)
    return acc2

def dgd_train(model, gmm, train_loader, validation_loader, n_epochs=500,
            export_dir='./', export_name='scDGD',
            lr_schedule_epochs=[0,300],lr_schedule=[[1e-3,1e-2,1e-2],[1e-4,1e-2,1e-2]], optim_betas=[0.5,0.7], wd=1e-4,
            acc_save_threshold=0.5,supervision_labels=None, wandb_logging=False):
    
    if wandb_logging:
        import wandb
    
    # prepare for saving the model
    if export_name is not None:
        if not os.path.exists(export_dir+export_name):
            os.makedirs(export_dir+export_name)
    
    # get some info from the data
    nsample = len(train_loader.dataset)
    nsample_test = len(validation_loader.dataset)
    out_dim = train_loader.dataset.n_genes
    latent = gmm.dim
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model=model.to(device)
    gmm=gmm.to(device)

    ###
    # set up representations and optimizers
    ###

    if lr_schedule_epochs is None:
        lr = lr_schedule[0]
        lr_rep = lr_schedule[1]
        lr_gmm = lr_schedule[2]
    else:
        lr = lr_schedule[0][0]
        lr_rep = lr_schedule[0][1]
        lr_gmm = lr_schedule[0][2]

    rep = RepresentationLayer(nrep=latent,nsample=nsample,values=torch.zeros(size=(nsample,latent))).to(device)
    test_rep = RepresentationLayer(nrep=latent,nsample=nsample_test,values=torch.zeros(size=(nsample_test,latent))).to(device)
    rep_optimizer = torch.optim.Adam(rep.parameters(), lr=lr_rep, weight_decay=wd,betas=(optim_betas[0],optim_betas[1]))
    testrep_optimizer = torch.optim.Adam(test_rep.parameters(), lr=lr_rep, weight_decay=wd,betas=(optim_betas[0],optim_betas[1]))

    if gmm is not None:
        gmm_optimizer = torch.optim.Adam(gmm.parameters(), lr=lr_gmm, weight_decay=wd,betas=(optim_betas[0],optim_betas[1]))
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd,betas=(optim_betas[0],optim_betas[1]))

    ###
    # start training
    ###

    # keep track of the losses and other metrics
    train_avg = []
    recon_avg = []
    dist_avg = []
    test_avg = []
    recon_test_avg = []
    dist_test_avg = []
    cluster_accuracies = []
    best_gmm_cluster = 0

    for epoch in range(n_epochs):
        
        # in case there is a scheduled change in learning rates, change them if the specified epoch is reached
        if lr_schedule_epochs is not None:
            if epoch in lr_schedule_epochs:
                lr_idx = [x for x in range(len(lr_schedule_epochs)) if lr_schedule_epochs[x] == epoch][0]
                lr_decoder = lr_schedule[lr_idx][0]
                model_optimizer = torch.optim.Adam(model.parameters(), lr=lr_decoder, weight_decay=wd, betas=(optim_betas[0],optim_betas[1]))
                lr_rep = lr_schedule[lr_idx][1]
                lr_gmm = lr_schedule[lr_idx][2]
                if gmm is not None:
                    gmm_optimizer = torch.optim.Adam(gmm.parameters(), lr=lr_gmm, weight_decay=wd, betas=(optim_betas[0],optim_betas[1]))
        
        # collect losses and other metrics for the epoch
        train_avg.append(0)
        recon_avg.append(0)
        dist_avg.append(0)
        test_avg.append(0)
        recon_test_avg.append(0)
        dist_test_avg.append(0)

        # train
        model.train()
        rep_optimizer.zero_grad()
        # standard mini batching
        for x,lib,i in train_loader:
            gmm_optimizer.zero_grad()
            model_optimizer.zero_grad()

            x = x.to(device)
            lib = lib.to(device)
            z = rep(i)
            y = model(z)

            # compute losses
            recon_loss_x = model.nb.loss(x, lib, y).sum()
            if supervision_labels is not None:
                sup_i = supervision_labels[i]
                gmm_error = -gmm(z,sup_i).sum()
            else:
                gmm_a, gmm_weights, gmm_prior = gmm.forward_split(z)
                gmm_likelihood = torch.logsumexp(gmm_a.clone()+gmm_weights.clone(), dim=-1)
                gmm_error = - (gmm_likelihood.clone() + gmm_prior.clone()).sum()
            loss = recon_loss_x.clone() + gmm_error.clone()
            
            # backpropagate and update
            loss.backward()
            gmm_optimizer.step()
            model_optimizer.step()

            # log losses
            train_avg[-1] += loss.item()/(nsample*out_dim)
            recon_avg[-1] += recon_loss_x.item()/(nsample*out_dim)
            dist_avg[-1] += gmm_error.item()/(nsample*latent)

        # update representations
        rep_optimizer.step()

        # validation run
        model.eval()
        testrep_optimizer.zero_grad()
        # same as above, but without updates for gmm and model
        for x,lib,i in validation_loader:
            x = x.to(device)
            lib = lib.to(device)
            z = test_rep(i)
            y = model(z)
            recon_loss_x = model.nb.loss(x, lib, y).sum()
            if supervision_labels is not None:
                gmm_error = -gmm(z).sum()
            else:
                gmm_a, gmm_weights, gmm_prior = gmm.forward_split(z)
                gmm_likelihood = torch.logsumexp(gmm_a.clone()+gmm_weights.clone(), dim=-1)
                gmm_error = - (gmm_likelihood.clone() + gmm_prior.clone()).sum()
            loss = recon_loss_x.clone() + gmm_error.clone()
            loss.backward()

            test_avg[-1] += loss.item()/(nsample_test*out_dim)
            recon_test_avg[-1] += recon_loss_x.item()/(nsample_test*out_dim)
            dist_test_avg[-1] += gmm_error.item()/(nsample_test*latent)
        testrep_optimizer.step()
        
        cluster_accuracies.append(gmm_cluster_acc(rep, gmm, train_loader.dataset.get_labels()))
        
        save_here = False
        if best_gmm_cluster < cluster_accuracies[-1]:
            best_gmm_cluster = cluster_accuracies[-1]
            best_gmm_epoch = epoch
            if best_gmm_cluster >= acc_save_threshold:
                save_here = True
        
        if wandb_logging:
            wandb.log({"loss_train": train_avg[-1],
                    "loss_test": test_avg[-1],
                    "loss_recon_train": recon_avg[-1],
                    "loss_recon_test": recon_test_avg[-1],
                    "loss_gmm_train": dist_avg[-1],
                    "loss_gmm_test": dist_test_avg[-1],
                    "cluster_accuracy": cluster_accuracies[-1],
                    "epoch": epoch})
        
        if export_name is not None:
            if save_here:
                print("model saved at epoch "+str(epoch)+" for having so far highest accuracy of "+str(cluster_accuracies[-1]))
                torch.save(model.state_dict(), export_dir+export_name+'/'+export_name+'_decoder.pt')
                torch.save(rep.state_dict(), export_dir+export_name+'/'+export_name+'_representation.pt')
                torch.save(test_rep.state_dict(), export_dir+export_name+'/'+export_name+'_valRepresentation.pt')
                torch.save(gmm.state_dict(), export_dir+export_name+'/'+export_name+'_gmm.pt')

    if wandb_logging:
        wandb.run.summary["best_gmm_cluster"] = best_gmm_cluster
        wandb.run.summary["best_gmm_epoch"] = best_gmm_epoch
    # create a history that is returned after training
    history = pd.DataFrame(
        {'train_loss': train_avg,
        'test_loss': test_avg,
        'train_recon_loss': recon_avg,
        'test_recon_loss': recon_test_avg,
        'train_gmm_loss': dist_avg,
        'test_gmm_loss': dist_test_avg,
        'cluster_accuracy': cluster_accuracies,
        'epoch': np.arange(1,n_epochs+1)
        })
    return model, rep, test_rep, gmm, history