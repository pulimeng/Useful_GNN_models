

import os
import os.path as osp
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import json 
import random

import torch
import torch.nn as nn
from torch.optim import SGD, Adadelta, AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

from data_utils import MyGraphDataset
from models import GCNnet, GINnet, GATnet, MPNNnet, GENnet

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('Current device: ' + str(device))

def create_model(model_name, opt):
    if model_name == 'GCN':
        model = GCNnet(input_dim=opt['input_dim'], output_dim=opt['output_dim'],
                       num_conv_layers=opt['num_conv_layers'], hidden_dim=opt['hidden_dim'],
                       batch_norm=opt['batch_norm'], jumping_knowledge=opt['jumping_knowledge'], 
                       global_pool=opt['global_pool'], process_step=opt['process_step'],
                       dense_dims=opt['dense_dims'], dropout=opt['dropout'])
        return model
    elif model_name == 'GAT':
        model = GATnet(input_dim=opt['input_dim'], output_dim=opt['output_dim'],
                       num_conv_layers=opt['num_conv_layers'], hidden_dim=opt['hidden_dim'],
                       num_heads=opt['num_heads'], batch_norm=opt['batch_norm'],
                       jumping_knowledge=opt['jumping_knowledge'], global_pool=opt['global_pool'],
                       process_step=opt['process_step'], dense_dims=opt['dense_dims'], 
                       dropout=opt['dropout'])
        return model
    elif model_name == 'GIN':
        model = GINnet(input_dim=opt['input_dim'], output_dim=opt['output_dim'],
                       num_conv_layers=opt['num_conv_layers'], hidden_dim=opt['hidden_dim'],
                       batch_norm=opt['batch_norm'], jumping_knowledge=opt['jumping_knowledge'], 
                       global_pool=opt['global_pool'], process_step=opt['process_step'],
                       dense_dims=opt['dense_dims'], dropout=opt['dropout'])
        return model
    elif model_name == 'MPNN':
        model = MPNNnet(input_dim=opt['input_dim'], output_dim=opt['output_dim'],
                        num_conv_layers=opt['num_conv_layers'], edge_input_dim=opt['edge_input_dim'],
                        edge_hidden_dim=opt['edge_hidden_dim'], hidden_dim=opt['hidden_dim'],
                        batch_norm=opt['batch_norm'], aggr=opt['aggr'],
                        jumping_knowledge=opt['jumping_knowledge'], global_pool=opt['global_pool'],
                        process_step=opt['process_step'], dense_dims=opt['dense_dims'], 
                        dropout=opt['dropout'])
        return model
    elif model_name == 'GEN':
        model = GENnet(input_dim=opt['input_dim'], output_dim=opt['output_dim'],
                       num_conv_layers=opt['num_conv_layers'], edge_input_dim=opt['edge_input_dim'],
                       embed_dim=opt['embed_dim'], edge_hidden_dim=opt['edge_hidden_dim'],
                       hidden_dim=opt['hidden_dim'], batch_norm=opt['batch_norm'],
                       aggregator=opt['aggregator'], learn=opt['learn'],
                       msg_norm=opt['msg_norm'], mlp_layers=opt['mlp_layers'],
                       jumping_knowledge=opt['jumping_knowledge'], global_pool=opt['global_pool'],
                       process_step=opt['process_step'], dense_dims=opt['dense_dims'],
                       dropout=opt['dropout'])
        return model
    else:
        raise('Unrecognize model')
    
def train(data_path, label_path, model_name, save_dir):
    
    with open('./params.json') as f:
        opt = json.load(f)
        
    output_path = os.path.join(save_dir, 'ckpts')
    log_path = os.path.join(save_dir, 'logs')
    config_path = os.path.join(save_dir, 'configs')
    os.mkdir(output_path)
    os.mkdir(log_path)
    os.mkdir(config_path)
    d = opt
    print(json.dumps(d, indent=4))
    with open(os.path.join(config_path, 'params.json'), 'w') as f:
        json.dump(d, f)
    
    if opt['manual_seed'] is None:
        opt['manual_seed'] = random.randint(1, 10000)
        print('Random Seed: ', opt['manual_seed'])
        random.seed(opt['manual_seed'])
        torch.manual_seed(opt['manual_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opt['manual_seed'])

    #model settings
    #load model
    model = create_model(model_name, opt)
    model = model.to(device)
    print('Total Params:', np.sum([p.numel() for p in model.parameters()]))
    print(model)
    # --------------------------------------------------------------------------
    #loss --> change if need to
    criterion = nn.CrossEntropyLoss().to(device)
    # --------------------------------------------------------------------------
    #optimizer
    if opt['optimizer'] == 'sgd':
        optimizer = SGD(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    elif opt['optimizer'] == 'adam':
        optimizer = AdamW(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    elif opt['optimizer'] == 'adadelta':
        optimizer = Adadelta(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    else:  # default to Adam
        optimizer = AdamW(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    # --------------------------------------------------------------------------
    # cross validation setup
    Nfold = 5
    df = pd.read_csv(label_path)
    skf = KFold(n_splits=Nfold, shuffle=True)
    f = 0
    dataset = MyGraphDataset(data_path, df, device=device)
    
    tr_losses = np.zeros((Nfold, opt['num_epochs']))
    val_losses = np.zeros((Nfold, opt['num_epochs']))
    val_acc = np.zeros((Nfold, opt['num_epochs']))
    val_cnf = np.zeros((Nfold, opt['output_dim'], opt['output_dim']))
    
    for tr_idx, val_idx in skf.split(df):
        print('===================Fold {} starts==================='.format(f+1))
        training_set = Subset(dataset, tr_idx)
        validation_set = Subset(dataset, val_idx)
        training_loader = DataLoader(training_set, batch_size=opt['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=opt['batch_size'], shuffle=False)

        model.reset_parameters()
        best_val_loss = 1e6
        with tqdm(total=opt['num_epochs']) as pbar:
            for epoch in range(opt['num_epochs']):
                model.train()
                losses = []
                for i, batch in enumerate(training_loader):          
                    optimizer.zero_grad()
                    output = model(batch) # TODO: change this line accordingly based on your data formatting
                    y_true = batch[1].to(device) # TODO: change this line accordingly based on your data formatting
                    loss = criterion(output, y_true)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.cpu().detach().numpy())
                mean_tr_loss = np.mean(np.array(losses))
                
                model.eval()
                losses = []
                y_preds = []
                y_trues = []
                for i, batch in enumerate(validation_loader):
                    with torch.no_grad():
                        output = model(batch) # TODO: change this line accordingly based on your data formatting
                        y_preds += torch.argmax(output, 1).tolist()
                    y_true = batch[1].to(device) # TODO: change this line accordingly based on your data formatting
                    y_trues += y_true.tolist()
                    loss = criterion(output, y_true)
                    losses.append(loss.cpu().detach().numpy())
                mean_val_loss = np.mean(np.array(losses))
                mean_acc = accuracy_score(y_trues, y_preds)
                cm = confusion_matrix(y_trues, y_preds)
                pbar.set_description('Epoch {}: Training Loss: {:.4f}, Validation loss: {:.4f}, Acc: {:.4f}, Confusion matrix:{}'\
                                     .format(epoch+1,mean_tr_loss,mean_val_loss,mean_acc, cm))
                
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    torch.save(model.state_dict(), os.path.join(output_path, 'model_output_fold{}_epoch{}.ckpt'.format(f+1, epoch)))
                pbar.update(1)
                
                tr_losses[f,epoch] = mean_tr_loss
                val_losses[f,epoch] = mean_val_loss
                val_acc[f,epoch] = mean_acc
                val_cnf[f, :,:] += cm
        print('===================Fold '+str(f+1)+' ends===================')
    
        f+=1
        del model
    np.save(os.path.join(log_path,'train_loss.npy'), tr_losses)
    np.save(os.path.join(log_path,'val_loss.npy'), val_losses)
    np.save(os.path.join(log_path,'val_acc.npy'), val_acc)
    np.save(os.path.join(log_path,'val_cnf.npy'), val_cnf)
    return
        
if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', '-i', type=str, help='Path to the folder contains input data')
    parser.add_argument('--labels', '-l', type=str, help='Path to the dataframe file with filename and labels')    
    parser.add_argument('--model', '-m', type=str, help='Model name')
    parser.add_argument('--output_folder', '-o', default='./results', type=str, help='Output folder location')

    opt = parser.parse_args()
    
    save_dir = osp.join(opt.output_folder,time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    os.mkdir(save_dir)
    
    train(opt.input_folder, opt.labels, opt.model, save_dir)



