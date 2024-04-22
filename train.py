import logging
import os
import os.path as osp
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import json 
import random

import torch
from torch.optim import SGD, Adadelta, AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

from data_utils import MyGraphDataset, CustomImbalancedDatasetSampler
from models import GCNnet, GINnet, GATnet, MPNNnet, GENnet
from utils import FocalLoss

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

def setup_logging(log_path):
    # Create a custom logger
    logger = logging.getLogger('POG')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of log messages

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(os.path.join(log_path, 'training_log.log'), mode='w')
    file_handler.setLevel(logging.DEBUG)  # File handler captures all debug and higher level messages

    # Create stream handler for logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)  # Console handler captures all debug and higher level messages

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Log a startup message
    logger.info('Logging setup complete - the log file is now active!')

    return logger

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('./params.json') as f:
        opt = json.load(f)
        
    output_path = os.path.join(save_dir, 'ckpts')
    log_path = os.path.join(save_dir, 'logs')
    config_path = os.path.join(save_dir, 'configs')
    os.mkdir(output_path)
    os.mkdir(log_path)
    os.mkdir(config_path)
    logger = setup_logging(log_path)
    d = opt
    logger.info('Current device: ' + str(device))
    logger.info(json.dumps(d, indent=4))
    with open(os.path.join(config_path, 'params.json'), 'w') as f:
        json.dump(d, f)
    
    if opt['manual_seed'] is None:
        opt['manual_seed'] = random.randint(1, 10000)
        logger.info('Random Seed: ', opt['manual_seed'])
        random.seed(opt['manual_seed'])
        torch.manual_seed(opt['manual_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opt['manual_seed'])

    #model settings
    #load model
    model = create_model(model_name, opt)
    model = model.to(device)
    logger.info(model)
    # --------------------------------------------------------------------------
    #loss --> change if need to
    criterion = FocalLoss().to(device)
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
    
    # Compute label frequencies
    label_counts = df['class'].value_counts().to_dict()  # Adjust 'label_column_name' to your DataFrame's actual label column
    label_weights = {label: 1.0 / count for label, count in label_counts.items()}
    logger.info(label_weights)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt['manual_seed'])
    f = 0
    dataset = MyGraphDataset(data_path, df, device=device)
    
    tr_losses = np.zeros((Nfold, opt['num_epochs']))
    val_losses = np.zeros((Nfold, opt['num_epochs']))
    val_acc = np.zeros((Nfold, opt['num_epochs']))
    val_cnf = np.zeros((Nfold, opt['output_dim'], opt['output_dim']))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(dataset, df['class'])):
        logger.info('===================Fold {} starts==================='.format(f+1))
        
        training_set = Subset(dataset, tr_idx)
        validation_set = Subset(dataset, val_idx)
        
        train_subsampler = CustomImbalancedDatasetSampler(training_set, logger, label_weights, device=device)
        training_loader = DataLoader(training_set, batch_size=opt['batch_size'], sampler=train_subsampler, collate_fn=MyGraphDataset.my_collate_fn)
        
        # training_loader = DataLoader(training_set, batch_size=opt['batch_size'], collate_fn=MyGraphDataset.my_collate_fn)
        validation_loader = DataLoader(validation_set, batch_size=opt['batch_size'], collate_fn=MyGraphDataset.my_collate_fn)
        
        model.reset_parameters()
        best_val_loss = 1e6
        with tqdm(total=opt['num_epochs'], desc=f"Fold {f+1}", leave=True) as pbar:
            for epoch in range(opt['num_epochs']):
                model.train()
                losses = []
                y_preds = []
                y_trues = []
                for i, batch in enumerate(training_loader):          
                    optimizer.zero_grad()
                    output = model(batch) # TODO: change this line accordingly based on your data formatting
                    y_preds += torch.argmax(output, 1).tolist()
                    y_true = batch.y.squeeze() # TODO: change this line accordingly based on your data formatting
                    y_trues += y_true.tolist()
                    loss = criterion(output, y_true)
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.cpu().detach().numpy())
                    
                mean_tr_loss = np.mean(np.array(losses))
                trcm = confusion_matrix(y_trues, y_preds)
                
                model.eval()
                losses = []
                y_preds = []
                y_trues = []
                for i, batch in enumerate(validation_loader):
                    with torch.no_grad():
                        output = model(batch) # TODO: change this line accordingly based on your data formatting
                        y_preds += torch.argmax(output, 1).tolist()
                    y_true = batch.y.squeeze() # TODO: change this line accordingly based on your data formatting
                    y_trues += y_true.tolist()
                    loss = criterion(output, y_true)
                    losses.append(loss.cpu().detach().numpy())
                mean_val_loss = np.mean(np.array(losses))
                mean_acc = balanced_accuracy_score(y_trues, y_preds)
                cm = confusion_matrix(y_trues, y_preds)
                
                log_msg = (f"Epoch {epoch+1}: Training Loss: {mean_tr_loss:.4f}, "
                           f"Validation Loss: {mean_val_loss:.4f}, "
                           f"Validation Accuracy: {mean_acc:.4f}")
                logger.info(log_msg)
                logger.info(f'Epoch {epoch + 1} Training Confusion Matrix:\n{trcm}')
                logger.info(f'Epoch {epoch + 1} Validation Confusion Matrix:\n{cm}')
                
                pbar.set_description(log_msg)
                pbar.update()
                
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    torch.save(model.state_dict(), os.path.join(output_path, 'model_output_fold{}_epoch{}.ckpt'.format(f+1, epoch)))
                # pbar.update(1)
                
                tr_losses[f,epoch] = mean_tr_loss
                val_losses[f,epoch] = mean_val_loss
                val_acc[f,epoch] = mean_acc
                val_cnf[f, :,:] += cm
        logger.info('===================Fold '+str(f+1)+' ends===================')
    
        f+=1
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



