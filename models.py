"""
Author: Limeng Pu
Date: 2023-07-26
Description: Integrated Graph Neural Network (GNN) Models using PyTorch Geometric.
"""

import torch
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, GINConv, NNConv, GENConv, JumpingKnowledge
from torch_geometric.nn import Set2Set
from torch_geometric.nn.pool import global_add_pool, global_mean_pool, global_max_pool

class GCNnet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_conv_layers, hidden_dim, batch_norm,
                 jumping_knowledge, global_pool, process_step, dense_dims, dropout):
        
        super(GCNnet, self).__init__()
        
        """
        GCN model initialization
        params: input_dim: input node feature dimension, int
                output_dim: number of classes for classification; 1 for regression
                num_conv_layers: number of convolutional layers, int
                hidden_dim: hidden dimension for convolutional layers, int
                batch_norm: use batch normalization, bool
                jumping_knowledge: use JumpingKnowledge, bool
                global_pool: name of global pooling module, string
                process_step: int if global pool is set2set
                dense_dims: fully connected layers' hidden dimensions (not including final FC layer), list
        """
        
        # Init conv layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        self.dense_dims = dense_dims
        
        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            conv = GCNConv(in_channels=in_channels, out_channels=hidden_dim)
            self.convs.append(conv)
            if batch_norm:
                self.batch_norms.append(BatchNorm1d(num_features=hidden_dim))
                
        # Init JK module
        self.jumping_knowledge = jumping_knowledge
        if jumping_knowledge:
            if jumping_knowledge.isdigit():
                jk_layer = int(jumping_knowledge)
                self.jump = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=jk_layer)
                jk_out = hidden_dim
            elif jumping_knowledge == 'cat':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim)
                jk_out = hidden_dim*num_conv_layers
            elif jumping_knowledge == 'max':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim)
                jk_out = hidden_dim
            else:
                raise('Unrecognized Jumping Knowledge module information, please check input!')
        
        # Init global pooling layers
        self.global_pool = global_pool
        global_pool = global_pool.lower()
        if global_pool == 'add':
            self.gpl = global_add_pool
        elif global_pool == 'mean':
            self.gpl = global_mean_pool
        elif global_pool == 'max':
            self.gpl = global_max_pool
        elif global_pool == 'set2set':
            self.gpl = Set2Set(jk_out, processing_steps=process_step)
        else:
            raise('Unrecognized global pooling strategy, please check input!')
        
        # Init dense layers
        self.dense_layers =torch.nn.ModuleList()
        for j in range(len(self.dense_dims)):
            if j == 0:
                if global_pool == 'set2set':
                    fc = Linear(in_features=2*hidden_dim, out_features=dense_dims[j])
                    self.dense_layers.append(fc)
                else:
                    fc = Linear(in_features=hidden_dim, out_features=dense_dims[j])
            else:
                fc = Linear(in_features=dense_dims[j-1], out_features=dense_dims[j])
                self.dense_layers.append(fc)
        self.final_dense = Linear(in_features=dense_dims[j], out_features=output_dim)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jumping_knowledge:
            self.jump.reset_parameters()
        if self.global_pool == 'set2set':
            self.gpl.reset_parameters()
        for dense in self.dense_layers:
            dense.reset_parameters()
        self.final_dense.reset_parameters()
    
    def forward(self, data):
        x, edge_index, _, batch = data.x, data.edge_index, data.edge_attr, data.batch        
        xs = []

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if len(self.batch_norms) > 0:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            xs += [x]
            
        if self.jumping_knowledge:
            x1 = self.jump(xs)
        else:
            x1 = xs[-1]
        x1 = self.gpl(x1, batch)
        for dense_layer in self.dense_layers:
            x1 = F.relu(dense_layer(x1))
            if self.dropout > 0.0:
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
        logits = self.final_dense(x1)
        
        return logits

class GATnet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_conv_layers, hidden_dim, num_heads, batch_norm,
                 jumping_knowledge, global_pool, process_step, dense_dims, dropout):
        
        super(GATnet, self).__init__()
        
        """
        GAT model initialization
        params: input_dim: input node feature dimension, int
                output_dim: number of classes for classification; 1 for regression
                num_conv_layers: number of convolutional layers, int
                hidden_dim: hidden dimension for convolutional layers, int
                num_heads: number of attention head, int
                batch_norm: use batch normalization, bool
                jumping_knowledge: use JumpingKnowledge, bool
                global_pool: name of global pooling module, string
                process_step: int if global pool is set2set
                dense_dims: fully connected layers' hidden dimensions (not including final FC layer), list
        """
        
        # Init conv layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        self.dense_dims = dense_dims

        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else num_heads*hidden_dim
            conv = GATConv(in_channels=in_channels, out_channels=hidden_dim, 
                           heads=num_heads, concat=True,
                           dropout=.2)
            self.convs.append(conv)
            if batch_norm:
                self.batch_norms.append(BatchNorm1d(num_features=hidden_dim))
        
        # Init JK module
        self.jumping_knowledge = jumping_knowledge
        if jumping_knowledge:
            if jumping_knowledge.isdigit():
                jk_layer = int(jumping_knowledge)
                self.jump = JumpingKnowledge(mode='lstm', channels=num_heads*hidden_dim, num_layers=jk_layer)
                jk_out = num_heads*hidden_dim
            elif jumping_knowledge == 'cat':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=num_heads*hidden_dim)
                jk_out = num_heads*hidden_dim*num_conv_layers
            elif jumping_knowledge == 'max':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=num_heads*hidden_dim)
                jk_out = num_heads*hidden_dim
            else:
                raise('Unrecognized Jumping Knowledge module information, please check input!')
        
        # Init global pooling layers
        self.global_pool = global_pool
        global_pool = global_pool.lower()
        if global_pool == 'add':
            self.gpl = global_add_pool
        elif global_pool == 'mean':
            self.gpl = global_mean_pool
        elif global_pool == 'max':
            self.gpl = global_max_pool
        elif global_pool == 'set2set':
            self.gpl = Set2Set(jk_out, processing_steps=process_step)
        else:
            raise('Unrecognized global pooling strategy, please check input!')
        
        # Init dense layers
        self.dense_layers =torch.nn.ModuleList()
        for j in range(len(self.dense_dims)):
            if j == 0:
                if global_pool == 'set2set':
                    fc = Linear(in_features=2*num_heads*hidden_dim, out_features=dense_dims[j])
                    self.dense_layers.append(fc)
                else:
                    fc = Linear(in_features=num_heads*hidden_dim, out_features=dense_dims[j])
            else:
                fc = Linear(in_features=dense_dims[j-1], out_features=dense_dims[j])
                self.dense_layers.append(fc)
        self.final_dense = Linear(in_features=dense_dims[j], out_features=output_dim)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jumping_knowledge:
            self.jump.reset_parameters()
        if self.global_pool == 'set2set':
            self.gpl.reset_parameters()
        for dense in self.dense_layers:
            dense.reset_parameters()
        self.final_dense.reset_parameters()
    
    def forward(self, data):
        x, edge_index, _, batch = data.x, data.edge_index, data.edge_attr, data.batch        
        xs = []

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if len(self.batch_norms) > 0:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            xs += [x]
        
        if self.jumping_knowledge:
            x1 = self.jump(xs)
        else:
            x1 = xs[-1]
        x1 = self.gpl(x1, batch)
        for dense_layer in self.dense_layers:
            x1 = F.relu(dense_layer(x1))
            if self.dropout > 0.0:
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
        logits = self.final_dense(x1)
        
        return logits

class GINnet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_conv_layers, hidden_dim, batch_norm,
                 jumping_knowledge, global_pool, process_step, dense_dims, dropout):
        
        super(GINnet, self).__init__()
        
        """
        GIN model initialization
        params: input_dim: input node feature dimension, int
                output_dim: number of classes for classification; 1 for regression
                num_conv_layers: number of convolutional layers, int
                hidden_dim: hidden dimension for convolutional layers, int
                batch_norm: use batch normalization, bool
                jumping_knowledge: use JumpingKnowledge, bool
                global_pool: name of global pooling module, string
                process_step: int if global pool is set2set
                dense_dims: fully connected layers' hidden dimensions (not including final FC layer), list
        """
        
        # Init conv layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        self.dense_dims = dense_dims

        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            conv = GINConv(Sequential(Linear(in_channels, hidden_dim),
                                      ReLU(),
                                      Linear(hidden_dim, hidden_dim),
                                      ReLU(),
                                      BatchNorm1d(hidden_dim)),\
                           train_eps=True)
            self.convs.append(conv)
            if batch_norm:
                self.batch_norms.append(BatchNorm1d(num_features=hidden_dim))

        # Init JK module
        self.jumping_knowledge = jumping_knowledge
        if jumping_knowledge:
            if jumping_knowledge.isdigit():
                jk_layer = int(jumping_knowledge)
                self.jump = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=jk_layer)
                jk_out = hidden_dim
            elif jumping_knowledge == 'cat':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim)
                jk_out = hidden_dim*num_conv_layers
            elif jumping_knowledge == 'max':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim)
                jk_out = hidden_dim
            else:
                raise('Unrecognized Jumping Knowledge module information, please check input!')
        
        # Init global pooling layers
        self.global_pool = global_pool
        global_pool = global_pool.lower()
        if global_pool == 'add':
            self.gpl = global_add_pool
        elif global_pool == 'mean':
            self.gpl = global_mean_pool
        elif global_pool == 'max':
            self.gpl = global_max_pool
        elif global_pool == 'set2set':
            self.gpl = Set2Set(jk_out, processing_steps=process_step)
        else:
            raise('Unrecognized global pooling strategy, please check input!')
        
        # Init dense layers
        self.dense_layers =torch.nn.ModuleList()
        for j in range(len(self.dense_dims)):
            if j == 0:
                if global_pool == 'set2set':
                    fc = Linear(in_features= 2*hidden_dim, out_features=dense_dims[j])
                    self.dense_layers.append(fc)
                else:
                    fc = Linear(in_features= hidden_dim, out_features=dense_dims[j])
            else:
                fc = Linear(in_features=dense_dims[j-1], out_features=dense_dims[j])
                self.dense_layers.append(fc)
        self.final_dense = Linear(in_features=dense_dims[j], out_features=output_dim)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jumping_knowledge:
            self.jump.reset_parameters()
        if self.global_pool == 'set2set':
            self.gpl.reset_parameters()
        for dense in self.dense_layers:
            dense.reset_parameters()
        self.final_dense.reset_parameters()
    
    def forward(self, data):
        x, edge_index, _, batch = data.x, data.edge_index, data.edge_attr, data.batch        
        xs = []

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if len(self.batch_norms) > 0:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            xs += [x]
        
        if self.jumping_knowledge:
            x1 = self.jump(xs)
        else:
            x1 = xs[-1]
        x1 = self.gpl(x1, batch)
        for dense_layer in range(self.dense_layers):
            x1 = F.relu(dense_layer(x1))
            if self.dropout > 0.0:
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
        logits = self.final_dense(x1)
        
        return logits

class MPNNnet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_conv_layers, edge_input_dim, edge_hidden_dim,
                 hidden_dim, batch_norm, aggr,
                 jumping_knowledge, global_pool, process_step, dense_dims, dropout):
        
        super(MPNNnet, self).__init__()
        
        """
        MPNN model initialization
        params: input_dim: input node feature dimension, int
                output_dim: number of classes for classification; 1 for regression
                num_conv_layers: number of convolutional layers, int
                edge_input_dim: input edge feature dimension, int
                edge_hidden_dim: hidden dimension for edge features, int
                hidden_dim: hidden dimension for convolutional layers, int
                batch_norm: use batch normalization, bool
                aggr: aggregator, string ('add', 'mean', 'max')
                jumping_knowledge: use JumpingKnowledge, bool
                global_pool: name of global pooling module, string
                process_step: int if global pool is set2set
                dense_dims: fully connected layers' hidden dimensions (not including final FC layer), list
        """

        # Init conv layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        self.dense_dims = dense_dims

        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            edge_in_channels = edge_input_dim if i == 0 else edge_hidden_dim
            conv = NNConv(in_channels=in_channels, out_channels=hidden_dim,
                          nn=Sequential(Linear(edge_in_channels, edge_hidden_dim),
                                        ReLU(),
                                        Linear(hidden_dim, in_channels*edge_hidden_dim)),
                          aggr=aggr)
            self.convs.append(conv)
            if batch_norm:
                self.batch_norms.append(BatchNorm1d(num_features=hidden_dim))
                                
        # Init JK module
        self.jumping_knowledge = jumping_knowledge
        if jumping_knowledge:
            if jumping_knowledge.isdigit():
                jk_layer = int(jumping_knowledge)
                self.jump = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=jk_layer)
                jk_out = hidden_dim
            elif jumping_knowledge == 'cat':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim)
                jk_out = hidden_dim*num_conv_layers
            elif jumping_knowledge == 'max':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim)
                jk_out = hidden_dim
            else:
                raise('Unrecognized Jumping Knowledge module information, please check input!')
        
        # Init global pooling layers
        self.global_pool = global_pool
        global_pool = global_pool.lower()
        if global_pool == 'add':
            self.gpl = global_add_pool
        elif global_pool == 'mean':
            self.gpl = global_mean_pool
        elif global_pool == 'max':
            self.gpl = global_max_pool
        elif global_pool == 'set2set':
            self.gpl = Set2Set(jk_out, processing_steps=process_step)
        else:
            raise('Unrecognized global pooling strategy, please check input!')
        
        # Init dense layers
        self.dense_layers =torch.nn.ModuleList()
        for j in range(len(self.dense_dims)):
            if j == 0:
                if global_pool == 'set2set':
                    fc = Linear(in_features= 2*hidden_dim, out_features=dense_dims[j])
                    self.dense_layers.append(fc)
                else:
                    fc = Linear(in_features= hidden_dim, out_features=dense_dims[j])
            else:
                fc = Linear(in_features=dense_dims[j-1], out_features=dense_dims[j])
                self.dense_layers.append(fc)
        self.final_dense = Linear(in_features=dense_dims[j], out_features=output_dim)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jumping_knowledge:
            self.jump.reset_parameters()
        if self.global_pool == 'set2set':
            self.gpl.reset_parameters()
        for dense in self.dense_layers:
            dense.reset_parameters()
        self.final_dense.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch        
        xs = []

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            if len(self.batch_norms) > 0:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            xs += [x]
        
        if self.jumping_knowledge:
            x1 = self.jump(xs)
        else:
            x1 = xs[-1]
        x1 = self.gpl(x1, batch)
        for dense_layer in self.dense_layers:
            x1 = F.relu(dense_layer(x1))
            if self.dropout > 0.0:
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
        logits = self.final_dense(x1)
        
        return logits

class GENnet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_conv_layers, edge_input_dim, embed_dim,
                 edge_hidden_dim, hidden_dim, batch_norm, aggr, learn, msg_norm, mlp_layers,
                 jumping_knowledge, global_pool, process_step, dense_dims, dropout):
        
        super(GENnet, self).__init__()
        
        """
        GEN model initialization
        params: input_dim: input node feature dimension, int
                output_dim: number of classes for classification; 1 for regression
                num_conv_layers: number of convolutional layers, int
                edge_input_dim: input edge feature dimension, int
                embed_dim: preprocessing embedding dim for node and edge features, int
                edge_hidden_dim: hidden dimension for edge features, int
                hidden_dim: hidden dimension for convolutional layers, int
                batch_norm: use batch normalization, bool
                aggr: aggregator, string ('add', 'mean', 'max')
                learn:
                msg_norm:
                mlp_layers:
                jumping_knowledge: use JumpingKnowledge, bool
                global_pool: name of global pooling module, string
                process_step: int if global pool is set2set
                dense_dims: fully connected layers' hidden dimensions (not including final FC layer), list
        """
        
        # Init conv layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        
        self.node_encoder = Linear(input_dim, embed_dim)
        self.edge_encoder = Linear(edge_input_dim, embed_dim)
        
        for i in range(num_conv_layers):
            conv = GENConv(in_channels=embed_dim, out_channels=hidden_dim,
                           aggr=aggr, learn_t=learn, learn_p=learn,
                           msg_norm=msg_norm, num_layers=mlp_layers)
            
            self.convs.append(conv)
            if batch_norm:
                self.batch_norms.append(BatchNorm1d(num_features=hidden_dim))
    
        # Init JK module
        self.jumping_knowledge = jumping_knowledge
        if jumping_knowledge:
            if jumping_knowledge.isdigit():
                jk_layer = int(jumping_knowledge)
                self.jump = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=jk_layer)
                jk_out = hidden_dim
            elif jumping_knowledge == 'cat':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim)
                jk_out = hidden_dim*num_conv_layers
            elif jumping_knowledge == 'max':
                self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim)
                jk_out = hidden_dim
            else:
                raise('Unrecognized Jumping Knowledge module information, please check input!')
        
        # Init global pooling layers
        self.global_pool = global_pool
        global_pool = global_pool.lower()
        if global_pool == 'add':
            self.gpl = global_add_pool
        elif global_pool == 'mean':
            self.gpl = global_mean_pool
        elif global_pool == 'max':
            self.gpl = global_max_pool
        elif global_pool == 'set2set':
            self.gpl = Set2Set(jk_out, processing_steps=process_step)
        else:
            raise('Unrecognized global pooling strategy, please check input!')
        
        # Init dense layers
        self.dense_layers =torch.nn.ModuleList()
        for j in range(len(self.dense_dims)):
            if j == 0:
                if global_pool == 'set2set':
                    fc = Linear(in_features= 2*hidden_dim, out_features=dense_dims[j])
                    self.dense_layers.append(fc)
                else:
                    fc = Linear(in_features= hidden_dim, out_features=dense_dims[j])
            else:
                fc = Linear(in_features=dense_dims[j-1], out_features=dense_dims[j])
                self.dense_layers.append(fc)
        self.final_dense = Linear(in_features=dense_dims[j], out_features=output_dim)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jumping_knowledge:
            self.jump.reset_parameters()
        if self.global_pool == 'set2set':
            self.gpl.reset_parameters()
        for dense in self.dense_layers:
            dense.reset_parameters()
        self.final_dense.reset_parameters()
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        xs = []
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
            if len(self.batch_norms) > 0:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            xs += [x]
        
        if self.jumping_knowledge:
            x1 = self.jump(xs)
        else:
            x1 = xs[-1]
        x1 = self.gpl(x1, batch)
        for dense_layer in self.dense_layers:
            x1 = F.relu(dense_layer(x1))
            if self.dropout > 0.0:
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
        logits = self.final_dense(x1)

        return logits