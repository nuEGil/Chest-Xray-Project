import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn 
import torch.optim as optim 
import torchvision

from torch.utils.tensorboard import SummaryWriter

# # classes for custom blocks will go here
class Block(nn.Module):
    def __init__(self, input_shape = [256, 256], n_kerns_in = 64, n_kerns_out=64, cc = 'enc'):
        super().__init__() # this makes sure the stuff in the keras layer init function still runs
        
        # convolutional type - I want to make this so that if i use this block in a decoder section
        # I can change one argument and have a new layer type
        convtype = {'enc' : nn.Conv2d,
                    'dec' : nn.ConvTranspose2d}

        self.conv0 = convtype[cc](n_kerns_in, n_kerns_in, kernel_size =(1, 1), 
                                  stride = 1, groups = 1, padding = 'same', bias = True,)

        self.conv1 = convtype[cc](n_kerns_in, n_kerns_in, kernel_size = (7, 7), 
                                  stride = 1, padding = 'same', bias = True,)
        
        self.act0 = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(input_shape)
        self.conv2 = convtype[cc](n_kerns_in, n_kerns_out, kernel_size = (1, 1), 
                                  stride = 1, padding = 'same', bias = True,)
            
    def forward(self, x):
        # so this is the function call 
        # define an intermediate x_ so we can sum the output later
        x_ = self.conv0(x) # 1x1 convolution - no activation
        x_ = self.conv1(x_) # ksize convolution - actiation
        x_ = x_ + x # summing block 
        x_ = self.act0(x_) # relu activation
        x_ = self.LayerNorm(x_) # layer normalization 
        x_ = self.conv2(x_) # set the number of filters on the way out. - no activation
        return x_

# Multilayer perceptron - as a custom layer -- this is normally 
# the last few layers for your classificaiton model
class MLP(nn.Module):
    def __init__(self,  neurons_in = 64, dropout_rate = 0.5, n_out = 1):
        super(MLP, self).__init__() # make sure the nn.module init funciton works

        self.neurs = [(neurons_in, neurons_in), 
                      (neurons_in,neurons_in*2),
                      (neurons_in*2, neurons_in)]
        
        # make a module list for these things -- dont have to spplit it up like this
        self.layers = nn.ModuleList() 
        self.dropout_lays = nn.ModuleList()
        
        for neur in self.neurs:
            # dense layers are also refered to as fully connected layers
            self.layers.append(nn.Linear(neur[0], neur[1]))
            self.dropout_lays.append(nn.Dropout(p=dropout_rate, inplace=False))

        # shape of object yo be normalized
        self.LayerNorm = nn.LayerNorm(neurons_in)
        
        # final layer
        self.final_lin = nn.Linear(neurons_in, n_out)
        self.output_activation = nn.Softmax(dim = 1)
    
    def forward(self, x):
        # applies the layers to x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.dropout_lays[i](x)
        x = self.LayerNorm(x)
        x = self.final_lin(x)
        x = self.output_activation(x)
        return x

# Network backbone is just a block stack
class BlockStack(nn.Module):
    # ok so this part works.. just need to keep flattening it out. 
    def __init__(self, nblocks = 9, n_kerns = 64, width_param = 2,  
                pool_rate = 2, ksize=(7, 7), activate = 'relu', 
                integrator='Add', last_activation = 'softmax', n_out=3):
        
        super(BlockStack, self).__init__() # make sure the nn.module init funciton works
        
        # create a module list
        self.layers = nn.ModuleList() 

        # remember its chanells in, channels out, then the rest
        self.layers.append(nn.Conv2d(3, n_kerns, kernel_size = (1, 1), 
                          stride = 1, padding = 'same', bias = True,))

        width = 1
        input_shape = [256, 256]
        num_pools = int((nblocks / pool_rate) - 1) # number of pools

        # itterate through the blocks
        for n in range(nblocks):
            # if we hit the end of a set of blocks, then max pool
            if (n % pool_rate == 0) and (n > 0):
                # max pool first
                
                self.layers.append(nn.MaxPool2d([2, 2]))
                # halve the input shape
                input_shape = [int(input_shape[0]/2), int(input_shape[1]/2)]
                print('input_shape ,', input_shape)  
                # remember how many kernels we want to use on subsequent layers
                width = width_param * width

                self.layers.append(Block(input_shape = input_shape, n_kerns_in = n_kerns, 
                                         n_kerns_out= width*n_kerns, cc = 'enc'))
                
                # update the number of kernels
                n_kerns = width*n_kerns
                
            else:
                # our base case is to just call a block 
                self.layers.append(Block(input_shape = input_shape, n_kerns_in = n_kerns, 
                                         n_kerns_out = n_kerns, cc = 'enc'))
        # once we are at the end. we wnat to flatten 
        self.layers.append(nn.Flatten())

        # the shape is going to be 
        flatten_shape = int(((input_shape[0]/num_pools)**2) * n_kerns)
        # this one reduces the flatten shape to something we can work with
        self.layers.append(nn.Linear(flatten_shape, 64))
        # then this one calls the MLP layer
        self.layers.append(MLP(neurons_in = 64, dropout_rate = 0.5, n_out = n_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x   

# Data loading . 
class DataLoader():
    def __init__(self, meta_data , batch_size = 5,):
        # meta data 
        self.meta_data = meta_data
        # set indices of the metadata
        self.indices = np.arange(meta_data.shape[0])
        # set the batch size 
        self.batch_size = batch_size
        # shuffle the indices
        self.shuffle_inds()
        # print('idx', self.indices.shape)
        # print('idx', self.indices)

    def get_batch(self, idx):
        # holder variables that we can write the image data to
        # torch does channels first
        X = torch.empty((self.batch_size, 3, 256, 256))
        Y = torch.empty((self.batch_size, 2))
        
        start_ = idx * self.batch_size
        stop_  = (idx + 1) * self.batch_size
        # print('start : stop = {},{}'.format(start_, stop_))
        # itterate through elements in the batch
        for io, ii in enumerate(range(start_, stop_)):
            # so there is the step that is reading in the csv
            ii_ = self.indices[ii] # get the shuffled index
            
            # put this in the training file maker . 
            fname = self.meta_data['full_paths'].iloc[ii_]
            fname = fname.replace('\\', '/')
            # print(fname)
            img_ = torchvision.io.read_image(fname )
            img_ = torchvision.transforms.Resize((256,256))(img_[0:3,...]) 
            
            X[io, ...] = 0 + img_
            if self.meta_data['num_lab'].iloc[ii_] > 0:
                Y[io, 0] =  0
                Y[io, 1] =  1
            else :
                Y[io, 0] =  1
                Y[io, 1] =  0
        
        # print(X.shape, Y.shape)   
        return X, Y

    def shuffle_inds(self):
        # shuffle indices in place
        np.random.shuffle(self.indices)

def Get_mock_data():
        file_name = '../Data_Entry_fullpath_2017.csv'
        full_data = pd.read_csv(file_name)
        negative = full_data[full_data['Finding Labels'] == 'No Finding']
        positive = full_data[full_data['Finding Labels'].str.contains("Infiltration")]
        negative['num_lab'] = np.zeros((negative.shape[0],))
        positive['num_lab'] = np.ones((positive.shape[0],))

        # mock training and testing data
        train_data = pd.concat([negative.iloc[0 : 100, :], positive.iloc[0 : 100, :]], axis = 0)
        test_data = pd.concat([negative.iloc[100 : 150, :], positive.iloc[100 : 150, :]], axis = 0)

        print('training and testing data samples \n')
        print(train_data.shape, train_data.iloc[0])
        print(test_data.shape, test_data.iloc[0])
        return train_data, test_data

## now the network trainer    
class Trainer():
    def __init__(self, args = None):
        self.args = args
        # check to see if the save location exists or not - make the file path
        # make the path something nice like this. 
        subname = self.args.model_type 
        subname = subname + '_lr-' + str(self.args.learning_rate) + '_epsi-' + str(self.args.epsilon) + '_bs-' + str(self.args.batch_size)
        subname = subname + '_blocks-' + str(self.args.blocks) + '_nfilts-' + str(self.args.n_filters)

        full_output_path = os.path.join(self.args.output_directory, subname)
        
        self.log_dir = os.path.join(full_output_path, "logs")
        self.run_dir = os.path.join(full_output_path, "runs")
        self.mod_dir = os.path.join(full_output_path, "models")

        # make the directories
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            with open(os.path.join(self.log_dir, "log.csv"), 'w') as fp:
                fp.write("")

        if not os.path.exists(self.mod_dir):
            os.makedirs(self.mod_dir)

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        
        # organize the data 
        # batch size, adn get the number of batches per file 
        self.batch_size = self.args.batch_size
        train_data, test_data = Get_mock_data()
        
        # make a data loader we can call 
        self.loader ={'training' : DataLoader(train_data), 
                      'testing'  : DataLoader(test_data)} 

        # calls 
        self.set_model()
        # self.train_one_epoch()
        self.per_epoch() # full training loop

    def set_model(self):
        print('Setting model')
        # model should come from dictionary of model types
        self.model = BlockStack(nblocks = self.args.blocks, 
                                n_kerns = self.args.n_filters, 
                                width_param = self.args.width_param,  
                                pool_rate = self.args.pool_rate, 
                                ksize=(7, 7), 
                                activate = 'relu', 
                                integrator='Add', 
                                last_activation = 'softmax', 
                                n_out = self.args.n_out)
        print(self.model)

        # TODO should save a copy of the initalized model --  

        # set the optimizer -- could be its own funciton 
        # self.optimizer ={'adam':torch.optim.Adam(params, lr=0.001, eps=1)} # with a dictionary to call other opts
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.args.learning_rate,
                                    eps = self.args.epsilon)

        # set the loss 
        # TODO put this into a dicitonary 
        self.loss_fn = nn.BCELoss()
        
    def train_one_epoch(self, epoch_index, tb_writer): # epoch_idx, tb_writer):
        running_loss = 0.
        last_loss = 0.
        
        n_batches = self.loader['training'].meta_data.shape[0] // self.batch_size

        # shuffle indices
        self.loader['training'].shuffle_inds()
        # need the number of batches 
        print('Training', '-'*8)
        print('train data shape ', self.loader['training'].meta_data.shape)
        print('n_batches ', n_batches)
        
        for i in range(n_batches):
            # zero gradients for every batch
            self.optimizer.zero_grad()

            # get a batch
            x_bat, y_true = self.loader['training'].get_batch(i)

            # Make predictions for this batch
            y_pred = self.model(x_bat)
            
            # Compute loss and gradient
            loss = self.loss_fn(y_pred, y_true)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            running_loss += loss.item()
            print('running_loss:{}'.format(running_loss))
            
            
        last_loss = running_loss / n_batches # loss per batch
        print('  batch {} loss: {}'.format(i+1, last_loss))
        tb_x = epoch_index * self.loader['training'].meta_data.shape[0] + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        running_loss = 0.

        return last_loss

    def per_epoch(self):
        # timestamp 
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # where the writer should save to
        patho = os.path.join(self.run_dir, 'xray_trainer_{}'.format(timestamp))
        writer = SummaryWriter(patho)
        
        epoch_number = 0

        EPOCHS = self.args.epochs

        best_vloss = 1_000_000.

        n_batches = self.loader['testing'].meta_data.shape[0] // self.batch_size
        
        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer)


            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i in range(n_batches):
                    x_bat, y_true = self.loader['testing'].get_batch(i)
                    y_pred = self.model(x_bat)

                    vloss = self.loss_fn(y_pred, y_true)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                model_path = os.path.join(self.mod_dir, model_path)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

# this function is so I can bug check the other layers. 
def layer_tester():
    # # test mlp layer
    # model = MLP(neurons_in = 64, dropout_rate = 0.5, n_out = 64)
    # print(model) # print the model
    # x = torch.rand(10, 64) #batch x vector length
    # y = model(x)
    # print(y.shape)

    # # residual block
    # model = Block(input_shape = [256, 256], n_kerns_in = 64, n_kerns_out = 64, cc = 'enc')
    # print(model)
    # # torch does  -- minibatch, channels, height, width
    # x = torch.rand(10, 64, 256, 256) #batch x vector length
    # y = model(x)
    # print(y.shape)

    ## test out the data loader in combination with a block stack 
    model = BlockStack(nblocks = 4, n_kerns = 64, width_param = 2,  
                        pool_rate = 2, ksize=(7, 7), activate = 'relu', 
                        integrator='Add', last_activation = 'softmax', n_out = 1)
    print(model)
    # torch does  -- minibatch, channels, height, width
    x = torch.rand(10, 3, 256, 256) #batch x vector length
    y = model(x)
    print(y.shape)

    # next lets test the data loader. 
    train_data, test_data = Get_mock_data()    
    # initialize the dataloder
    dd = DataLoader(train_data) 
    x_bat, y_bat = dd.get_batch(0) # input is batch number
    
    # pass batch to model 
    y = model(x_bat)
    print('output', y.shape)
    print(y)


# only runs if we call this file. 
if __name__ == '__main__':
    # layer_tester()
    Trainer()