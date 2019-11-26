# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 
Getting to Know GAN Better
@author: Ajay Solanki
"""
 
#Imports
import glob
import os
import time

import numpy as np
import scipy.io as io
import scipy.ndimage as nd
import tensorflow as tf
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class DGan:
    
    def get3DImages(self,data_dir):
        all_files = np.random.choice(glob.glob(data_dir), size=32)
        # all_files = glob.glob(data_dir)
        all_volumes = np.asarray([self.getVoxelsFromMat(f) for f in all_files], dtype=np.bool)
        return all_volumes

    def Initialize(self):
        self.z_size = 200
        self.gen_filters = [512,256,128,64,1]
        self.gen_kernel_size = [4,4,4,4,4]
        self.gen_strides = [1,2,2,2,2]
        self.gen_input_shape =(1,1,1,self.z_size)
        self.gen_activations = ['relu','relu','relu','relu','sigmoid']
        self.gen_convolution_blocks = 5
        
        self.dis_input_shape = (64, 64, 64, 1)
        self.dis_filters = [64, 128, 256, 512, 1]
        self.dis_kernel_sizes = [4, 4, 4, 4, 4]
        self.dis_strides = [2, 2, 2, 2, 1]
        self.dis_paddings = ['same', 'same', 'same', 'same', 'valid']
        self.dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid']
        self.dis_convolutional_blocks = 5
        self.home_directory = cwd = os.getcwd()
        self.log_dir = os.path.join(self.home_directory,"logs")
        

    def getVoxelsFromMat(self, filename,cube_len=64):
        # 2 things we need to pad each axes to reflect same number of elements at each axis
        # 3D Image needs to be converted to 64 X 64 X 64 for CNN consumption
        voxels = io.loadmat(filename)['instance']
        #padded to the edges of each axes, the mode values (constant), and the constant_values
        voxels = np.pad(voxels, (1,1), 'constant', constant_values=(0,0))
        if (cube_len != 32 and cube_len == 64):
            # zoom() function from the scipy.ndimage module to convert the 3D image to a 3D image with dimensions of 64x64x64. 
            voxels = nd.zoom(voxels, (2,2,2) , mode='constant', order=0)
        return (voxels)
    

    
    def load_data(self):
        self.Initialize()
        object_name = "airplane" # Change the object of your choice found in the volumetric_data directory
        data_dir = "E:\\workdirectory\\Code Name Val Halen\\DS Sup\\DL\\GAN - Projects\\9781789136678_Code\\Chapter02\\3DShapeNets\\volumetric_data" \
               "\\{}\\30\\train\\*.mat".format(object_name) # Change the location to file on your device;
        print(data_dir)
        volumes = self.get3DImages(data_dir=data_dir)
        self.volumes = volumes[..., np.newaxis].astype(np.float)
        print("No of volumes")
        print(len(self.volumes))
        print(self.volumes.shape[0])
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
        self.Create_Generator()
        self.Create_Discriminator()
        tensorboard.set_model(self.gen_model)
        tensorboard.set_model(self.dis_model)
        self.tensorboard = tensorboard
        
    def Create_Generator(self):
        print(self.gen_input_shape)
        input_layer = Input(shape = self.gen_input_shape)
        a = Deconv3D(filters = self.gen_filters[0],
                     kernel_size = self.gen_kernel_size[0],
                     strides = self.gen_strides[0])(input_layer)
        a = BatchNormalization()(a, training= True)
        a = Activation(activation=self.gen_activations[0])(a)
        
        for i in range(self.gen_convolution_blocks -1):
            a = Deconv3D(filters = self.gen_filters[i + 1],
                     kernel_size = self.gen_kernel_size[i + 1],
                     strides = self.gen_strides[i + 1], padding='same')(a)
            a = BatchNormalization()(a, training= True)
            a = Activation(activation=self.gen_activations[i + 1])(a)
        
        self.gen_model  = Model(inputs=input_layer, outputs=a)
        print(self.gen_model.summary())
        return(self.gen_model)
    
    def Create_Discriminator(self):
        # Input Layer
        dis_input_layer = Input(shape=self.dis_input_shape)
        # 3 Convolution Layer with a pre specified set filters
        a = Conv3D(filters=self.dis_filters[0],
           kernel_size=self.dis_kernel_sizes[0],
           strides=self.dis_strides[0],
           padding=self.dis_paddings[0])(dis_input_layer)
        a = BatchNormalization()(a, training=True)
        # Leakly Rely
        a = LeakyReLU(self.dis_alphas[0])(a)
        for i in range(self.dis_convolutional_blocks - 1):
            a = Conv3D(filters=self.dis_filters[i + 1],
               kernel_size=self.dis_kernel_sizes[i + 1],
               strides=self.dis_strides[i + 1],
               padding=self.dis_paddings[i + 1])(a)
            a = BatchNormalization()(a, training=True)
            if self.dis_activations[i + 1] == 'leaky_relu':
                a = LeakyReLU(self.dis_alphas[i + 1])(a)
                #a = LeakyReLU(dis_alphas[i + 1])(a)
            elif self.dis_activations[i + 1] == 'sigmoid':
                a = Activation(activation='sigmoid')(a) 
                
        self.dis_model = Model(inputs=dis_input_layer, outputs=a)
        print(self.dis_model.summary())
        return (self.dis_model)
    
    def Execute(self):
        # Training the GAN is similar to training a vanilla GAN.
        # Train the discriminator network on both generated and real images freeze the generator
        # Train the generator freeze the discriminator
        # Repeat the process for specified number of epochs.
        # During one iteration we train both network in a sequence.
        
        gen_learning_rate = 0.0025 
        dis_learning_rate = 0.00001
        beta = 0.5
        batch_size = 12
        z_size = 200
        DIR_PATH = ""
        generated_volumes_dir = ""
        epochs = 25
        
        
        #create instances
        #self.Create_Generator()
        #self.Create_Discriminator()
        
        # Specify optimizer
        gen_optimizer = Adam(lr=gen_learning_rate,beta_1 = beta)
        dis_optimizer = Adam(lr=dis_learning_rate,beta_1 = beta)
        
        #Compile Networks
        self.gen_model.compile(loss="binary_crossentropy", optimizer =gen_optimizer)
        self.dis_model.compile(loss="binary_crossentropy", optimizer =dis_optimizer)
        
        # Create the adversarial mode
        self.dis_model.trainable = False
        adversarial_model = Sequential()
        adversarial_model.add(self.gen_model)
        adversarial_model.add(self.dis_model)
        adversarial_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=gen_learning_rate, beta_1=beta))
        self.adversarial_model = adversarial_model
        for epoch in range(epochs):
            print("Epoch:", epoch)
            gen_losses = []
            dis_losses = []
            number_of_batches = int(self.volumes.shape[0] / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)
                
                # Sample a batch of images from the set of real images and a batch of noise vectors
                # from a Gaussian Distribution, Shape of the noise vector is (1,1,1,200)
                z_sample = np.random.normal(0, 0.33, size =[batch_size,1,1,1, z_size]).astype(np.float32)
                volumes_batch = self.volumes[index * batch_size:(index + 1) * batch_size,:,:,:]
                
                #Generate fake Images
                gen_volumes = self.gen_model.predict(z_sample,verbose = 3)
                
                # Train the discriminator
                self.dis_model.trainable = True 
                
                #Create fake and real labels
                labels_real = np.reshape([1] * batch_size, (-1,1,1,1,1))
                labels_fake = np.reshape([1] * batch_size, (-1,1,1,1,1))
                
                #Train the discriminator
                loss_real = self.dis_model.train_on_batch(volumes_batch, labels_real)
                loss_fake = self.dis_model.train_on_batch(gen_volumes, labels_fake)
                
                # Calculate the total discrimnator loss
                d_loss = 0.5 * (loss_real + loss_fake)
                discriminator.trainable = False
                
                z = np.random.normal(0, 0.33, size =[batch_size,1,1,1, z_size]).astype(np.float32)
                #Train the adverserial network
                g_loss = self.adversarial_model.train_on_batch(z, np.reshape([1] * batch_size, (-1, 1, 1, 1, 1)))
                gen_losses.append(g_loss)
                dis_losses.append(d_loss)
                if index % 10 == 0:
                    z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                    generated_volumes = self.gen_model.predict(z_sample2, verbose=3)
                    for i, generated_volume in enumerate(generated_volumes[:5]):
                        voxels = np.squeeze(generated_volume)
                        voxels[voxels < 0.5] = 0.
                        voxels[voxels >= 0.5] = 1.
                        self.saveFromVoxels(voxels, "results/img_{}_{}_{}".format(epoch, index, i))
                
                
                self.write_log(self.tensorboard, 'g_loss', np.mean(gen_losses), epoch)
                self.write_log(self.tensorboard, 'd_loss', np.mean(dis_losses), epoch)
            
            
            
            self.gen_model.save_weights(os.path.join(generated_volumes_dir, "generator_weights.h5"))
            self.dis_model.save_weights(os.path.join(generated_volumes_dir, "discriminator_weights.h5"))

        
    def write_log(self,callback, name, value, batch_no):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
            
    def saveFromVoxels(self,voxels, path):
        z, x, y = voxels.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, -z, zdir='z', c='red')
        plt.savefig(path)

    def plotAndSaveVoxel(file_path, voxel):
        """
        Plot a voxel
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        ax.voxels(voxel, edgecolor="red")
        # plt.show()
        plt.savefig(file_path)
        
gan = DGan()
gan.load_data()
gan.Execute()