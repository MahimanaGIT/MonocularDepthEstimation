'''
Things to try:
1. Change deconvolution to upconvolution

2. Do stereo i.e. input of network is combination of left and right image
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import keras
from tensorflow.keras import Sequential
import numpy as np

from losses import Loss

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class MonocularDepthModel(object):
    left  = None
    right = None

    def __init__(self):
        self.build_model()
        self.build_outputs()
        self.build_losses()
        
    @staticmethod
    def conv2d(self, x, num_out_channel, kernel_size, stride, activation_fn = tf.nn.elu):
        '''
        2D Convolution Padded Layer: 
        Arguments:
        num_out_channel 
        kernel_size
        strides

        Constant Padding is applied on the input image and convolution is applied on padded image
        '''
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        padded_image = tf.pad(x, [[0, 0],[p, p],[p, p],[0, 0]])     # Padding using 4 * 2 sequence
        slim.conv2d(padded_image, num_out_channel, kernel_size, stride, 'VALID', activation_fn = activation_fn)

    @staticmethod
    def max_pool_2d(self, x, kernel_size):
        '''
        First Constant Padding is applied and then max pooling is applied on the padded image
        '''
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        padded_image = tf.pad(x, [[0, 0],[p, p],[p, p],[0, 0]])     # Padding using 4 * 2 sequence
        return slim.maxpool2d(padded_image, kernel_size)

    @staticmethod
    def conv2d_transpose(self, x, num_out_channel, kernel_size, scale):
        '''
        Alternative for upsampling the image to match the output dimensions
        '''
        padded_image = tf.pad(x, [[0, 0],[1, 1],[1, 1],[0, 0]])
        result = slim.conv2d_transpose(padded_image, num_out_channel, kernel_size, scale, 'SAME')
        return result[:, 3:-1, 3:-1, :]

    @staticmethod
    def upsample_2d(self, x, scale):
        '''
        The input image is scaled to increase the size of the image
        '''
        shape = tf.shape(x)
        return tf.image.resize_nearest_neighbor(x, [shape[1] * scale, shape[2] * scale])

    @staticmethod
    def get_disparity(self, x):
        '''
        Function for disparity output on a particular scale, the output channels are 2 - probably one for left and one for right
        the activation is sigmoid and the output of a scale is constrained between 0 and dmax = 0.3 * width of image at that scale 
        '''
        ans = 0.3 * self.conv2d(x, 2, 3, 1, tf.nn.sigmoid)
        return x

    def build_residual_vae(self):
        conv = self.conv2d

        if True:
            upconv = self.conv2d_transpose

        with tf.variable_scope('encoder'):
            conv1   = self.conv2d(self.model_input,      32, 7, 1) 
            conv1b  = self.conv2d(conv1,                 32, 7, 2) 
            conv2   = self.conv2d(conv1b,                64, 5, 1)
            conv2b  = self.conv2d(conv2,                 64, 5, 2)
            conv3   = self.conv2d(conv2b,               128, 3, 1)
            conv3b  = self.conv2d(conv3,                128, 3, 2)
            conv4   = self.conv2d(conv3b,               256, 3, 1)
            conv4b  = self.conv2d(conv4,                256, 3, 2)
            conv5   = self.conv2d(conv4b,               512, 3, 1)
            conv5b  = self.conv2d(conv5,                512, 3, 2)
            conv6   = self.conv2d(conv5b,               512, 3, 1)
            conv6b  = self.conv2d(conv6,                512, 3, 2)
            conv7   = self.conv2d(conv6b,               512, 3, 1)
            conv7b  = self.conv2d(conv7,                512, 3, 2)
            
        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7, 512, 3, 2)
            iconv7  = conv(tf.concat([upconv7, conv6b], 3), 512, 3, 1)

            upconv6 = upconv(conv7, 512, 3, 2)
            iconv6  = conv(tf.concat([upconv6, conv5b], 3), 512, 3, 1)

            upconv5 = upconv(conv7, 256, 3, 2)
            iconv5  = conv(tf.concat([upconv5, conv4b], 3), 256, 3, 1)

            upconv4 = upconv(conv7, 128, 3, 2)
            iconv4  = conv(tf.concat([upconv4, conv3b], 3), 128, 3, 1)

            self.disp4 = self.get_disparity(iconv4)

            upconv3 = upconv(conv7,  64, 3, 2)
            iconv3  = conv(tf.concat([upconv3, conv2b, self.upsample_2d(self.disp4, 2)], 3),  64, 3, 1)

            self.disp3 = self.get_disparity(iconv3)

            upconv2 = upconv(conv7,  32, 3, 2)
            iconv2  = conv(tf.concat([upconv2, conv1b, self.upsample_2d(self.disp3, 2)], 3),  32, 3, 1)

            self.disp2 = self.get_disparity(iconv2)

            upconv1 = upconv(iconv2,  16, 3, 2)
            iconv1  = conv(tf.concat([upconv1, self.upsample_2d(self.disp2, 2)], 3),  16, 3, 1)

            self.disp1 = self.get_disparity(iconv1)

    def build_model(self):
        self.model_input = self.left

        self.left_pyramid = Loss.scale_pyramid(self.left, 4)
        self.right_pyramid = Loss.scale_pyramid(self.right, 4)

        self.build_residual_vae()

    def build_outputs(self):
        '''
        Compute the output of the neural network i.e. the left and right disparity at various scales.

        Using these disparities, compute the reconstructed left and right images
        '''
        with tf.variable_scope('disparities'):
            self.predicted_combined_disp = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.predicted_left_disp  = [tf.expand_dims(disp[:,:,:,0], 3) for disp in self.predicted_combined_disp]
            self.predicted_right_disp = [tf.expand_dims(disp[:,:,:,1], 3) for disp in self.predicted_combined_disp]

        with tf.variable_scope('images'):
            self.predicted_left  = [Loss.generate_img_left(self.right_pyramid[i], self.predicted_left_disp[i])  for i in range (4)]
            self.predicted_right = [Loss.generate_img_right(self.left_pyramid[i], self.predicted_right_disp[i]) for i in range (4)]

        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [Loss.generate_img_left(self.predicted_right_disp[i], self.predicted_left_disp[i])  for i in range (4)]
            self.left_to_right_disp = [Loss.generate_img_right(self.predicted_left_disp[i], self.predicted_right_disp[i]) for i in range (4)]

        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness  = Loss.disparity_smoothness(self.predicted_left_disp,   self.left_pyramid)
            self.disp_right_smoothness = Loss.disparity_smoothness(self.predicted_right_disp, self.right_pyramid) 

    def build_losses(self):
        alpha_image_loss = 0.85

        disp_gradient_loss_weight = 0.1
        lr_loss_weight = 1.0

        with tf.variable_scope('losses'):
            # L1 Image Reconstruction Loss
            self.l1_left = [tf.abs(self.predicted_left[i] - self.left_pyramid[i]) for i in range (4)]
            sef.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.predicted_right[i] - self.right_pyramid[i]) for i in range (4)]
            sef.l1_reconstruction_loss_right = [tf.reduce_mean(r) for r in self.l1_right]

            # SSIM
            self.ssim_left = [Loss.SSIM(self.predicted_left[i], self.left_pyramid[i]) for i in range (4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [Loss.SSIM(self.predicted_right[i], self.right_pyramid[i]) for i in range (4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # Weighted Sum
            self.image_loss_right = [alpha_image_loss * self.ssim_loss_right[i] + (1 - alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [alpha_image_loss * self.ssim_loss_left[i]  + (1 - alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # Disparity Smoothness Loss
            self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range (4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range (4)]

            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR Consistency Loss
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.predicted_left_disp[i]))  for i in range (4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.predicted_right_disp[i])) for i in range (4)]

            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # Total Loss
            self.total_loss = self.image_loss + disp_gradient_loss_weight * self.disp_gradient_loss + lr_loss_weight * self.lr_loss