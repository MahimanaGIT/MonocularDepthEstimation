import tensorflow as tf
import tensorflow.contrib.slim as slim

class Loss:
    '''
    Class for losses
    '''

    def __init__(self):
        self.build_losses()

    @staticmethod
    def scale_pyramid(self, img, num_scales):
        '''
        Scaling a ground truth image into four different scales at which the disparity is computed to 
        compute the loss for each disparity
        Resizes image using area interpolation (averaging of pixels)
        '''
        size = tf.shape(img)
        scaled_img_list = [img]
        for i in range (num_scales - 1):
            ratio = 2 ** (i+1)
            scaled_img_list.append(tf.image.resize_area(img, [size[1] // ratio, size[2] // ratio]))

    @staticmethod
    def gradient_x(self, img):
        '''
        Computes gradient of image in x direction
        '''
        return img[:,:,:-1,:] - img[:,:,1:,:]

    @staticmethod
    def gradient_y(self, img):
        '''
        Computes gradient of image in y direction
        '''
        return img[:,:-1,:,:] - img[:,1:,:,:]

    @staticmethod
    def upsample_nn(self, img, ratio):
        '''
        Resizes the input image to the scale of given ratio
        '''
        size = tf.shape(img)
        return tf.image.resize_nearest_neighbor(img, [size[1] * ratio, size[2] * ratio])

    @staticmethod
    def SSIM(self, x, y):
        '''
        Reconstruction loss having three components:
        1. Luminance (average intensity of the image)
        2. Contrast (standard deviation of image by making the mean zero)
        3. Structural Similarity (Similarity of image having zero mean and 1 standard deviation)
        '''
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        numerator_SSIM = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator_SSIM = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = numerator_SSIM / denominator_SSIM

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    @staticmethod
    def disparity_smoothness(self, disp, img_pyramid):
        '''
        Calculates disparity smoothness loss for the images
        '''
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in img_pyramid]
        image_gradients_y = [self.gradient_y(img) for img in img_pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(grad), 3, keep_dims = True)) for grad in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(grad), 3, keep_dims = True)) for grad in image_gradients_y]
    
        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    @staticmethod
    def generate_img_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    @staticmethod
    def generate_img_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

            
