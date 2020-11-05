'''
batch size = 8
Epochs = 50
Optimizer: Adam(beta1 = 0.9, beta2 = 0.999, epsilon = 10^-8, lr = 10^-4 (constant for first 30 epochs and halve it every 10 subsequently))

Data Augmentation:
Horizontal Flipping
Color Augmentation

Post Processing:
To reduce the effect of stereo disocclusions which create disparity ramps on both the left and right side of the image
1. For input I, we compute d, flip that input image horizontally and again compute disparity d'
2. Flip this d' horizontally to align with d to get d''
3. Combine both disparity maps to form final result
Image: {------][-------------------][------]}
           5%           90%            5%
           d''      avg(d, d'')        d'
                        ^- confirm it


Things to play with:
    C1 and C2 in SSIM
    Add reuse variables
    Use combination of left and right images as input to model
    Alpha image loss
    Disparity and lr consistency loss
'''

import tensorflow as tf
import model

if __name__ == '__main__':
    model = model.MonocularDepthModel()
    model.build_model()