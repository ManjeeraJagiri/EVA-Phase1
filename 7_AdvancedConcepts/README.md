# Assignment 7

# Members
- Jagiri Manjeera - manjeera.jagiri@gmail.com
- Ashwin A G - ashwin100196@gmail.com

# Data Augmentations
- HorizontalFlip 
- ShiftScaleRotate 
- CoarseDropout 

# Convolution Types
## Dilated Convolution
- Alternative to MaxPooling
- Gives local pixel level accuracy as well as wider global context
- Use when we want to increase RF fast and gain a vetter view at global perspective
## Depthwise Separable Convolution
- Reduces the number of parameters as compared to a normal convolution 3x3

# Model Analysis
- Initially worked with C1C2C3C40 architecture with 3 convolutions layers each in first 3 blocks, followed by 1x1 convolution and GAP. 
- We have Transition blocks in between C1-C2 and C2-C3, and maintained channel size across layers in each block
- Followed Resnet-like architecture with # channels i.e 16--->32--->64
- Introduced Depthwise Separable Convolution when increasing number of channels from 32 to 64
- Introduced Dilated Convolution at C3 block to decrease overall number of parameters

# Model Analysis

