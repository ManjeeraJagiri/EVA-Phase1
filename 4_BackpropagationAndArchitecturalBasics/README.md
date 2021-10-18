# Group Members
1. Ashwin AG
2. Manjeera Jagiri

# Model Architecture

1. Convolutional layer (#kernels : 16)
2. Convolutional layer (#kernels : 16)
3. MaxPooling
4. Convolutional layer (#kernels : 16)
5. Convolutional layer (#kernels : 32)
6. Convolutional layer (#kernels : 32)
7. GAP
8. FC

Except the last convolutional layer, the rest have +BatchNorm layer and +Dropout layer with dropout value=0.05

# Model Hyperparameters

1. Optimizer : SGD, with lr=0.05 and momemtum=0.9
2. Epochs : 20

# Results
1. Model parameters : 19056
2. Validation accuracy : @Epoch 7, we get 99.44

