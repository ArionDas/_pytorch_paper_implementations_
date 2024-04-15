# Configurations required for DCGAN

# size of the latent z vector
nz = 100

# number of channels
nc = 3 # RGB, use 1 if grey scale images

# number of generator filters
ngf = 64

# number of discriminator filters
ndf = 64

# Learning rate
lr = 1e-2 # 1e-1 might be too high

# Adam hyperparameter
beta1 = 0.5

# number of epochs
EPOCHS = 10

# batch size
batchSize = 16

# Image Size
imageSize = (226, 226)