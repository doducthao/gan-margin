import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import torch.nn as nn 

# print network
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# save images
def save_images(images, size, image_path):
    return imsave(images, size, image_path)

# write images
def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)

# merge images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

# gen animation
def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = 'epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(os.path.join(path, img_name)))
    imageio.mimsave(os.path.join(path, 'generate_animation.gif'), images, fps=5)

# draw loss 
def train_loss_plot(hist, path):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']
    y3 = hist['C_loss']

    fig = plt.figure(figsize=(15,8))
    plt.plot(x, y1, label='Discriminator')
    plt.plot(x, y2, label='Generator')
    plt.plot(x, y3, label='Classifier')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss of (D, G, C)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'TrainingLoss.png'))
    return 

def test_loss_plot(hist, path):
    x = range(len(hist['test_loss']))
    y = hist['test_loss']
    fig = plt.figure(figsize=(15,8))
    plt.plot(x, y, label = 'Classifier')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Test Loss of (C)')

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'TestLoss.png'))
    return 

def acc_plot(hist, path):
    x = range(len(hist['test_accuracy']))
    y = hist['test_accuracy']
    fig = plt.figure(figsize=(15,8))
    plt.plot(x, y, label='Classifier')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy of (C)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(path, 'TestAcc.png')
    plt.savefig(path)
    return 

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()