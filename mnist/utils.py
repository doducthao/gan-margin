import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    desc = "Relativistic Large Margin Softmax Semi-supervised Learning"
    parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gan_type', type=str, default='rlmsoftmax-ssl',
                        choices=['rlmsoftmax-ssl'],
                        help='Type of GAN-SSL')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--acc_time_dir', type=str, default='acc_time', help='Directory name to save accuracy and time training')
    parser.add_argument('--labeled_data_indices', type=str, default='labeled_data_indices', help='Directory name to save labeled data indices')

    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lrC', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--num_labels', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--epoch',  type=int, default=40, help='The number of epochs to run')

    parser.add_argument('--change_nlabels', type=bool, default=False, help="change number of labeled data")
    parser.add_argument('--change_alpha', type=bool, default=False, help="change alpha values")

    args = parser.parse_args()

    return args

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