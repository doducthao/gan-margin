from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import numpy as np
from parser import create_parser 
import re 
import os 

class MyDataset(Dataset):
    def __init__(self, images, labels):
        super(MyDataset, self).__init__()
        self.images = images
        self.labels = labels
    
    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target
    
    def __len__(self):
        return len(self.images)
    
def get_dataset(indexes, raw_loader):
    images, labels = [], []
    for idx in indexes:
        image, label = raw_loader[idx]
        images.append(image)
        labels.append(label)
    
    images = torch.stack(images, 0) # shape [100, 1, 28, 28]
    labels = torch.from_numpy(np.array(labels, dtype=np.int64)).squeeze() # torch.Size([100])
    return images, labels

def transform_func(input_size):
    transform = transforms.Compose([
                                    transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    return transform 

# make dataloader and save indexes of choosen data
def dataloader(args):
    transform = transform_func(args.input_size)

    if args.dataset == 'mnist':
        training_set = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

        indexes = np.arange(len(training_set))
        np.random.shuffle(indexes)

        mask = np.zeros(shape=indexes.shape, dtype=np.bool)
        labels = np.array([training_set[i][1] for i in indexes], dtype = np.int64)

        for i in range(10):
            mask[np.where(labels == i)[0][: args.num_labeled // 10]] = True # choosen labeled data
        
        labeled_indexes, unlabeled_indexes = indexes[mask], indexes[~ mask]
        labeled_indexes = list(labeled_indexes)

        labeled_set = get_dataset(labeled_indexes, training_set)
        unlabeled_set = get_dataset(unlabeled_indexes, training_set)
        labeled_set = MyDataset(labeled_set[0], labeled_set[1])
        unlabeled_set = MyDataset(unlabeled_set[0], unlabeled_set[1])

        labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            datasets.MNIST('./data/mnist', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False
            )
    
    return labeled_loader, unlabeled_loader, test_loader, labeled_indexes

# make dataloader from the saved indexes
def dataloader_given_indexes(args):
    transform = transform_func(args.input_size)
    
    if args.dataset == 'mnist':
        training_set = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
        indexes = np.arange(len(training_set))
        with open(args.labeled_indexes, mode='r', encoding='utf-8') as f:
            labeled_indexes = f.readlines()
            
        labeled_indexes = [int(re.sub('\n', '', i).split(",")[0]) for i in labeled_indexes]
        unlabeled_indexes = [i for i in indexes if (not i in labeled_indexes)]

        labeled_set = get_dataset(labeled_indexes, training_set)
        unlabeled_set = get_dataset(unlabeled_indexes, training_set)
        labeled_set = MyDataset(labeled_set[0], labeled_set[1])
        unlabeled_set = MyDataset(unlabeled_set[0], unlabeled_set[1])

        labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            datasets.MNIST('./data/mnist', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False
            )
    
    return labeled_loader, unlabeled_loader, test_loader, labeled_indexes

def create_num_random_labels_custom(num):
    while True:
        num_random_labels = []
        for i in range(9):
            ratio = 0
            while ratio < 0.02 or ratio > 0.2:
                ratio = np.random.uniform(0,1,1)
            n_labels = int(num * ratio)
            num_random_labels.append(n_labels)
        total_9_labels = sum(num_random_labels)
        n_labels_last = num - total_9_labels
        ratio_last = n_labels_last / num 
        if ratio_last <= 0.2 and ratio_last >= 0.02:
            num_random_labels.append(n_labels_last)
            return num_random_labels    

def create_num_random_labels_dirichlet(num):
    ratios = np.random.dirichlet((1,1,1,1,1,1,1,1,1,1), 5)
    ratios = ratios.mean(axis=0)
    num_random_labels = list(map(int, ratios * num))
    num_random_labels[-1] = num - sum(num_random_labels[:-1]) # guarantee sum of list random = num
    return num_random_labels

def create_labeled_indexes_mnist(random_seed=10, uniform=True):
    training_set = datasets.MNIST('data/mnist', train=True, download=True)
    indexes = np.arange(len(training_set))
    np.random.seed(random_seed)

    if uniform:
        num_labels = [50, 100, 600, 1000, 3000]
        dir_name = 'uniform_labeled'
    else:
        num_labels = [1000, 3000]
        dir_name = 'non_uniform_labeled'

    for num in num_labels:
        for stt in range(20):
            indexes = np.arange(len(training_set))
            np.random.shuffle(indexes)

            mask = np.zeros(shape=indexes.shape, dtype=np.bool)
            labels = np.array([training_set[i][1] for i in indexes], dtype = np.int64)

            if uniform:
                for i in range(10):
                    mask[np.where(labels == i)[0][: num // 10]] = True # choosen labeled data
            else:
                num_random_labels = create_num_random_labels_dirichlet(num=num)
                for i in range(10):
                    mask[np.where(labels == i)[0][:num_random_labels[i]]] = True # choosen labeled data
        
            labeled_indexes = list(indexes[mask])
            labeled_indexes = sorted(labeled_indexes, reverse=False)
            labels = [training_set[i][1] for i in labeled_indexes]

            os.makedirs(f'{dir_name}/{num}', exist_ok=True)
            with open(f'{dir_name}/{num}/{stt}.txt', mode='w', encoding='utf-8') as f:
                for (index, label) in zip(labeled_indexes, labels):
                    f.write(str(index) + ',' + str(label)+'\n')                
            with open(f'{dir_name}/{num}/{stt}_desc.txt', mode='w', encoding='utf-8') as f:
                for i in range(10):
                    f.write(str(i) + ':' + str(labels.count(i))+'\n')
    

if __name__ == "__main__":
    # create_labeled_indexes_mnist(uniform=True)
    create_labeled_indexes_mnist(uniform=False)
