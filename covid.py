from scipy.sparse.construct import random
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
import collections
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import torch

torch.manual_seed(0)

def show(img, name, norm):
	# convert tensor to numpy array
    npimg = img.numpy()
    if norm:
        npimg = 0.5146469 + 0.3062062 * npimg
	# Convert to H*W*C shape
    npimg_tr=np.transpose(npimg, (1,2,0))
    plt.imsave(name, npimg_tr)
    
def label_statistics(data_set):
    if isinstance(data_set, Subset):
        labels = np.array(data_set.dataset.targets)[data_set.indices]
    else:
        labels = data_set.targets
    counter_stat = collections.Counter(labels)
    return counter_stat

def mean(data_set):
    meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in data_set]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    return [meanR, meanG, meanB]

def std(data_set):
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in data_set]

    stdR = np.mean([m[0] for m in stdRGB])
    stdG = np.mean([m[1] for m in stdRGB])
    stdB = np.mean([m[2] for m in stdRGB])

    return [stdR, stdG, stdB]

def get_data_loader(path_to_data, batch_size, result, max_data):
    # Transformers
    temp_transformer = transforms.Compose([transforms.ToTensor()])

    # Data set
    covid_ds = ImageFolder(path_to_data, temp_transformer)#load_dataset(path_to_data, temp_transformer)
    print("Total dataset class :", label_statistics(covid_ds))

    # index of list
    indices = list(range(len(covid_ds)))
    max_len = min(max_data, len(covid_ds))
    
    train_size = int(max_len*0.6)
    val_size = int(max_len * 0.2)
    test_size = int(max_len * 0.2)#test_size - val_size
    etc = len(covid_ds) - train_size - val_size - test_size
    
    torch.manual_seed(0)
    train_ds, val_ds, test_ds, _ = random_split(covid_ds, [train_size, val_size, test_size, etc])
    print("Train dataset class :", label_statistics(train_ds))
    print("val dataset class :", label_statistics(val_ds))
    print("test dataset class :", label_statistics(test_ds))

    # Sample images
#     sample_size = 4

#     train_sample = [train_ds[i][0] for i in range(sample_size)]
#     val_sample = [val_ds[i][0] for i in range(sample_size)]
#     test_sample = [test_ds[i][0] for i in range(sample_size)]

#     train_sample = make_grid(train_sample, nrow=8, padding=1)
#     val_sample = make_grid(val_sample, nrow=8, padding=1)
#     test_sample = make_grid(test_sample, nrow=8, padding=1)
    
    #show(train_sample, os.path.join(result, "train_sample.png"), True)
    #show(val_sample, os.path.join(result, "val_sample.png"), True)
    #show(test_sample, os.path.join(result, "test_sample.png"), True)
    
    # Transformers
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    global_transformer = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
        norm
    ])

    # Change transformer
    covid_ds.transform = global_transformer
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl, test_dl

