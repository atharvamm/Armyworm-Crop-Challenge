import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from cv2 import imread
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,x_train,y_train,transforms):
        super(ImageDataset,self).__init__()
        self.image_dir = x_train
        self.image_labels = y_train
        self.transforms = transforms

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self,idx):
        img = Image.open("Images/"+ self.image_dir[idx])
        return self.transforms(img),self.image_labels[idx]

class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self,x_test,transforms):
        super(TestImageDataset,self).__init__()
        self.image_dir = x_test
        self.transforms = transforms

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self,idx):
        img = Image.open("Images/"+self.image_dir[idx][0])
        return self.image_dir[idx][0],self.transforms(img)

# Add transformations to the images in the training set to make a larger set.