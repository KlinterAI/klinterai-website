import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import zipfile
from model import Net
from skimage import io
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

classes = ('No Issue', 'Risk', 'Potential Risk', 'Issue',
                    'Potential Issue')
mapping = dict(zip(classes, range(len(classes))))

class CustomImageDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1].replace('\\', '/'))
        image = io.imread(img_path)
        try:
          label = mapping[self.img_labels.iloc[idx, 4].strip()]
          if self.transform:
              image = self.transform(image)
        except:
          print(classes, self.img_labels.iloc[idx, 4].strip())
        return image, label


class entry():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def download_dataset():
        url = "http://example.com/KlinterAI-VSA-Dataset.zip"
        local_file = "./KlinterAI.zip"
        urllib.request.urlretrieve(url, local_file)
        zip_ref = zipfile.ZipFile("./KlinterAI.zip", 'r')
        zip_ref.extractall("./KlinterAI/Dataset")
        zip_ref.close()
        pass

    def return_dataset(self):
        if self.dataset_name == "vsa":
            self.download_dataset()
            return self.vsa_dataset()

    def vsa_dataset(self):
        
        batch_size = 32

        if self.dataset_name == "vsa":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((224, 224))])

            trainset = CustomImageDataset(annotations_file='./train_anomaly_detection.csv', img_dir='./KlinterAI/Dataset', transform=transform)
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

            testset = CustomImageDataset(annotations_file='./test_anomaly_detection.csv', img_dir='./KlinterAI/Dataset', transform=transform)
            testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)