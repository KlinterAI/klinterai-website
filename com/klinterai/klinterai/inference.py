from model import Net
from PIL import Image
from io import BytesIO
import base64
import requests
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

classes = ('No Issue', 'Risk', 'Potential Risk', 'Issue',
                    'Potential Issue')
mapping = dict(zip(classes, range(len(classes))))

class Dataset():
    _entry = None  # This is a class variable

    @property
    def entry(cls):
        return cls._entry

    @entry.setter
    def entry(cls, value):
        cls._entry = value
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.entry = entry(dataset_name)

    @staticmethod
    def get_entry():
        return Dataset._entry

class entry():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def return_dataset(self):
        if self.dataset_name == "vsa":
            return self.vsa_dataset()

    def vsa_dataset(self):
        
        batch_size = 32
        transform = None

        if self.dataset_name == "vsa":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((224, 224))])
            
        return transform

def run_image_inference(image_url, dataset_name = "vsa"):
    image = None
    inference = Dataset(dataset_name)
    if (image_url.find("data:image") is not -1):
        # data url scheme
        header, encoded = image_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
    else:
        # a URL with extension
        image_data = requests.get(image_url, stream=True)
        image = Image.open(image_data.raw)
    
    # resize image to 224x224
    if image is not None:
        image = image.resize((224, 224))
        # Define the transform
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        # Apply the transform to convert the image to a tensor
        torch_tensor = transform(image)

    # stacking to torch tensor for batch size evaluation
    if torch_tensor is not None:
        net = Net()
        batch_size = 1
        # image_zeros = torch.zeros((31, 3, 224, 224))
        torch_tensor = torch_tensor[None, :]
        torch_tensor = torch_tensor.permute(0, 3, 1, 2)
        # torch_tensor = torch.stack([torch_tensor, image_zeros])

    # pre-processing the tensor
    if torch_tensor is not None:
        transform = inference.get_entry().return_dataset()
        torch_tensor = transform(torch_tensor)

    # inference on image outputs
    if torch_tensor is not None:
        outputs = net(torch_tensor)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                    for j in range(batch_size)))
    
    pass
