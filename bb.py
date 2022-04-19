import torch
import numpy
import pandas
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ResNet34
from dataset import data_set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet34()
parameters = torch.load('./path(old)/train_cat.pth', map_location=torch.device(device))
print(parameters)
exit(0)
model.load_state_dict(parameters)
model.to(device)
data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])