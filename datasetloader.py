# Python imports
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.utils.data  as dataloader


class Datasetloader(Dataset):
    def __init__(self, dataset_path="training_data", resize_value=572):
        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([transforms.Resize((resize_value, resize_value)),
                                             transforms.ToTensor(),
                                             transforms.Grayscale()])
        self.dataset = datasets.ImageFolder(dataset_path, transform=self.transform)
        self.dataloader = dataloader.DataLoader(self.dataset, batch_size=8, shuffle=True)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label)