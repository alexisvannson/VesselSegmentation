# Python imports
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

class Datasetloader(datasets):
    def __init__(self, dataset_path="training_data", resize_value=572):
        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([transforms.Resize(resize_value, resize_value),
                                             transforms.ToTensor()])
        self.dataset = datasets.ImageFolder(dataset_path, transform=self.transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label)