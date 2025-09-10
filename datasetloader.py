# Python imports
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data  as dataloader
import os
import PIL

class SegmentationDataset(Dataset):
    def __init__(self, images_path, label_path, resize_value=572):
        super().__init__()
        self.images_path = images_path
        self.label_path = label_path
        self.img_transform = transforms.Compose([transforms.Resize((resize_value, resize_value)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5], std=[0.5])])
        
        self.label_path = transforms.Compose([transforms.Resize((resize_value, resize_value)),
                                            transforms.ToTensor(),
                                            transforms.Grayscale()])
        
        self.images = sorted(os.listdir(images_path))
        self.label = sorted(os.listdir(label_path))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = PIL.Image.open(os.path.join(self.images_path , self.images[idx])).convert('RGB')  
        label = PIL.Image.open(os.path.join(self.label_path , self.label[idx])).convert('L')
        
        image = self.img_transform(image)
        label = self.label_transform(label)
        return image, label



dataset = SegmentationDataset(images_path='images', label_path='label')
dataloader = dataloader.DataLoader(dataset, batch_size=8, shuffle=True)