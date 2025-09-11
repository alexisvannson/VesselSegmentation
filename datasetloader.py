# Python imports
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data  as dataloader
import os
import PIL
import glob

class SegmentationDataset(Dataset):
    def __init__(self, images_path, label_path, resize_value=572):
        super().__init__()
        self.images_path = images_path
        self.label_path = label_path
        self.img_transform = transforms.Compose([transforms.Resize((resize_value, resize_value)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5], std=[0.5])])
        
        self.label_transform = transforms.Compose([transforms.Resize((resize_value, resize_value)),
                                            transforms.ToTensor(),
                                            transforms.Grayscale()])
        
        self.images = sorted(os.listdir(images_path))
        self.labels = sorted(os.listdir(label_path))

    def __len__(self):
        assert len(self.images) == len(self.labels), "Number of images and labels should be the same"
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.images[idx])
        print(image_path)
        label_path = os.path.join(self.label_path, self.labels[idx])
        print(label_path)
        print(os.path.basename(image_path))
        print(os.path.basename(label_path))
        image = PIL.Image.open(image_path)
        label = PIL.Image.open(label_path)
        image = self.img_transform(image)
        label = self.label_transform(label)
        return image, label




if __name__ == "__main__":
    dataset = SegmentationDataset(images_path='/Users/philippevannson/Desktop/ongoing_stuff/VesselSegmentation/training_data/images', label_path='/Users/philippevannson/Desktop/ongoing_stuff/VesselSegmentation/training_data/1st_manual')
    print(dataset.__len__())
    dataloader = dataloader.DataLoader(dataset, batch_size=8, shuffle=True)
    for image, label in dataloader:
        print(image.shape)
        print(label.shape)
        break