# Python imports
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics.classification  
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm

#Module imports
import Unet

def load_dataset(resize_value=128, dataset_path='trainnig_data'):
    transform = transforms.Compose([transforms.Resize(resize_value, resize_value),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(dataset_path, transform= transform)  #get library for image folder
    return dataset



def train_model(model, dataset, epochs: int, output_dir: str, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001) #why model params ?
    criterion = torchmetrics.classification.Dice(average='micro') #have to check if this is the proper setup
    best_loss = float('inf')
    for i in range(epochs):
        loss_aggregation = 0
        for image, label in tqdm(dataset):
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_aggregation += loss # why loss.item()????
        avrg_loss = loss_aggregation / len(dataset)
        
        if avrg_loss < best_loss:
            best_loss = avrg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter > patience:
            print(f'Final average loss of epoch {i + 1} is {avrg_loss}')
            torch.save(model.state_dict(), output_dir)
            break
        print(f'average loss of epoch {i + 1} is {avrg_loss}')
        torch.save(model.state_dict(), output_dir)
        
        
if __name__ == "__main__":
    dataset = load_dataset()
    model = Unet.MyGoatedUnet()
    train_model(model, dataset,)