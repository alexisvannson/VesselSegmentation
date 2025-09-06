import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics.classification import Dice
import tqdm

def train_model(model, dataloader, epochs: int, output_dir: str, patience =5):
    optimizer = optim.Adam(model.parameters(), lr=0.001) #why model params ?
    criterion = Dice(average='micro') #have to check if this is the proper setup
    best_loss = float('inf')
    for i in range(epochs):
        loss_aggregation = 0
        for image, label in tqdm(dataloader):
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_aggregation += loss # why loss.item()????
        avrg_loss = loss_aggregation / len(dataloader)
        
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