import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, dataloader, epochs: int, output_dir: str, patience =5):
    optimizer = optim.Adam(model.parameters(), lr=0.001) #why model params ?
    criterion = nn.CrossEntropyLoss() # have to change that with prooper segmentation loss
    previous_loss = float('inf')
    for i in range(epochs):
        loss_aggregation = 0
        for image, label in dataloader:
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_aggregation += loss
        avrg_loss = loss_aggregation / len(dataloader)
        if avrg_loss < previous_loss:
            previous_loss = avrg_loss
            patience = patience
        else:
            patience -= 1
        if patience == 0:
            torch.save(model.state_dict(), output_dir)
            return
        print(f'average loss of epoch {i + 1} is {avrg_loss}')
        torch.save(model.state_dict(), output_dir)