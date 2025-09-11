# Python imports
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics.classification  
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
import torchsummary
import torch.utils.data

#Module imports
import unet
import datasetloader


def train_model(model, images_path, label_path, epochs: int, output_dir: str, patience=5):
    dataset = datasetloader.SegmentationDataset(images_path, label_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    criterion = nn.CrossEntropyLoss() 
    best_loss = float('inf')
    for i in range(epochs):
        loss_aggregation = 0
        for image, label in tqdm.tqdm(dataloader):
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_aggregation += loss.item()
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
    model = unet.MyGoatedUnet()
    print("Model instance created")
    train_model(model, 'training_data/images', 'training_data/1st_manual', 10, "results")
    print("the End")
"""
    # There are several ways to speed up UNet training:
    # 1. Use a GPU if available:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
    model.to(device)

    # 2. In your train_model, move images and labels to the device:
    #    (You should update train_model to move data to the device inside the training loop.)
    # 3. Increase batch size if memory allows.
    # 4. Use mixed precision training (torch.cuda.amp) for faster computation on modern GPUs.
    # 5. Use num_workers > 0 in DataLoader for faster data loading.
    # 6. Profile your data pipeline to ensure no bottlenecks.
    # 7. Optionally, reduce model size or input resolution for faster epochs.
    """