import torch
import torch.nn as nn
import torchvision.transforms.functional  as transforms

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),     
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class MyGoatedUnet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, feature_dims = [64, 128, 256, 512, 1024]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
         # --- Encoder ---
        for feature in feature_dims:
            self.downs.append(DoubleConv(input_channels, feature))
            input_channels = feature
        
        # --- Decoder ---
        for feature in reversed(feature_dims):
            self.ups.append(nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(2 * feature, feature))
        
        self.bottleneck = DoubleConv(feature_dims[-1], feature_dims[-1]*2)
        self.final_conv = nn.Conv2d(feature_dims[0], num_classes, kernel_size=1)
    

    def forward(self, x):
        skip_layers = []
        for layer in self.downs:
            x = layer(x)
            skip_layers.append(x)
            x = self.maxpool(x)
        x = self.bottleneck(x)
        skip_layers = skip_layers[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_layers[idx//2]
            if x.shape[2:] != skip_connection.shape[2:]:
                x = transforms.resize(x, size=skip_connection.shape[2:])
            
            concat_layer = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_layer)

        return self.final_conv(x)
