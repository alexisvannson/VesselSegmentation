import torch
import torch.nn as nn

class MyGoatedUnet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(2, 2)

        # --- Encoder ---
        self.down_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.down_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.down_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.down_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU()
        )

        # --- Decoder ---
        self.upscale1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_block1 = nn.Sequential(
            nn.Conv2d(512+512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )

        self.upscale2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_block2 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        self.upscale3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_block3 = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.upscale4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block4 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)   # final classifier
        )

    def forward(self, x):
        # --- Encoder ---
        out1 = self.down_block1(x)
        x = self.maxpool(out1)

        out2 = self.down_block2(x)
        x = self.maxpool(out2)

        out3 = self.down_block3(x)
        x = self.maxpool(out3)

        out4 = self.down_block4(x)
        x = self.maxpool(out4)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder ---
        x = self.upscale1(x)
        x = torch.cat([out4, x], dim=1)
        x = self.up_block1(x)

        x = self.upscale2(x)
        x = torch.cat([out3, x], dim=1)
        x = self.up_block2(x)

        x = self.upscale3(x)
        x = torch.cat([out2, x], dim=1)
        x = self.up_block3(x)

        x = self.upscale4(x)
        x = torch.cat([out1, x], dim=1)
        x = self.up_block4(x)

        return x
