import torch.nn as nn
import torch

class MyGoatedUnet(nn.Module):
    def __init__(self, input_size):
        super().__init__()           
        self.input_size = input_size
        self.maxpool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        
        self.down_block1 = nn.Sequential([nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0, stride=1), self.relu(), 
                                          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1), self.relu()])
        
        self.down_block2 = nn.Sequential([nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=1), self.relu(),
                                          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, stride=1), self.relu()])
        
        self.down_block3 = nn.Sequential([nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1), self.relu(),
                                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=1), self.relu()])
        
        self.down_block4 = nn.Sequential([nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0, stride=1), self.relu(),
                                          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, stride=1), self.relu()])
        
        self.upscale1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, padding=0, stride=1)
        
        self.up_block1 = nn.Sequential([nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, stride=1), self.relu(),
                                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, stride=1), self.relu()])
        
        self.upscale2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=0, stride=1)
        
        self.up_block2 = nn.Sequential([nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=1), self.relu(),
                                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=1), self.relu()])
        
        self.upscale3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=0, stride=1)
        
        self.up_block3 = nn.Sequential([nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, stride=1), self.relu(),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, stride=1), self.relu()])
        
        self.upscale4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=0, stride=1)
        
        self.up_block4 = nn.Sequential([nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1), self.relu(),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1), self.relu(),
                                        nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding=0, stride=1)])
    def forward(self, x):
        # Encoder
        output_block1 = self.down_block1(x)
        input_block2 = self.maxpool(output_block1)
        
        output_block2 = self.down_block2(input_block2)
        input_block3 = self.maxpool(output_block2)
        
        output_block3 = self.down_block3(input_block3)
        input_block4 = self.maxpool(output_block1)
        
        output_final_block = self.down_block4(input_block4)
        
        # Decoder
        input_block_up1 = torch.concat((input_block4, self.upscale1(output_final_block)), dim=1)
        output_block_up_1 = self.up_block1(input_block_up1)
        
        input_block_up2 = torch.concat((input_block3, self.upscale2(output_block_up_1)), dim=1)
        output_block_up_2 = self.up_block2(input_block_up2)
        
        input_block_up3 = torch.concat((input_block3, self.upscale3(output_block_up_2)), dim=1)
        output_block_up_3 = self.up_block3(input_block_up3)
        
        input_block_up4 = torch.concat((input_block3, self.upscale4(output_block_up_2)), dim=1)
        output_block_up_4 = self.up_block4(input_block_up4)
        
        return output_block_up_4