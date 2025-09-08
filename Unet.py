import torch.nn as nn

class MyGoatedUnet(nn.Module):
    def __init__(self, input_size):    # change conv2d to conv 3d  (oops)
        super().__init__()            # n_out = (500 + (2* 0) - 2) / 2
        # check how conv3d works
        self.input_size = input_size
        self.down_block1 = [nn.Conv3d(in_channels=1, out_channels=64, kernel_size=2, padding=0, stride=1), nn.Conv3d(in_channels=64, out_channels=64, kernel_size=2, padding=0, stride=1), nn.Conv3d(in_channels=64, out_channels=64, kernel_size=2, padding=0, stride=1)]
        self.down_block2 = [nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2, padding=0, stride=1), nn.Conv3d(in_channels=128, out_channels=128, kernel_size=2, padding=0, stride=1), nn.Conv3d(in_channels=128, out_channels=128, kernel_size=2, padding=0, stride=1)]
        self.down_block3 = [nn.Conv3d(in_channels=128, out_channels=256, kernel_size=2, padding=0, stride=1), nn.Conv3d(in_channels=256, out_channels=256, kernel_size=2, padding=0, stride=1), nn.Conv3d(in_channels=256, out_channels=256, kernel_size=2, padding=0, stride=1)]
        self.down_block4 = [nn.Conv3d(in_channels=256, out_channels=512, kernel_size=2, padding=0, stride=1), nn.Conv3d(in_channels=512, out_channels=512, kernel_size=2, padding=0, stride=1), nn.Conv3d(in_channels=512, out_channels=512, kernel_size=2, padding=0, stride=1)]
        #self.down = nn.Sequential([self.down_block1 + [self.maxpool] + self.down_block2 + [self.maxpool] + self.down_block3 + [self.maxpool] + self.down_block4])
        self.maxpool = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        output_block1 = self.down_block1(x)
        input_block2 = self.maxpool(output_block1)
        
        output_block2 = self.down_block2(input_block2)
        input_block3 = self.maxpool(output_block2)
        
        output_block3 = self.down_block3(input_block3)
        input_block4 = self.maxpool(output_block1)
        
        output_final_block = self.down_block4(input_block4)
        
        x = self.up(x)
        return x