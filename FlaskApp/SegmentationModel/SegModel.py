import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

class SegModel_UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size = 3, output_padding = 0):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size = kernel_size, in_channels = in_channels, out_channels = out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size = kernel_size, in_channels = out_channels, out_channels = out_channels, padding = output_padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels)
        )
        return block
    
    def expanding_block(self, in_channels, mid_channels, out_channels, kernel_size = 3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size = kernel_size, in_channels = in_channels, out_channels = mid_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.Conv2d(kernel_size = kernel_size, in_channels = mid_channels, out_channels = mid_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ConvTranspose2d(in_channels = mid_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        )
        return block

    def final_block(self, in_channels, mid_channels, out_channels, kernel_size = 3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size = kernel_size, in_channels = in_channels, out_channels = mid_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.Conv2d(kernel_size = kernel_size, in_channels = mid_channels, out_channels = mid_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.Conv2d(kernel_size = kernel_size, in_channels = mid_channels, out_channels = out_channels, padding = 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels)
        )
        return block

    def crop_concat(self, upsampled, bypass, crop = False):
        # print("upsampled shape: ", upsampled.shape)
        # print("bypass shape before padding: ", bypass.shape)
        if crop:
            r_red = (bypass.size()[2] - upsampled.size()[2])//2
            c_red = (bypass.size()[3] - upsampled.size()[3])//2
            bypass = F.pad(bypass, (-c_red, -r_red, -c_red, -r_red)) #[left, right, top, bot]
        # print("bypass shape after padding: ", bypass.shape)
        if(bypass.size()[2]!=upsampled.size()[2] or bypass.size()[3]!=upsampled.size()[3]):
            bypass = F.pad(bypass,((bypass.size()[2]-upsampled.size()[2]), 0, (bypass.size()[3]-upsampled.size()[3]), 0))
        return torch.cat((upsampled, bypass), 1)
    
    def __init__(self, in_channels, out_channels):
        super(SegModel_UNet, self).__init__()

        #Encoding
        self.conv_encode_1 = self.contracting_block(in_channels = in_channels, out_channels = 16)
        self.conv_maxpool_1 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv_encode_2 = self.contracting_block(in_channels = 16, out_channels = 32, output_padding = 1)
        self.conv_maxpool_2 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv_encode_3 = self.contracting_block(in_channels = 32, out_channels = 64, output_padding = 1)
        self.conv_maxpool_3 = nn.MaxPool2d(kernel_size = 2)

        # self.conv_encode_4 = self.contracting_block(in_channels = 256, out_channels = 512)
        # self.conv_maxpool_4 = nn.MaxPool2d(kernel_size = 2)

        #Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),   
            torch.nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        )

        #Decode
        # self.conv_decode_4 = self.expanding_block(1024, 512, 256)
        
        self.conv_decode_3 = self.expanding_block(128, 64, 32)
        
        self.conv_decode_2 = self.expanding_block(64, 32, 16)
        
        self.final_layer = self.final_block(32, 32, out_channels)
 
    def forward(self, x, print_shapes = False):   
        #Encoding
        encode_block_1 = self.conv_encode_1(x)
        encode_pool_1 = self.conv_maxpool_1(encode_block_1)
        encode_block_2 = self.conv_encode_2(encode_pool_1)
        encode_pool_2 = self.conv_maxpool_2(encode_block_2)
        encode_block_3 = self.conv_encode_3(encode_pool_2)
        encode_pool_3 = self.conv_maxpool_3(encode_block_3)
        # encode_block_4 = self.conv_encode_4(encode_pool_3)
        # encode_pool_4 = self.conv_maxpool_4(encode_block_4)
        
        #Bottleneck
        bottleneck1 = self.bottleneck(encode_pool_3)
        # bottleneck1 = self.bottleneck(encode_pool_4)
        
        #Decode
        # decode_input_4 = self.crop_concat(bottleneck1, encode_block_4, crop = True)
        # upsampled_4 = self.conv_decode_4(decode_input_4)
        # decode_input_3 = self.crop_concat(decode_block_4, encode_block_3, crop = True)
        # upsampled_3 = self.conv_decode_3(decode_input_3)
        decode_input_3 = self.crop_concat(bottleneck1, encode_block_3, crop = True)
        upsampled_3 = self.conv_decode_3(decode_input_3)
        decode_input_2 = self.crop_concat(upsampled_3, encode_block_2, crop = True)
        upsampled_2 = self.conv_decode_2(decode_input_2)
        decode_input_1 = self.crop_concat(upsampled_2, encode_block_1, crop = True)
        upsampled_1 = self.final_layer(decode_input_1)

        #Softmax to get class probabilities
        final_output = F.softmax(upsampled_1, dim = 1)
        
        if(print_shapes == True):
            print("encode_block_1: ", encode_block_1.shape)
            print("encode_pool_1: ", encode_pool_1.shape)
            print("encode_block_2: ", encode_block_2.shape)
            print("encode_pool_2: ", encode_pool_2.shape)
            print("encode_block_3: ", encode_block_3.shape)
            print("encode_pool_3: ", encode_pool_3.shape)
            print("bottleneck1: ",bottleneck1.shape)
            print("decode_input_3: ", decode_input_3.shape)
            print("upsampled_3: ", upsampled_3.shape)
            print("decode_input_2: ", decode_input_2.shape)
            print("upsampled_2: ", upsampled_2.shape)
            print("decode_input_1: ", decode_input_1.shape)
            print("upsampled_1: ", upsampled_1.shape)
            print("final_output: ", final_output.shape)
        return final_output
    
def load_model(path_to_model):
    model = SegModel_UNet(in_channels = 1, out_channels = 32)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    # model.eval()
    return model