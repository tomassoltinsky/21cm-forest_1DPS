import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time  # Import time module
import logging

class UnetModel(nn.Module):
    logger = logging.getLogger(__name__)
    def __init__(self, input_size, input_channels, output_size, dropout=0.2, step=4):
        super(UnetModel, self).__init__()
        self.timing_info = {
            'enc1_time': 0,
            'enc2_time': 0,
            'enc3_time': 0,
            'enc4_time': 0,
            'dense_time': 0,
            'dec0_time': 0,
            'dec1_time': 0,
            'dec2_time': 0,
            'dec3_time': 0,
            'dec0cat_time': 0,
            'dec1cat_time': 0,
            'dec2cat_time': 0,

            'overall_time': 0
        }

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, 5, padding=2),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )
        
        # Bottleneck
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * (input_size//step**4), 256) 
        print("## NN architecture: input_size:", input_size)
        print("## NN architecture: step size:", step)
        print("## NN architecture: fc1 Weight shape:", self.fc1.weight.shape)
        print("## NN architecture: fc1 Bias shape:", self.fc1.bias.shape)

        ## Expansion starts here
        self.fc2 = nn.Linear(256, 512 * (input_size//step**4))  # Expand back to match decoder input
        print("## NN architecture: fc1 Weight shape:", self.fc1.weight.shape)
        print("## NN architecture: fc1 Bias shape:", self.fc1.bias.shape)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, (input_size//step**4)))

        # Decoder
        self.dec0 = nn.Sequential(
            nn.ConvTranspose1d(1024, 256, step, stride=step, output_padding=0),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, step, stride=step, output_padding=0),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, step, stride=step, output_padding=0),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, step, stride=step, output_padding=0),
            nn.BatchNorm1d(32),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final layer
        channels = 32 #input_channels + 
        self.final = nn.Sequential(
            nn.Conv1d(channels, 1, 1),  # Change output channels to 1
            nn.Flatten()  # Add flatten layer to match target shape
        )

    def forward(self, x):
        start_time = time.time()  # Start timing

        # Print shapes for debugging
        #print(f"Input shape: {x.shape}")
        # If input is single channel, add channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        # If input already has channels, it will remain unchanged
        #print(f"After unsqueeze: {x.shape}")        
        # Encoder
        enc1_start = time.time()  # Start timing for enc1
        enc1 = self.enc1(x)
        self.timing_info['enc1_time'] += (time.time() - enc1_start)  # Time taken for enc1
        #print(f"After enc1: {enc1.shape}")        
        enc2_start = time.time()  # Start timing for enc2
        enc2 = self.enc2(enc1)
        self.timing_info['enc2_time'] += (time.time() - enc2_start)  # Time taken for enc2
        #print(f"After enc2: {enc2.shape}")        
        enc3_start = time.time()  # Start timing for enc3
        enc3 = self.enc3(enc2)
        self.timing_info['enc3_time'] += (time.time() - enc3_start)  # Time taken for enc3
        #print(f"After enc3: {enc3.shape}")        
        enc4_start = time.time()  # Start timing for enc4
        enc4 = self.enc4(enc3)
        self.timing_info['enc4_time'] += (time.time() - enc4_start)  # Time taken for enc3
        #print(f"After enc4: {enc4.shape}")        

        dense_start = time.time()  # Start timing for dense layers

        x = self.flatten(enc4)
        x = self.fc1(x)

        ## Expansion starts here
        x = self.fc2(x)
        x = self.unflatten(x)
        x = torch.cat([x, enc4], dim=1)
        self.timing_info['dense_time'] += (time.time() - dense_start)  # Time taken for dense

        # Decoder with skip connections
        dec0_start = time.time()  # Start timing for dec1
        dec0 = self.dec0(x)
        self.timing_info['dec0_time'] += (time.time() - dec0_start)  # Time taken for dec0
        #print(f"After dec0: {dec0.shape}")  
        dec0cat_start = time.time()  # Start timing for dec1
        dec0 = torch.cat([dec0, enc3], dim=1)
        self.timing_info['dec0cat_time'] += (time.time() - dec0cat_start)  # Time taken for dec1cat

        dec1_start = time.time()  # Start timing for dec1
        dec1 = self.dec1(dec0)
        self.timing_info['dec1_time'] += (time.time() - dec1_start)  # Time taken for dec1
        #print(f"After dec1: {dec1.shape}")  

        dec1cat_start = time.time()  # Start timing for dec1
        dec1 = torch.cat([dec1, enc2], dim=1)
        self.timing_info['dec1cat_time'] += (time.time() - dec1cat_start)  # Time taken for dec1cat
        #print(f"After dec1-cat: {dec1.shape}")        

        # Decoder with skip connections
        dec2_start = time.time()  # Start timing for dec2
        dec2 = self.dec2(dec1)
        self.timing_info['dec2_time'] += (time.time() - dec2_start)  # Time taken for dec2
        #print(f"After dec1: {dec1.shape}")        

        dec2cat_start = time.time()  # Start timing for dec1
        dec2 = torch.cat([dec2, enc1], dim=1)
        self.timing_info['dec2cat_time'] += (time.time() - dec2cat_start)  # Time taken for dec1
        #print(f"After dec1-cat: {dec1.shape}")        
        
        dec3_start = time.time()  # Start timing for dec3
        dec3 = self.dec3(dec2)
        self.timing_info['dec3_time'] += (time.time() - dec3_start)  # Time taken for dec3
        #print(f"After dec2: {dec2.shape}")        
        #dec3 = torch.cat([dec3, x], dim=1)
                 
        #print(f"Before final: {dec4.shape}")
        out = self.final(dec3)
        # Calculate the difference in size
        #print(f"out.shape={out.shape}, x.shape={x.shape}")
        #print(f"out.size(1)={out.size(1)}, x.size(2)={x.size(2)}")
        #print(f"Output shape: {out.shape}")

        # Print overall time taken for the forward pass
        self.timing_info['overall_time'] += (time.time() - start_time)


        return out

    def get_denoised_signal(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            return self.forward(x)

    def get_latent_features(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Pass input to the first layer and get output from enc3
            #input_tensor = self.convert_to_pytorch_tensors(X_train, y_train, None, None)[0]
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add channel dimension
            # If input already has channels, it will remain unchanged
            #print(f"After unsqueeze: {x.shape}")        
            # Encoder
            enc1 = self.enc1(x)
            #print(f"After enc1: {enc1.shape}")        
            enc2 = self.enc2(enc1)
            #print(f"After enc2: {enc2.shape}")        
            enc3 = self.enc3(enc2)
            #print(f"After enc3: {enc3.shape}")        
            enc4 = self.enc4(enc3)
            #print(f"After enc3: {enc3.shape}")   
            x = self.flatten(enc4)
            x = self.fc1(x)     
            latent = x.cpu().numpy()  # Flatten and convert to numpy
            return latent

    def save_model(self, file_path):
        """Save the model to a file."""
        torch.save(self.state_dict(), file_path)  # Save the model's state_dict

    def load_model(self, file_path):
        """Load the model from a file."""
        self.load_state_dict(torch.load(file_path))  # Load the model's state_dict
        self.eval()  # Set the model to evaluation mode

    def get_timing_info(self):
        """Print the timing information for each stage."""
        timing_info_str = f"Timing Information:\n"
        timing_info_str += f"Encoder 1 time: {self.timing_info['enc1_time']:.4f}s\n"
        timing_info_str += f"Encoder 2 time: {self.timing_info['enc2_time']:.4f}s\n"
        timing_info_str += f"Encoder 3 time: {self.timing_info['enc3_time']:.4f}s\n"
        timing_info_str += f"Encoder 4 time: {self.timing_info['enc4_time']:.4f}s\n"
        timing_info_str += f"Dense time: {self.timing_info['dense_time']:.4f}s\n"
    
        timing_info_str += f"Decoder 0 time: {self.timing_info['dec0_time']:.4f}s\n"
        timing_info_str += f"Decoder 1 time: {self.timing_info['dec1_time']:.4f}s\n"
        timing_info_str += f"Decoder 1 Concat time: {self.timing_info['dec1cat_time']:.4f}s\n"
        timing_info_str += f"Decoder 2 time: {self.timing_info['dec2_time']:.4f}s\n"
        timing_info_str += f"Decoder 2 Concat time: {self.timing_info['dec2cat_time']:.4f}s\n"
        timing_info_str += f"Decoder 3 time: {self.timing_info['dec3_time']:.4f}s\n"
        timing_info_str += f"Overall time for forward pass: {self.timing_info['overall_time']:.4f}s\n"
        return timing_info_str        