import torch
import torch.nn as nn

# --- 1. Define the EXACT same model architecture as in app.py ---
# This is crucial for the weights to load correctly.

# Define the number of classes
NUM_CLASSES = 5 
CLASS_NAMES = ['Playing Guitar', 'Jogging', 'Typing on Keyboard', 'Waving Hand', 'Drinking Water']

class MyVideoModel(nn.Module):
    def __init__(self):
        super(MyVideoModel, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        # The output size must match the number of classes
        self.fc = nn.Linear(16, NUM_CLASSES) 

    def forward(self, x):
        x = self.relu(self.conv3d(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# --- 2. Create and Save the Dummy Model ---

# Instantiate the model
model = MyVideoModel()

# The model is initialized with random weights by default. 
# For a real model, you would train it here. For our test, random weights are fine.

# Define the output filename
output_model_path = "your_model.pth"

# Save the model's state dictionary (its learned weights)
torch.save(model.state_dict(), output_model_path)

print(f"✅ Dummy model created and saved to '{output_model_path}'")
print("This file can now be used with your Flask app.")