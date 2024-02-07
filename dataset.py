import torch
import numpy as np
import os
from torch.utils.data import Dataset

class HPFrameDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npz')]
        print(self.file_paths)
        self.transform = transform
        self.frames = []
        # get all frames from all files
        for file_path in self.file_paths:
            data = np.load(file_path)['kps']
            for frame in data:
                self.frames.append(frame.reshape(-1))
        print("Loaded {} frames".format(len(self.frames)))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        sample = self.frames[idx]
        label = torch.zeros(1, dtype=torch.float32)

        """if create_negative_samples:
            # 1. create negative samples by randomly permuting the keypoints
            # 2. create negative samples by combing keypoints from different frames
            label = torch.ones(1, dtype=torch.float32)
            sample = self.generate_negative_sample(sample)"""


        return torch.tensor(sample, dtype=torch.float32), label
