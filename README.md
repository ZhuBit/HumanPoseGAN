# HumanPoseGAN: Generating and Discriminating Human Poses

Welcome to the `HumanPoseGAN` GitHub repository. This repository hosts the code for a GAN (Generative Adversarial Network) designed to generate and discriminate human poses for the purpose of my master's thesis. The project leverages the power of PyTorch, a leading deep-learning library, to create realistic human poses.

## Overview

HumanPoseGAN is a machine learning project focusing on the generation and discrimination of human poses using GANs. The project consists of two main components: the HumanPoseGenerator and the HumanPoseDiscriminator, both implemented as neural network models using PyTorch.

## Features

- **Human Pose Generation**: Using the HumanPoseGenerator model, generate realistic human pose data.
- **Human Pose Discrimination**: The HumanPoseDiscriminator model distinguishes between real and generated human poses.
- **3D Visualization**: Visualize generated poses in 3D using Plotly.
- **Dataset Handling**: Includes HPFrameDataset for efficient handling of pose data.

## Installation

To set up this project, follow these steps:
1. Clone the repository.
2. Install the required packages: pip install -r requirements.py 
3. Ensure you have a suitable Python environment.

## Usage

To use the HumanPoseGAN, run the `main.py` script. This will initiate the training process for the GAN. The script includes the creation of both the generator and discriminator models, training on a dataset of human poses, and periodically visualizing the generated poses in 3D.

## Code Structure

- `HumanPoseDiscriminator`: Defines the discriminator model.
- `HumanPoseGenerator`: Defines the generator model.
- `HPFrameDataset`: Custom dataset class for loading and transforming pose data.
- `visualize_frame`: Function to visualize a generated pose in 3D.

## Training

- The models are trained using a dataset of human poses.
- Training involves both the generator and discriminator.
- Visualization of generated poses occurs every 100 epochs.

## Contributions

Contributions to the project are welcome. Please ensure to follow the existing code structure and maintain the readability and quality of the code.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
