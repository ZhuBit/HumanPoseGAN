import os
import plotly.graph_objects as go
import webbrowser
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HumanPoseDiscriminator(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(HumanPoseDiscriminator, self).__init__()
        # (17 keypoints * 3 coordinates)
        self.model = nn.Sequential(
            nn.Linear(51, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, frame):
        validity = self.model(frame)
        return validity

class HumanPoseGenerator(nn.Module):
    def __init__(self):
        super(HumanPoseGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 51), # 17 keypoints * 3 coordinates
            nn.Tanh()
        )

    def forward(self, z):
        pose = self.model(z)
        return pose
class HPFrameDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npz')]
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

def visualize_frame(frame):
    x = frame[:, 0]
    y = frame[:, 2]
    z = - frame[:, 1]

    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=5, color='red')
    )

    pairs = [(0,1), (0,4), (0,7), (7,8), (8,9), (9,10), (4,5), (1,2), (5,6), (2,3),
             (8,11), (8,14), (11,12), (14,15), (12,13), (15,16)]

    lines = []
    for pair in pairs:
        lines.append(
            go.Scatter3d(
                x=[x[pair[0]], x[pair[1]]],
                y=[y[pair[0]], y[pair[1]]],
                z=[z[pair[0]], z[pair[1]]],
                mode='lines',
                line=dict(color='green', width=5)
            )
        )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[min(x), max(x)]),
            yaxis=dict(nticks=10, range=[min(y), max(y)]),
            zaxis=dict(nticks=10, range=[min(z), max(z)])
        )
    )

    fig = go.Figure(data=[scatter] + lines, layout=layout)

    filename = "plot.html"
    fig.write_html(filename)

    webbrowser.open(filename)
    time.sleep(3)


def visualize_and_save_frame_with_belief(frame, title, epoch, image_type, belief, probability,
                                         output_dir='data/outputs/GAN1'):
    x = frame[:, 0]
    y = frame[:, 2]
    z = -frame[:, 1]  # Correcting the orientation here

    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color='red')
    )

    pairs = [(0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (9, 10), (4, 5), (1, 2), (5, 6),
             (2, 3), (8, 11), (8, 14), (11, 12), (14, 15), (12, 13), (15, 16)]

    lines = [go.Scatter3d(
        x=[x[pair[0]], x[pair[1]]],
        y=[y[pair[0]], y[pair[1]]],
        z=[z[pair[0]], z[pair[1]]],
        mode='lines',
        line=dict(color='green', width=2)
    ) for pair in pairs]


    # Adding text annotation for discriminator's belief, probability, and additional details
    annotations = [
        dict(
            showarrow=False,
            x=sum(x) / len(x), y=sum(y) / len(y), z=sum(z) / len(z),
            text=f"{title}, Epoch: {epoch}, Type: {image_type}, Belief: {belief}, Probability: {probability:.2f}",
            xanchor="left",
            xshift=10,
            font=dict(color="black", size=14)
        ),
        dict(
            showarrow=False,
            x=sum(x) / len(x), y=min(y), z=min(z),  # Positioning this at the bottom
            text=f"{title}, Epoch: {epoch}, Type: {image_type}, Belief: {belief}, Probability: {probability:.2f}",
            xanchor="left",
            yanchor="bottom",
            font=dict(color="blue", size=12)
        )
    ]

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', nticks=10, range=[min(x), max(x)]),
            yaxis=dict(title='Y', nticks=10, range=[min(y), max(y)]),
            zaxis=dict(title='Z', nticks=10, range=[min(z), max(z)]),
            annotations=annotations
        ),
        margin=dict(l=0, r=0, b=0, t=30)  # Adjust margins to make room for title and details
    )

    fig = go.Figure(data=[scatter] + lines, layout=layout)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Constructing filename
    filename = os.path.join(output_dir, f"{epoch}_{image_type}_{belief}_{probability:.2f}.html")
    fig.write_html(filename)

    return filename

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_data_path = 'data/npz'


    output_dir = "data/outputs/GANL1L20.002"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = 32
    num_epochs = 1000

    total_samples = 0
    correct_predictions = 0
    true_positives = 0
    total_predicted_positives = 0

    train_dataset = HPFrameDataset(train_data_path)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_dataset = HPFrameDataset(test_data_path)


    generator = HumanPoseGenerator().to(device)
    discriminator = HumanPoseDiscriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999), weight_decay=1e-5)
    adversarial_loss = torch.nn.BCELoss()

    total_samples = 0
    correct_predictions = 0
    true_positives = 0
    total_predicted_positives = 0

    generator_losses = []
    discriminator_losses = []
    avg_generator_losses = []
    avg_discriminator_losses = []

    real_accuracies = []
    fake_accuracies = []

    learning_rates_G = []
    learning_rates_D = []

    discriminator_scores_real = []
    discriminator_scores_generated = []

    training_times = []
    l1_lambda = 1e-5

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch))
        sum_generator_loss = 0
        sum_discriminator_loss = 0
        total_batches = 0
        epoch_discriminator_losses = discriminator_losses[epoch * total_batches: (epoch + 1) * total_batches]
        for i, (item, _) in enumerate(dataloader):
            print("Batch: {}".format(i))
            item = item.to(device)
            valid = torch.ones((item.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((item.size(0), 1), requires_grad=False).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(item.shape[0], 100).to(device)
            generated_item = generator(z)

            g_loss = adversarial_loss(discriminator(generated_item), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(item), valid)
            fake_loss = adversarial_loss(discriminator(generated_item.detach()), fake)

            l1_loss = 0
            for param in discriminator.parameters():
                l1_loss += torch.sum(torch.abs(param))
            d_loss = (real_loss + fake_loss) / 2 + l1_lambda * l1_loss

            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())
            d_loss.backward()
            optimizer_D.step()

            with torch.no_grad():
                real_predictions = discriminator(item)
                fake_predictions = discriminator(generated_item.detach())

                # Threshold predictions for binary classification
                real_predictions = real_predictions > 0.5
                fake_predictions = fake_predictions > 0.5

                real_accuracy = (real_predictions > 0.5).float().mean().item()
                fake_accuracy = (fake_predictions <= 0.5).float().mean().item()

                real_accuracies.append(real_accuracy)
                fake_accuracies.append(fake_accuracy)

                discriminator_scores_real.extend(real_predictions.cpu().numpy())
                discriminator_scores_generated.extend(fake_predictions.cpu().numpy())

                total_samples += item.size(0) * 2  # real and fake samples
                correct_predictions += (real_predictions == valid).sum().item() + (
                        fake_predictions == fake).sum().item()

                true_positives += real_predictions.sum().item()
                total_predicted_positives += real_predictions.sum().item() + fake_predictions.sum().item()

                if epoch % 10 == 0 and i == 0 and epoch > 10:
                    with torch.no_grad():
                        real_predictions = discriminator(item)
                        fake_predictions = discriminator(generated_item.detach())

                        # Calculate binary predictions and probabilities
                        real_pred_binary = (real_predictions > 0.5).float()
                        fake_pred_binary = (fake_predictions > 0.5).float()

                        # Select a single sample for visualization
                        sample_real = item[0]  # First item in the batch for real
                        sample_generated = generated_item[0]  # First item in the generated batch

                        real_pred_for_viz = real_pred_binary[0].item()  # Binary prediction for real sample
                        fake_pred_for_viz = fake_pred_binary[0].item()  # Binary prediction for fake sample

                        real_prob_for_viz = real_predictions[0].item()  # Probability for real sample
                        fake_prob_for_viz = fake_predictions[0].item()  # Probability for fake sample

                        # Visualization calls
                        visualize_and_save_frame_with_belief(sample_real.cpu().reshape(17, 3), f"Epoch {epoch} Real",
                                                             epoch, "real",
                                                             real_pred_for_viz, real_prob_for_viz, output_dir)
                        visualize_and_save_frame_with_belief(sample_generated.cpu().reshape(17, 3),
                                                             f"Epoch {epoch} Generated", epoch,
                                                             "generated", fake_pred_for_viz, fake_prob_for_viz,
                                                             output_dir)
            sum_generator_loss += g_loss.item()
            sum_discriminator_loss += d_loss.item()
            total_batches += 1

            # Add the individual losses to their respective lists (if you're tracking per batch losses)
            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())
            learning_rates_G.append(optimizer_G.param_groups[0]['lr'])
            learning_rates_D.append(optimizer_D.param_groups[0]['lr'])
            accuracy = correct_predictions / total_samples
            precision = true_positives / total_predicted_positives if total_predicted_positives > 0 else 0

            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Accuracy: {accuracy:.2f}] [Precision: {precision:.2f}]")

        average_generator_loss = sum_generator_loss / total_batches
        average_discriminator_loss = sum_discriminator_loss / total_batches
        avg_generator_losses.append(average_generator_loss)
        avg_discriminator_losses.append(average_discriminator_loss)

    torch.save(generator.state_dict(), 'models/generator.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')
    plt.figure(figsize=(15, 10))

    # Generator Loss vs. Discriminator Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Generator Loss vs. Discriminator Loss")

    # Real vs. Fake Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(real_accuracies, label="Real Accuracy")
    plt.plot(fake_accuracies, label="Fake Accuracy")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Real vs. Fake Accuracy")

    # Learning Rates plot
    plt.subplot(2, 3, 3)
    plt.plot(learning_rates_G, label="Generator Learning Rate")
    plt.plot(learning_rates_D, label="Discriminator Learning Rate")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.title("Generator and Discriminator Learning Rates")

    # Discriminator Scores Distribution plot
    plt.subplot(2, 3, 4)
    discriminator_scores_real = np.array(discriminator_scores_real).flatten().astype(int)
    discriminator_scores_generated = np.array(discriminator_scores_generated).flatten().astype(int)
    plt.hist(discriminator_scores_real, bins=50, alpha=0.5, label="Real Scores", color='blue')
    plt.hist(discriminator_scores_generated, bins=50, alpha=0.5, label="Generated Scores", color='red')
    plt.legend()
    plt.xlabel("Discriminator Scores")
    plt.ylabel("Frequency")
    plt.title("Discriminator Scores Distribution")

    # Training Loss Curves vs Epochs
    epochs = list(range(1, num_epochs + 1))
    plt.subplot(2, 3, 5)
    plt.plot(epochs, avg_generator_losses, label="Generator Loss", marker='o')
    plt.plot(epochs, avg_discriminator_losses, label="Discriminator Loss", marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    comparison_plot_filename = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_plot_filename)
    plt.close()



