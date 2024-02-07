import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from dataset import HPFrameDataset
from visualizer_html import visualize_and_save_frame_with_belief
class ConvolutionalDiscriminator(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ConvolutionalDiscriminator, self).__init__()
        # 3x17
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 5), stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=2, padding=1)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 5), stride=2, padding=1)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 5), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout2d(dropout_rate)

        self.fc = nn.Linear(512 * 1 * 2, 1)

    def forward(self, x):
        x = x.view(-1, 1, 3, 17)

        x = F.leaky_relu(self.dropout1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.dropout2(self.bn2(self.conv2(x))), 0.2)
        x = F.leaky_relu(self.dropout3(self.bn3(self.conv3(x))), 0.2)
        x = F.leaky_relu(self.dropout4(self.bn4(self.conv4(x))), 0.2)

        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))
        return x

class ConvolutionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, dropout_rate=0.1):
        super(ConvolutionalGenerator, self).__init__()
        self.init_size = 512
        self.fc1 = nn.Linear(latent_dim, self.init_size * 1 * 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(self.init_size),
            nn.Upsample(scale_factor=2),

            nn.Conv1d(self.init_size, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv1d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):

        out = self.fc1(z)
        out = out.view(out.shape[0], 512, 1, 2)

        # Pass through conv blocks
        for layer in self.conv_blocks:
            out = layer(out)

        # Final reshape to match the output size
        out = out.view(out.shape[0], 51)
        return out

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_data_path = 'data/npz'


    output_dir = "data/outputs/GAN+NoNoise"
    #output_dir = "data/outputs/TEST"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = 64
    num_epochs = 1000

    total_samples = 0
    correct_predictions = 0
    true_positives = 0
    total_predicted_positives = 0

    noise_factor = 0.02
    apply_noise_prob = 0.5 # How often do noises

    train_dataset = HPFrameDataset(train_data_path)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_dataset = HPFrameDataset(test_data_path)


    generator = ConvolutionalGenerator().to(device)
    discriminator = ConvolutionalDiscriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999), weight_decay=1e-5)
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

            if torch.rand(1).item() < apply_noise_prob:
                for n in range(item.shape[0]):  # Iterate over each item in the batch
                    num_keypoints_to_noise = torch.randint(0, min(5, item.shape[1] // 3),
                                                           (1,)).item()
                    keypoints_indices = torch.randperm(item.shape[1] // 3)[
                                        :num_keypoints_to_noise]  # Adjusted for flattened structure
                    for kp in keypoints_indices:
                        start_index = kp * 3
                        end_index = start_index + 3
                        # Adjust noise addition for a 2D tensor
                        item[n, start_index:end_index] += noise_factor * torch.randn_like(
                            item[n, start_index:end_index]).to(device)

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

    generator_model_path = os.path.join(output_dir, 'generator.pth')
    discriminator_model_path = os.path.join(output_dir, 'discriminator.pth')

    torch.save(generator.state_dict(), generator_model_path)
    torch.save(discriminator.state_dict(), discriminator_model_path)

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
    description_text = (f'Batch size: {batch_size}\n'
                        f'Num epochs: {num_epochs}\n'
                        f'Gen Loss: {avg_generator_losses[-1]:.2f}, Discrm Loss: {avg_discriminator_losses[-1]:.2f}\n'
                        f'Real Acc: {real_accuracies[-1] * 100:.2f}%, Fake Acc: {fake_accuracies[-1] * 100:.2f}%')

    plt.text(0.95, 0.05, description_text,
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes,
             fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()

    plt.tight_layout()
    comparison_plot_filename = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_plot_filename)

    plt.show()
    plt.close()
