import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        test_input = torch.rand(1, 1, 128, 128)
        flattened_size = self.encoder(test_input).size(1)

        self.embedding_layer = nn.Sequential(
            nn.Linear(flattened_size, embedding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, flattened_size),
            nn.ReLU(),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        embedding = self.encoder(x)
        embedding = self.embedding_layer(embedding)
        reconstruction = self.decoder(embedding) # Pass embedding through decoder
        return reconstruction, embedding # Return both reconstruction and embedding


def train_autoencoder(autoencoder, dataloader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    for epoch in range(epochs):
        autoencoder.train()
        epoch_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructions, _ = autoencoder(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


    return autoencoder

