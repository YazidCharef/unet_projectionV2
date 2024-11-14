import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_CHANNELS, SEQUENCE_LENGTH

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class TemporalUNet(nn.Module):
    def __init__(self, n_channels=INPUT_CHANNELS, n_frames=SEQUENCE_LENGTH):
        super().__init__()
        
        # Initial number of features
        initial_features = 64

        # Entrée : adaptation pour les 5 variables
        self.n_channels = n_channels
        self.n_frames = n_frames

        # Encoder
        self.inc = DoubleConv(n_channels * n_frames, initial_features)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(initial_features, initial_features * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(initial_features * 2, initial_features * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(initial_features * 4, initial_features * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(initial_features * 8, initial_features * 16)
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(initial_features * 16, initial_features * 8, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(initial_features * 16, initial_features * 8)
        
        self.up3 = nn.ConvTranspose2d(initial_features * 8, initial_features * 4, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(initial_features * 8, initial_features * 4)
        
        self.up2 = nn.ConvTranspose2d(initial_features * 4, initial_features * 2, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(initial_features * 4, initial_features * 2)
        
        self.up1 = nn.ConvTranspose2d(initial_features * 2, initial_features, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(initial_features * 2, initial_features)

        # Couche finale
        self.outc = nn.Conv2d(initial_features, n_frames, kernel_size=1)
        
        # Initialisation des poids
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [batch_size, n_channels, n_frames, height, width]
        batch_size, n_channels, n_frames, height, width = x.shape
        
        # Reshape pour traiter la séquence temporelle
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, n_channels * n_frames, height, width)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder avec skip connections
        x = self.up4(x5)
        x = self.upconv4(torch.cat([x4, x], dim=1))
        
        x = self.up3(x)
        x = self.upconv3(torch.cat([x3, x], dim=1))
        
        x = self.up2(x)
        x = self.upconv2(torch.cat([x2, x], dim=1))
        
        x = self.up1(x)
        x = self.upconv1(torch.cat([x1, x], dim=1))

        # Couche finale
        x = self.outc(x)  # [batch_size, n_frames, height, width]

        return x

    def count_parameters(self):
        """Compte le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def get_memory_usage(batch_size=1, height=256, width=256):
        """Estime l'utilisation mémoire pour une taille d'entrée donnée"""
        total_params = sum(p.numel() for p in model.parameters())
        param_size = total_params * 4  # 4 bytes pour float32
        
        # Estimation grossière de la mémoire d'activation
        activation_size = batch_size * height * width * 64 * 4  # Première couche
        
        total_size = param_size + activation_size
        return {
            'parameters_mb': param_size / (1024 * 1024),
            'activation_mb': activation_size / (1024 * 1024),
            'total_mb': total_size / (1024 * 1024)
        }

