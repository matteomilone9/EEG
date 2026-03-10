import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F

import math


### 1S ###
class NS_1(nn.Module):
    def __init__(self, sample_rate=16000, segment_duration=1):
        super(NS_1, self).__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = segment_duration
        self.frames = 5
        self.n_mels = 128
        self.t_bins = 1 + (self.duration_seconds * self.sample_rate) // 512
        self.vector_array_size = self.t_bins - self.frames + 1
        self.transform_tf = T.MelSpectrogram(sample_rate=self.sample_rate,
                                                n_fft=1024,
                                                win_length=1024,
                                                hop_length=512,
                                                center=True,
                                                pad_mode="reflect",
                                                power=2.0,
                                                norm="slaney",
                                                n_mels=self.n_mels,
                                                mel_scale="htk",)

        self.encoder = nn.Sequential(
            nn.Linear(in_features = 640, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features = 128, out_features = 8),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features = 8, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 640)
        )

        # WEIGHTS INIT LIKE KERAS
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.xavier_uniform_(self.encoder[3].weight)
        nn.init.xavier_uniform_(self.encoder[6].weight)
        nn.init.xavier_uniform_(self.encoder[9].weight)
        nn.init.xavier_uniform_(self.bottleneck[0].weight)
        nn.init.xavier_uniform_(self.decoder[0].weight)
        nn.init.xavier_uniform_(self.decoder[3].weight)
        nn.init.xavier_uniform_(self.decoder[6].weight)
        nn.init.xavier_uniform_(self.decoder[9].weight)
        nn.init.xavier_uniform_(self.decoder[-1].weight)

        # BIAS INIT LIKE KERAS
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.zeros_(self.encoder[3].bias)
        nn.init.zeros_(self.encoder[6].bias)
        nn.init.zeros_(self.encoder[9].bias)
        nn.init.zeros_(self.bottleneck[0].bias)
        nn.init.zeros_(self.decoder[0].bias)
        nn.init.zeros_(self.decoder[3].bias)
        nn.init.zeros_(self.decoder[6].bias)
        nn.init.zeros_(self.decoder[9].bias)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x):
        x = self.preprocessing(x)
        original = x
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return original, x

    def preprocessing(self, x):
        # compute mel spectrogram
        batch_size = x.size()[0]
        x = self.transform_tf(x)  # shape: (B, n_mels, t_bins)

        # Convert to decibels (dB)
        x = 10 * torch.log10(x + 1e-8)

        # Normalize dB range from [-80, 0] → [0, 1]
        x = (x + 80) / 80
        x = torch.clamp(x, 0.0, 1.0)

        # Frame stacking (vectorize mel frames into context windows)
        vector_dim = self.frames * self.n_mels
        feature_vector = torch.zeros((batch_size, self.vector_array_size, vector_dim), device=x.device)

        for batch in range(batch_size):
            for i in range(self.frames):
                feature_vector[batch, :, self.n_mels * i: self.n_mels * (i + 1)] = \
                    x[batch, :, i: i + self.vector_array_size].T

        return feature_vector


### VAR AUTOENCODER + TRANSFORMER ###
class VAT_Anomaly(nn.Module):
    def __init__(self, sample_rate=16000, segment_duration=1):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = segment_duration
        self.frames=5
        self.n_mels = 128
        self.t_bins = 1 + (self.duration_seconds * self.sample_rate) // 512
        self.vector_array_size = self.t_bins - self.frames + 1
        self.vector_dim = self.frames * self.n_mels

        ###=== MEL SPECTROGRAM ===###
        # -----------------
        self.transform_tf = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="htk",
        )

        ###=== ENCODER ===###
        # -----------------
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.vector_dim, 256),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

    
        ###=== TRANSFORMER BLOCK ===###
        # -----------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)


        ###=== VAE bottleneck ===###
        # -----------------
        latent_dim = 16
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        
        ###=== DECODER  ===###
        # -----------------
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.vector_dim)
        )

    ###=== Forward ===###
    def forward(self, x):
        x = self.preprocessing(x)
        original = x  # for reconstruction loss

        # Encoder
        x = self.encoder_fc(x)               # (B, T, 128)
        x = self.transformer(x)              # (B, T, 128)

        # Bottleneck (mean & logvar per frame)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                   # (B, T, latent_dim)

        # Decode
        recon = self.decoder_fc(z)           # (B, T, vector_dim)
        return original, recon, mu, logvar

    def preprocessing(self, x):
        batch_size = x.size(0)
        x = self.transform_tf(x)  # (B, n_mels, t_bins)
        x = 10 * torch.log10(x + 1e-8)
        x = (x + 80) / 80
        x = torch.clamp(x, 0.0, 1.0)

        # Frame stacking
        feature_vector = torch.zeros(
            (batch_size, self.vector_array_size, self.vector_dim), device=x.device
        )
        for b in range(batch_size):
            for i in range(self.frames):
                feature_vector[b, :, self.n_mels * i:self.n_mels * (i + 1)] = \
                    x[b, :, i:i + self.vector_array_size].T
        return feature_vector


### VAR AUTOENCODER + SPECT LINEARE ###
class VAT_Anomaly_Linear(nn.Module):
    def __init__(self, sample_rate=192000, segment_duration=1):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = segment_duration
        self.frames = 5
        self.n_fft = 2048
        self.hop_length = 1024
        self.n_bins = self.n_fft // 2 + 1  # spettrogramma lineare
        self.t_bins = 1 + (self.duration_seconds * self.sample_rate) // self.hop_length
        self.vector_array_size = self.t_bins - self.frames + 1
        self.vector_dim = self.frames * self.n_bins

        ###=== SPETTROGRAMMA LINEARE ===###
        self.transform_tf = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            normalized=True,
        )

        ###=== ENCODER ===###
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.vector_dim, 256),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

        ###=== TRANSFORMER BLOCK ===###
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        ###=== VAE bottleneck ===###
        latent_dim = 32
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        ###=== DECODER ===###
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.vector_dim)
        )

    ###=== Forward ===###
    def forward(self, x):
        x = self.preprocessing(x)
        original = x  # per loss di ricostruzione

        # Encoder
        x = self.encoder_fc(x)       # (B, T, 128)
        x = self.transformer(x)      # (B, T, 128)

        # Bottleneck (mean & logvar per frame)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std           # (B, T, latent_dim)

        # Decode
        recon = self.decoder_fc(z)   # (B, T, vector_dim)
        return original, recon, mu, logvar

    ###=== Preprocessing ===###
    def preprocessing(self, x):
        batch_size = x.size(0)
        x = self.transform_tf(x)  # (B, n_bins, t_bins)
        x = 10 * torch.log10(x + 1e-8)
        x = (x + 80) / 80
        x = torch.clamp(x, 0.0, 1.0)

        # Frame stacking
        feature_vector = torch.zeros(
            (batch_size, self.vector_array_size, self.vector_dim), device=x.device
        )
        for b in range(batch_size):
            for i in range(self.frames):
                feature_vector[b, :, self.n_bins*i:self.n_bins*(i+1)] = \
                    x[b, :, i:i + self.vector_array_size].T
        return feature_vector

#VERSIONE ORIGINALE vecchia
###=== CNN + LINEAR SPECT ===###
class CNN_VAT_Anomaly_ADV(nn.Module):
    def __init__(self, sample_rate=192000, segment_duration=1, n_fft=2048, hop_length=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = segment_duration
        self.n_fft = n_fft
        self.hop_length = hop_length  
        self.n_bins = self.n_fft // 2 + 1
        self.t_bins = 1 + (self.duration_seconds * self.sample_rate) // self.hop_length

        self.latent_dim = 32
        self.d_model = 256  # Dimensione embedding per Transformer

        ###=== SPETTROGRAMMA LINEARE ===###
        self.transform_tf = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            normalized=True,
        )

        ###=== CNN FRONTEND OTTIMIZZATA ===###
        self.cnn_frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,1), padding=(1,1)),  # kernel più piccolo
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,3), stride=(4,2), padding=(2,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((8, None))  # riduce freq a 8 bande

        ###=== PROIEZIONE PER TRANSFORMER ===###
        self.proj = nn.Linear(64*8, self.d_model)

        ###=== POSIZIONAL ENCODING ===###
        self.pos_encoding = PositionalEncoding(d_model=self.d_model, max_len=self.t_bins)

        ###=== TRANSFORMER BLOCK ===###
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        ###=== VAE bottleneck ===###
        self.fc_mu = nn.Linear(self.d_model, self.latent_dim)
        self.fc_logvar = nn.Linear(self.d_model, self.latent_dim)

        ###=== DECODER ===###
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 64*8),  # ricostruzione feature CNN
        )

    ###=== Forward ===###
    def forward(self, x):
        x = self.preprocessing(x)
        original = x.clone()

        x = self.proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        recon = self.decoder_fc(z)
        return original, recon, mu, logvar



    ###=== Preprocessing con CNN ===###
    # def preprocessing(self, x):
    #     batch_size = x.size(0)
    #     x = self.transform_tf(x)          # (B, F, T)
    #     x = 10 * torch.log10(x + 1e-8)
    #     x = torch.clamp((x + 80) / 80, 0.0, 1.0)
    #     x = x.unsqueeze(1)                # (B, 1, F, T)

    #     x = self.cnn_frontend(x)          # (B, C=64, F', T')
    #     x = self.freq_pool(x)             # (B, 64, 8, T')
    #     B, C, Fp, Tp = x.shape
    #     x = x.view(B, C*Fp, Tp).permute(0, 2, 1)  # (B, T', feature_dim=512)
    #     return x


    ###=== Preprocessing con CNN ===###
    def preprocessing(self, x):
        batch_size = x.size(0)

        # 1️⃣ Spettrogramma lineare
        x = self.transform_tf(x)  # (B, F, T)
        # 2️⃣ Conversione in dB
        x = 10 * torch.log10(x + 1e-8)  # evita log(0)
        # 3️⃣ Aggiungiamo canale per CNN
        x = x.unsqueeze(1)  # (B, 1, F, T)

        # 4️⃣ CNN frontend
        x = self.cnn_frontend(x)  # (B, C=64, F', T')
        x = self.freq_pool(x)     # (B, 64, 8, T')

        # 5️⃣ Rimodellamento per Transformer
        B, C, Fp, Tp = x.shape
        x = x.view(B, C*Fp, Tp).permute(0, 2, 1)  # (B, T', feature_dim=512)

        return x
        

###=== Positional Encoding standard ===###
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
