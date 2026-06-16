# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedWilsonCowanPINN(nn.Module):
    def __init__(self, num_channels=22, temporal_filters=16, time_steps=500, num_classes=4):
        super(AdvancedWilsonCowanPINN, self).__init__()

        self.temporal_filters = temporal_filters

        # ==========================================
        # 1. MULTI-SCALE TEMPORAL FILTERING
        # ==========================================
        self.temp_conv_mu = nn.Conv1d(in_channels=1,
                                      out_channels=temporal_filters,
                                      kernel_size=65, padding=32, bias=False)

        self.temp_conv_beta = nn.Conv1d(in_channels=1,
                                        out_channels=temporal_filters,
                                        kernel_size=33, padding=16, bias=False)

        self.bn_temp = nn.BatchNorm2d(temporal_filters * 2)

        # ==========================================
        # 2. DEPTHWISE SPATIAL CONVOLUTION (Matches your 78.pth weights!)
        # ==========================================
        self.depthwise_spatial = nn.Conv2d(in_channels=temporal_filters * 2, 
                                           out_channels=temporal_filters * 2,
                                           kernel_size=(num_channels, 1), 
                                           groups=temporal_filters * 2, 
                                           bias=False)

        self.bn_spatial = nn.BatchNorm2d(temporal_filters * 2)
        self.gelu = nn.GELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.spatial_dropout = nn.Dropout2d(p=0.4)

        # ==========================================
        # 3. SQUEEZE-AND-EXCITATION (SE) ATTENTION
        # ==========================================
        self.se_fc1 = nn.Linear(temporal_filters * 2, (temporal_filters * 2) // 4)
        self.se_fc2 = nn.Linear((temporal_filters * 2) // 4, temporal_filters * 2)

        # ==========================================
        # 4. WILSON-COWAN GAT & PHYSICS PARAMETERS
        # ==========================================
        self.attention = nn.MultiheadAttention(embed_dim=temporal_filters * 2, num_heads=4, batch_first=True)

        self.project_E = nn.Linear(temporal_filters * 2, 1)
        self.project_I = nn.Linear(temporal_filters * 2, 1)
        self.project_P = nn.Linear(temporal_filters * 2, 1)
        self.project_Q = nn.Linear(temporal_filters * 2, 1)

        self.w_EE = nn.Parameter(torch.tensor(1.2))
        self.w_EI = nn.Parameter(torch.tensor(1.0))
        self.w_IE = nn.Parameter(torch.tensor(1.0))
        self.w_II = nn.Parameter(torch.tensor(0.5))
        self.tau_E = nn.Parameter(torch.tensor(0.1))
        self.tau_I = nn.Parameter(torch.tensor(0.1))

        # ==========================================
        # 5. ROBUST CLASSIFIER
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(temporal_filters * 2, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, C, T = x.shape

        x = x.unsqueeze(1)

        # 1. Temporal Extraction
        x_flat = x.view(B * C, 1, T)
        x_mu = self.temp_conv_mu(x_flat)
        x_beta = self.temp_conv_beta(x_flat)

        x_temp = torch.cat([x_mu, x_beta], dim=1)
        x_temp = x_temp.view(B, C, self.temporal_filters * 2, T).permute(0, 2, 1, 3)
        x_temp = self.bn_temp(x_temp)

        # 2. Unified Depthwise Spatial Processing over all channels
        x_spat = self.depthwise_spatial(x_temp)

        x_spat = self.bn_spatial(x_spat)
        x_spat = self.gelu(x_spat)
        x_spat = self.avg_pool(x_spat)
        x_spat = self.spatial_dropout(x_spat)

        # 3. Squeeze-and-Excitation Recalibration
        se_weight = x_spat.mean(dim=-1).squeeze(-1)
        se_weight = self.gelu(self.se_fc1(se_weight))
        se_weight = torch.sigmoid(self.se_fc2(se_weight)).unsqueeze(-1).unsqueeze(-1)
        x_spat = x_spat * se_weight

        x_attn = x_spat.squeeze(2).permute(0, 2, 1)

        # 4. GAT Layer
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)

        # Extract Wilson-Cowan States
        E = torch.sigmoid(self.project_E(attn_out))
        I = torch.sigmoid(self.project_I(attn_out))
        P = self.project_P(attn_out)
        Q = self.project_Q(attn_out)

        loss_wc = self.compute_wc_loss(E, I, P, Q)

        # 5. Classification
        final_features = attn_out.mean(dim=1)
        logits = self.classifier(final_features)

        return logits, loss_wc

    def compute_wc_loss(self, E, I, P, Q):
        wEE = F.softplus(self.w_EE)
        wEI = F.softplus(self.w_EI)
        wIE = F.softplus(self.w_IE)
        wII = F.softplus(self.w_II)
        tauE = F.softplus(self.tau_E)
        tauI = F.softplus(self.tau_I)

        dE_dt = E[:, 1:, :] - E[:, :-1, :]
        dI_dt = I[:, 1:, :] - I[:, :-1, :]

        E_t, I_t = E[:, :-1, :], I[:, :-1, :]
        P_t, Q_t = P[:, :-1, :], Q[:, :-1, :]

        E_input = (wEE * E_t) - (wEI * I_t) + P_t
        res_E = (tauE * dE_dt) + E_t - torch.sigmoid(E_input)

        I_input = (wIE * E_t) - (wII * I_t) + Q_t
        res_I = (tauI * dI_dt) + I_t - torch.sigmoid(I_input)

        return torch.mean(res_E**2) + torch.mean(res_I**2)

def get_model(model_name, num_classes=4, **kwargs):
    if model_name == 'bci_sub2a' or model_name == 'pinn':
        return AdvancedWilsonCowanPINN(num_channels=22, time_steps=500, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture request: '{model_name}'")