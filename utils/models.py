# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class WCEEGNetPINN(nn.Module):
    def __init__(self, num_channels=22, F1=8, D=2, num_classes=4, fs=250, pool_factor=8):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.fs = fs
        self.pool_factor = pool_factor
        self.dt_phys = pool_factor / fs

        assert D == 2, "D must be 2 to split spatial features into Excitatory and Inhibitory pairs."

        # ==========================================
        # 1. MULTI-SCALE TEMPORAL BRANCHES
        # ==========================================
        # Branch 1: Long window (0.5s) for mu-rhythm
        self.temp_conv_mu = nn.Conv2d(1, F1, kernel_size=(1, fs // 2), padding='same', bias=False)
        # Branch 2: Short window (0.25s) for beta-rhythm
        self.temp_conv_beta = nn.Conv2d(1, F1, kernel_size=(1, fs // 4), padding='same', bias=False)

        # We now have 2 * F1 temporal filters
        F_multi = F1 * 2

        self.bn1 = nn.BatchNorm2d(F_multi)

        # Depthwise scales to the new F_multi dimension
        self.depthwise = nn.Conv2d(F_multi, F_multi * D, kernel_size=(num_channels, 1), groups=F_multi, bias=False)
        self.bn2 = nn.BatchNorm2d(F_multi * D)
        self.activation = nn.ELU()

        # ==========================================
        # 2. SCALED PHYSICS PARAMETERS
        # ==========================================
        # The ODE now simulates F_multi (16) populations instead of 8
        self.tau_E = nn.Parameter(torch.full((1, F_multi, 1), 0.01))
        self.tau_I = nn.Parameter(torch.full((1, F_multi, 1), 0.01))
        self.w_EE = nn.Parameter(torch.full((1, F_multi, 1), 1.2))
        self.w_EI = nn.Parameter(torch.full((1, F_multi, 1), 1.0))
        self.w_IE = nn.Parameter(torch.full((1, F_multi, 1), 1.0))
        self.w_II = nn.Parameter(torch.full((1, F_multi, 1), 0.5))
        self.P = nn.Parameter(torch.zeros(1, F_multi, 1))
        self.Q = nn.Parameter(torch.zeros(1, F_multi, 1))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        # FUSED CLASSIFIER: Adjusted for F_multi
        fused_feature_dim = (F_multi * D) + F_multi + F_multi
        self.classifier = nn.Linear(fused_feature_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_wc_loss(self, E, I):
        tE, tI = F.softplus(self.tau_E), F.softplus(self.tau_I)
        wEE, wEI = F.softplus(self.w_EE), F.softplus(self.w_EI)
        wIE, wII = F.softplus(self.w_IE), F.softplus(self.w_II)

        dE_dt = (E[:, :, 1:] - E[:, :, :-1]) / self.dt_phys
        dI_dt = (I[:, :, 1:] - I[:, :, :-1]) / self.dt_phys
        E_t, I_t = E[:, :, :-1], I[:, :, :-1]

        drv_E = wEE * E_t - wEI * I_t + self.P
        drv_I = wIE * E_t - wII * I_t + self.Q

        res_E = tE * dE_dt + E_t - torch.sigmoid(drv_E)
        res_I = tI * dI_dt + I_t - torch.sigmoid(drv_I)

        return torch.mean(res_E ** 2) + torch.mean(res_I ** 2)

    def forward(self, x, fingerprint_mode=False):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # 1. Parallel Temporal Extraction
        x_mu = self.temp_conv_mu(x)
        x_beta = self.temp_conv_beta(x)

        # Concatenate along the channel dimension (dim=1)
        x_concat = torch.cat([x_mu, x_beta], dim=1)

        # 2. Main Spatial Extraction
        x_norm = self.bn1(x_concat)
        x_main = self.activation(self.bn2(self.depthwise(x_norm)))

        # 3. Physics Branch
        x_phys = F.avg_pool1d(x_main.squeeze(2), self.pool_factor)

        # Split into Excitatory and Inhibitory (using the dynamically scaled F_multi)
        F_multi = self.F1 * 2
        E_phys = torch.sigmoid(x_phys[:, :F_multi, :])
        I_phys = torch.sigmoid(x_phys[:, F_multi:, :])

        # 4. Feature Fusion
        E_feat = E_phys.mean(dim=-1)
        I_feat = I_phys.mean(dim=-1)
        x_pool = torch.flatten(self.avg_pool(x_main), 1)

        fused_features = self.dropout(torch.cat([x_pool, E_feat, I_feat], dim=-1))
        logits = self.classifier(fused_features)

        # ADV-TRA Routing Escape Hatch
        if fingerprint_mode:
            return logits

        loss_wc = self.compute_wc_loss(E_phys, I_phys)
        return logits, loss_wc

def get_model(model_name, num_classes=4, **kwargs):
    return WCEEGNetPINN(num_channels=22, F1=8, D=2, num_classes=num_classes, fs=250, pool_factor=8)
