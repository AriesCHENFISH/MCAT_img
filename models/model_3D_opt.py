import sys
sys.path.append('models')

import torch
import torch.nn as nn
import torch.nn.functional as F
class MCAT_Surv(nn.Module):
    def __init__(self, fusion='concat', model_size_ct: str='small', model_size_dsa: str='small', n_classes=1, dropout=0.25):
        super(MCAT_Surv, self).__init__()
        self.fusion = fusion
        self.n_classes = n_classes
        self.ct_norm = nn.LayerNorm(512)


        self.size_dict_ct = {"small": [512, 512], "big": [512, 384]}
        self.size_dict_dsa = {'small': [512, 512], "big": [512, 384]}

        # FC for CT features
        size_ct = self.size_dict_ct[model_size_ct]
        self.ct_fc = nn.Sequential(
            nn.Linear(size_ct[0], size_ct[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # FC for DSA features
        size_dsa = self.size_dict_dsa[model_size_dsa]
        self.dsa_fc = nn.Sequential(
            nn.Linear(size_dsa[0], size_dsa[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multihead Attention
        self.coattn = nn.MultiheadAttention(embed_dim=512, num_heads=1, batch_first=False)

        # Transformer Encoders
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.ct_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dsa_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Attention heads
        self.ct_attention = nn.Linear(512, 1)
        self.dsa_attention = nn.Linear(512, 1)

        # Fusion Layer
        if self.fusion == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(512 * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
            )
        elif self.fusion == 'bilinear':
            self.fusion_layer = nn.Bilinear(512, 512, 512)
        else:
            self.fusion_layer = None

        # Binary Classification
        self.classifier = nn.Linear(512, 1)

    def forward(self, x_ct, x_dsa):
        

        # print(f"h_ct fc: {x_ct.shape}")
        # print(f"h_dsa fc: {x_dsa.shape}")

        # CT feature 
        h_ct = self.ct_fc(x_ct.squeeze(0))  
        # h_ct = h_ct.squeeze()
        # print(f"h_ct shape after fc: {h_ct.shape}")

        # DSA feature
        h_dsa = self.dsa_fc(x_dsa.squeeze(0))  
        # print(f"h_dsa shape after fc: {h_dsa.shape}")

        h_dsa = h_dsa.unsqueeze(0)  # (1, 1, 512)
        h_ct = h_ct.unsqueeze(0)
        

        # print(f"h_ct shape: {h_ct.shape}, h_dsa shape: {h_dsa.shape}")

        # Co-Attention 
        h_ct2dsa, coattn_matrix = self.coattn(h_ct, h_dsa, h_dsa)  
        h_ct = self.ct_norm(h_ct2dsa + h_ct)
        # print(f"h_ct after coattn: {h_ct.shape}")

        # Transformer Encoding 
        h_ct_trans = self.ct_transformer(h_ct.permute(1, 0, 2))  
        h_dsa_trans = self.dsa_transformer(h_dsa.permute(1, 0, 2))  
        # print(f"h_ct_trans shape: {h_ct_trans.shape}, h_dsa_trans shape: {h_dsa_trans.shape}")

        # Attention Pooling
        ct_attention_weights = F.softmax(self.ct_attention(h_ct_trans), dim=1)  
        dsa_attention_weights = F.softmax(self.dsa_attention(h_dsa_trans), dim=1)  

        h_ct_final = torch.sum(ct_attention_weights * h_ct_trans, dim=1)  
        h_dsa_final = torch.sum(dsa_attention_weights * h_dsa_trans, dim=1)  

        # print(f"h_ct_final shape: {h_ct_final.shape}, h_dsa_final shape: {h_dsa_final.shape}")

        # Fusion
        if self.fusion == 'bilinear':
            h_final = self.fusion_layer(h_ct_final, h_dsa_final) 
        elif self.fusion == 'concat':
            h_final = self.fusion_layer(torch.cat([h_ct_final, h_dsa_final], dim=1))  
        else:
            h_final = h_ct_final + h_dsa_final 

        # print(f"h_final shape: {h_final.shape}")

        # Classification
        logits = self.classifier(h_final)  
        probs = torch.sigmoid(logits)  

        return probs, coattn_matrix

