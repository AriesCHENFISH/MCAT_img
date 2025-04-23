import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
i =1

class EnhancedMCAT(nn.Module):
    def __init__(self, fusion='gated', model_size_ct='small', model_size_dsa='small', n_classes=1, dropout=0.25):
        super(EnhancedMCAT, self).__init__()
        self.fusion = fusion
        self.n_classes = n_classes

        # 扩展特征维度
        self.size_dict_ct = {"small": [512, 1024, 512], "big": [512, 1024, 768]}
        self.size_dict_dsa = {"small": [512, 1024, 512], "big": [512, 1024, 768]}

        size_ct = self.size_dict_ct[model_size_ct]
        size_dsa = self.size_dict_dsa[model_size_dsa]

        # 更强大的特征提取网络
        self.ct_fc = nn.Sequential(
            nn.Linear(size_ct[0], size_ct[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size_ct[1], size_ct[2]),
            nn.LayerNorm(size_ct[2])
        )

        self.dsa_fc = nn.Sequential(
            nn.Linear(size_dsa[0], size_dsa[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size_dsa[1], size_dsa[2]),
            nn.LayerNorm(size_dsa[2])
        )

        # 增强的交叉注意力机制（多头）
        self.coattn = nn.MultiheadAttention(
            embed_dim=size_ct[2], 
            num_heads=8,  # 增加注意力头数
            dropout=dropout,
            batch_first=True
        )
        
        # 交叉注意力后的残差连接和层归一化
        self.ct_norm = nn.LayerNorm(size_ct[2])
        self.dsa_norm = nn.LayerNorm(size_dsa[2])

        # 更深的Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=size_ct[2], 
            nhead=8, 
            dim_feedforward=1024,  # 增加前馈网络维度
            dropout=dropout, 
            activation='gelu',  # 使用GELU激活函数
            batch_first=True,
            norm_first=True  # 先归一化再计算
        )
        self.ct_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # 增加层数
        self.dsa_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 改进的注意力机制
        self.ct_attention = nn.Sequential(
            nn.Linear(size_ct[2], size_ct[2]//2),
            nn.GELU(),
            nn.Linear(size_ct[2]//2, 1)
        )
        self.dsa_attention = nn.Sequential(
            nn.Linear(size_dsa[2], size_dsa[2]//2),
            nn.GELU(),
            nn.Linear(size_dsa[2]//2, 1)
        )

        # 改进的融合策略
        if self.fusion == 'gated':
            self.gate_ct = nn.Linear(size_ct[2], size_ct[2])
            self.gate_dsa = nn.Linear(size_dsa[2], size_dsa[2])
            self.sigmoid = nn.Sigmoid()
        elif self.fusion == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(size_ct[2] + size_dsa[2], (size_ct[2] + size_dsa[2])//2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear((size_ct[2] + size_dsa[2])//2, size_ct[2]),
                nn.LayerNorm(size_ct[2])
            )
        elif self.fusion == 'bilinear':
            self.fusion_layer = nn.Bilinear(size_ct[2], size_dsa[2], size_ct[2])
        
        # 分类头增强
        self.classifier = nn.Sequential(
            nn.Linear(size_ct[2], size_ct[2]//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size_ct[2]//2, n_classes)
        )

    def forward(self, x_ct, x_dsa):
        # 确保输入维度正确
        x_ct = x_ct.squeeze() if x_ct.dim() > 2 else x_ct
        x_dsa = x_dsa.squeeze() if x_dsa.dim() > 2 else x_dsa
        
        # 特征提取
        h_ct = self.ct_fc(x_ct).unsqueeze(1)      # [B, 1, D]
        h_dsa = self.dsa_fc(x_dsa).unsqueeze(1)   # [B, 1, D]
        
        # 增强的交叉注意力
        h_ct_coattn, coattn_matrix = self.coattn(
            query=h_dsa, 
            key=h_ct, 
            value=h_ct
        )
        h_ct_coattn = self.ct_norm(h_ct + h_ct_coattn)  # 残差连接
        
        h_dsa_coattn, _ = self.coattn(
            query=h_ct, 
            key=h_dsa, 
            value=h_dsa
        )
        h_dsa_coattn = self.dsa_norm(h_dsa + h_dsa_coattn)  # 残差连接
        
        # Transformer编码
        h_ct_trans = self.ct_transformer(h_ct_coattn)
        h_dsa_trans = self.dsa_transformer(h_dsa_coattn)
        
        # 改进的注意力池化
        ct_attention_weights = F.softmax(self.ct_attention(h_ct_trans), dim=1)
        dsa_attention_weights = F.softmax(self.dsa_attention(h_dsa_trans), dim=1)
        
        h_ct_final = torch.sum(ct_attention_weights * h_ct_trans, dim=1)
        h_dsa_final = torch.sum(dsa_attention_weights * h_dsa_trans, dim=1)
        
        # 改进的融合策略
        if self.fusion == 'gated':
            gate = self.sigmoid(self.gate_ct(h_ct_final) + self.gate_dsa(h_dsa_final))
            h_final = gate * h_ct_final + (1 - gate) * h_dsa_final
        elif self.fusion == 'bilinear':
            h_final = self.fusion_layer(h_ct_final, h_dsa_final)
        elif self.fusion == 'concat':
            h_final = self.fusion_layer(torch.cat([h_ct_final, h_dsa_final], dim=1))
        else:  # 默认求和
            h_final = h_ct_final + h_dsa_final
        
        # 分类
        logits = self.classifier(h_final)
        probs = torch.sigmoid(logits)
        
        return probs, coattn_matrix