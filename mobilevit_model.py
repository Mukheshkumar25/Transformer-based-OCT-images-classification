import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=p, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden_dim, k=1, s=1, p=0))
        else:
            hidden_dim = in_ch

        layers.append(ConvBNAct(hidden_dim, hidden_dim, k=3, s=stride, groups=hidden_dim))

        layers.append(nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1,
                                padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = x + out
        return out


class MobileViTBlock(nn.Module):
    def __init__(self, in_ch, transformer_dim, ffn_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.local_rep = nn.Sequential(
            ConvBNAct(in_ch, in_ch, k=3, s=1),
            ConvBNAct(in_ch, transformer_dim, k=1, s=1, p=0),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fusion = nn.Sequential(
            ConvBNAct(transformer_dim, in_ch, k=1, s=1, p=0),
            ConvBNAct(in_ch, in_ch, k=3, s=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        y = self.local_rep(x)         # [B, Ct, H, W]
        B, Ct, Ht, Wt = y.shape

        y = y.flatten(2).transpose(1, 2)  # [B, H*W, Ct]
        y = self.transformer(y)
        y = y.transpose(1, 2).reshape(B, Ct, Ht, Wt)

        y = self.fusion(y)
        return y


class MobileViT_XS(nn.Module):
    def __init__(self, num_classes=4, img_size=128):
        super().__init__()

        self.stem = ConvBNAct(3, 16, k=3, s=2)  # 128 -> 64

        self.layer1 = nn.Sequential(
            InvertedResidual(16, 24, stride=1, expand_ratio=2),
            InvertedResidual(24, 24, stride=1, expand_ratio=2),
        )

        self.layer2 = nn.Sequential(
            InvertedResidual(24, 48, stride=2, expand_ratio=2),  # 64 -> 32
        )
        self.mvit_block1 = MobileViTBlock(48, transformer_dim=64, ffn_dim=128)

        self.layer3 = nn.Sequential(
            InvertedResidual(48, 64, stride=2, expand_ratio=2),  # 32 -> 16
        )
        self.mvit_block2 = MobileViTBlock(64, transformer_dim=80, ffn_dim=160)

        self.layer4 = nn.Sequential(
            InvertedResidual(64, 80, stride=2, expand_ratio=2),  # 16 -> 8
        )
        self.mvit_block3 = MobileViTBlock(80, transformer_dim=96, ffn_dim=192)

        self.conv_head = ConvBNAct(80, 128, k=1, s=1, p=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.mvit_block1(x)

        x = self.layer3(x)
        x = self.mvit_block2(x)

        x = self.layer4(x)
        x = self.mvit_block3(x)

        x = self.conv_head(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
