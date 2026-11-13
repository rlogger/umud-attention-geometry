"""ResNet encoder + additive skip attention, inspired by Oktay et al. (2018).

This is an independent 2D implementation, not the original authors' weights.
Two separately trained instances predict aponeurosis and fascicle masks.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


def _groups(channels):
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout), nn.SiLU(inplace=True))


class AttentionGate(nn.Module):
    """alpha=sigmoid(psi(ReLU(W_x(skip)+W_g(decoder)))); output=alpha*skip."""
    def __init__(self, skip_channels, gate_channels, inter_channels):
        super().__init__()
        self.skip = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.gate = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1)
        # Initially pass most of the skip; avoid erasing thin structures early.
        nn.init.constant_(self.psi.bias, 1.0)

    def forward(self, skip, gate):
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(self.skip(skip) + self.gate(gate))))
        return skip * alpha


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout, attention):
        super().__init__()
        self.attention = AttentionGate(skip, cin, max(cout // 2, 8)) if attention else None
        self.conv = ConvBlock(cin + skip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attention(skip, x) if self.attention else skip], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True, attention=True, decoder_width=32):
        super().__init__()
        choices = {"resnet18": (resnet18, ResNet18_Weights.DEFAULT),
                   "resnet34": (resnet34, ResNet34_Weights.DEFAULT)}
        if encoder not in choices:
            raise ValueError(f"Encoder must be one of {list(choices)}")
        ctor, weights = choices[encoder]
        base = ctor(weights=weights if pretrained else None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool, self.enc1, self.enc2, self.enc3, self.enc4 = (
            base.maxpool, base.layer1, base.layer2, base.layer3, base.layer4)
        w = decoder_width
        self.dec4 = DecoderBlock(512, 256, w * 8, attention)
        self.dec3 = DecoderBlock(w * 8, 128, w * 4, attention)
        self.dec2 = DecoderBlock(w * 4, 64, w * 2, attention)
        self.dec1 = DecoderBlock(w * 2, 64, w, attention)
        self.refine = ConvBlock(w, w)
        self.head = nn.Conv2d(w, 1, 1)
        self.aux = nn.Conv2d(w * 2, 1, 1)

    def forward(self, image, return_aux=False):
        shape = image.shape[-2:]
        s = self.stem(image)
        e1 = self.enc1(self.pool(s))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, s)
        logits = self.head(self.refine(F.interpolate(d1, size=shape, mode="bilinear", align_corners=False)))
        if return_aux:
            return logits, F.interpolate(self.aux(d2), size=shape, mode="bilinear", align_corners=False)
        return logits

    def freeze_encoder_batchnorm(self):
        # Small ultrasound batches: keep ImageNet running statistics stable.
        for block in (self.stem, self.enc1, self.enc2, self.enc3, self.enc4):
            for layer in block.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
