import torch
from umud_attention.model import AttentionUNet, AttentionGate
from umud_attention.losses import segmentation_loss


def test_attention_forward_backward_and_plain_control():
    torch.set_num_threads(2)
    for attention in (True, False):
        model = AttentionUNet('resnet18', pretrained=False, attention=attention, decoder_width=8)
        x = torch.randn(2, 3, 65, 97)
        y = torch.zeros(2, 1, 65, 97)
        y[:, :, 20:23, 5:90] = 1
        logits, aux = model(x, return_aux=True)
        assert logits.shape == y.shape == aux.shape
        loss = segmentation_loss(logits, y, torch.ones_like(y))
        loss.backward()
        assert torch.isfinite(loss)
        if attention:
            assert model.dec1.attention.psi.weight.grad.abs().sum() > 0


def test_padding_does_not_change_loss():
    logits = torch.randn(1, 1, 32, 32)
    target = torch.zeros_like(logits)
    target[:, :, 10:12, 8:25] = 1
    valid = torch.ones_like(logits)
    valid[:, :, :4] = 0
    expected = segmentation_loss(logits, target, valid)
    changed = logits.clone()
    changed[:, :, :4] = 100
    assert torch.allclose(expected, segmentation_loss(changed, target, valid))
