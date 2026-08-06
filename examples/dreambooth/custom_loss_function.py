import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    """Flow-matching MSE with velocity-direction and high-frequency residual regularization.

    ``model_pred`` and ``target`` must be velocity fields shaped ``[B, C, H, W]``.
    ``weighting`` contains one flow-matching loss weight per sample.
    """

    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 1.0, hf_weight: float = 1.0):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.cosine_weight = float(cosine_weight)
        self.hf_weight = float(hf_weight)
        self.last_terms = {}

    def forward(self, model_pred: torch.Tensor, target: torch.Tensor, weighting: torch.Tensor) -> torch.Tensor:
        model_pred = model_pred.float()
        target = target.float()
        residual = model_pred - target
        batch_size = target.shape[0]
        sample_weight = weighting.float().reshape(batch_size, -1).mean(dim=1)

        mse_per_sample = residual.square().reshape(batch_size, -1).mean(dim=1)
        mse_loss = (sample_weight * mse_per_sample).mean()

        # ModaFlow Eq. (5), interpreted per sample over the flattened velocity field.
        cosine_per_sample = 1.0 - F.cosine_similarity(
            model_pred.reshape(batch_size, -1),
            target.reshape(batch_size, -1),
            dim=1,
            eps=1e-8,
        )
        cosine_loss = cosine_per_sample.mean()

        # One-level orthonormal Haar transform of the velocity residual. The low-frequency
        # subband is intentionally omitted so this term does not duplicate the full MSE.
        # Right/bottom zero padding matches DWTForward(mode="zero") for odd spatial sizes.
        height, width = residual.shape[-2:]
        residual = F.pad(residual, (0, width % 2, 0, height % 2))
        top_left = residual[..., 0::2, 0::2]
        top_right = residual[..., 0::2, 1::2]
        bottom_left = residual[..., 1::2, 0::2]
        bottom_right = residual[..., 1::2, 1::2]

        high_vertical = (top_left + top_right - bottom_left - bottom_right) / 2.0
        high_horizontal = (top_left - top_right + bottom_left - bottom_right) / 2.0
        high_diagonal = (top_left - top_right - bottom_left + bottom_right) / 2.0
        high_frequency = torch.stack((high_vertical, high_horizontal, high_diagonal), dim=2)

        hf_per_sample = high_frequency.square().reshape(batch_size, -1).mean(dim=1)
        hf_loss = (sample_weight * hf_per_sample).mean()

        loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss + self.hf_weight * hf_loss
        self.last_terms = {
            "mse": mse_loss.detach(),
            "cosine": cosine_loss.detach(),
            "high_frequency": hf_loss.detach(),
            "total": loss.detach(),
        }
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, channels, height, width = 2, 16, 64, 64
    pred = torch.randn(batch_size, channels, height, width, requires_grad=True)
    target = torch.randn_like(pred)
    weighting = torch.tensor([1.0, 2.0]).reshape(batch_size, 1, 1, 1)

    criterion = CustomLoss(mse_weight=1.0, cosine_weight=0.1, hf_weight=0.1)
    loss = criterion(pred, target, weighting)
    terms = {name: value.item() for name, value in criterion.last_terms.items()}
    loss.backward()
    assert loss.ndim == 0
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    odd_pred = torch.randn(1, channels, 9, 11, requires_grad=True)
    odd_loss = criterion(odd_pred, torch.randn_like(odd_pred), torch.ones(1, 1, 1, 1))
    odd_loss.backward()
    assert odd_pred.grad is not None and torch.isfinite(odd_pred.grad).all()

    mse_only = CustomLoss(mse_weight=1.0, cosine_weight=0.0, hf_weight=0.0)
    mse_loss = mse_only(pred.detach(), target, weighting)
    expected_mse = torch.mean((weighting * (pred.detach() - target).square()).reshape(batch_size, -1), dim=1).mean()
    torch.testing.assert_close(mse_loss, expected_mse)

    aligned_target = torch.randn(batch_size, channels, height, width)
    cosine_only = CustomLoss(mse_weight=0.0, cosine_weight=1.0, hf_weight=0.0)
    aligned_cosine_loss = cosine_only(2.0 * aligned_target, aligned_target, torch.ones_like(weighting))
    torch.testing.assert_close(aligned_cosine_loss, torch.zeros_like(aligned_cosine_loss), atol=1e-6, rtol=0.0)

    constant_residual = torch.ones(1, 1, 4, 4)
    checkerboard_residual = torch.tensor([[[[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]] * 2]])
    hf_only = CustomLoss(mse_weight=0.0, cosine_weight=0.0, hf_weight=1.0)
    unit_weight = torch.ones(1, 1, 1, 1)
    constant_hf_loss = hf_only(constant_residual, torch.zeros_like(constant_residual), unit_weight)
    checkerboard_hf_loss = hf_only(checkerboard_residual, torch.zeros_like(checkerboard_residual), unit_weight)
    torch.testing.assert_close(constant_hf_loss, torch.zeros_like(constant_hf_loss))
    assert checkerboard_hf_loss > constant_hf_loss

    print("loss:", loss.item())
    print("terms:", terms)
