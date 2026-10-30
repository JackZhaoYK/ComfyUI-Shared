"""Stable, opt-in workflow adapter for the existing bounded HSV calibrator.

The older ComfyUI-Change-Color package registers a different RGB-matrix node
under the same MaskedAreaColorAdjuster name. Use a unique class identifier so
worker import order cannot select that older repaint algorithm by accident.
"""
import math
import torch
from .nodes import MaskedAreaColorAdjuster


class CarWrapGentleHSVCalibration(MaskedAreaColorAdjuster):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "mask": ("MASK",),
            "enabled": ("BOOLEAN", {"default": False}),
            "target_hue": ("FLOAT", {"default": 0, "min": 0, "max": 360}),
            "target_saturation": ("FLOAT", {"default": 50, "min": 0, "max": 100}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "calibrate"
    CATEGORY = "custom"

    def calibrate(self, image, mask, enabled, target_hue, target_saturation):
        if not enabled:
            return (image,)
        if not (math.isfinite(target_hue) and 0 <= target_hue <= 360
                and math.isfinite(target_saturation) and 0 <= target_saturation <= 100):
            raise ValueError("Target hue must be 0..360 and saturation 0..100")
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Expected BHWC RGB image")
        if mask.ndim not in (2, 3) or (mask.ndim == 3 and mask.shape[0] not in (1, image.shape[0])):
            raise ValueError("Mask batch must be one or match the image batch")
        # Keep the existing algorithm and protection defaults. Saturation is
        # a proportional adjustment (0.90x..1.10x), not an additive offset.
        result = self.adjust(image.detach().float(), mask.detach().float(),
                             target_hue, target_saturation / 100.0,
                             max_hue_shift=8.0, max_saturation_shift=0.10, strength=1.0)
        adjusted = result[0].to(device=image.device, dtype=image.dtype)
        effective_mask = result[5].to(device=image.device)
        # Avoid even HSV round-trip rounding outside the effective selection.
        original = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        return (torch.where(effective_mask[..., None] > 0, adjusted, original),)
