"""Exercise the real calibration classes without importing YOLO/ComfyUI.

Extract their class definitions from the source AST; the numerical methods are
unmodified. This keeps unit checks independent of GPU models and worker setup.
"""
import ast
import math
from pathlib import Path
import unittest
import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
namespace = dict(np=np, cv2=cv2, torch=torch, math=math)
for file, name in [("nodes.py", "MaskedAreaColorAdjuster"),
                   ("render_calibration.py", "CarWrapGentleHSVCalibration")]:
    tree = ast.parse((root / file).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(root / file), "exec"), namespace)
Calibrator = namespace["CarWrapGentleHSVCalibration"]


def fixture(hue=359, saturation=0.6):
    hsv = np.zeros((96, 128, 3), np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = saturation
    hsv[..., 2] = np.linspace(0.3, 0.7, 96)[:, None]
    return torch.from_numpy(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))[None]


class RenderCalibrationTest(unittest.TestCase):
    def test_off_is_exact_identity(self):
        image = fixture()
        self.assertIs(image, Calibrator().calibrate(image, torch.ones(96, 128), False, 180, 10)[0])

    def test_small_target_across_red_wrap_moves_in_shortest_direction(self):
        image = fixture()
        output = Calibrator().calibrate(image, torch.ones(96, 128), True, 3, 65)[0]
        before = cv2.cvtColor(image[0].numpy(), cv2.COLOR_RGB2HSV)
        after = cv2.cvtColor(output[0].numpy(), cv2.COLOR_RGB2HSV)
        delta = (after[..., 0] - before[..., 0] + 180) % 360 - 180
        self.assertGreater(float(delta[48, 64]), 0.1)
        self.assertLessEqual(float(np.max(delta)), 4.001)
        self.assertGreater(float(after[48, 64, 1]), float(before[48, 64, 1]))
        np.testing.assert_allclose(after[..., 2], before[..., 2], atol=2e-6)

    def test_bounded_saturation_ratio_and_hue_preserve_shading_and_background(self):
        for target_hue, target_sat in [(180, 0), (240, 100)]:
            image = fixture(30)
            original = image.clone()
            mask = torch.zeros(96, 128); mask[5:91, 8:120] = 1
            output = Calibrator().calibrate(image, mask, True, target_hue, target_sat)[0]
            before, after = [cv2.cvtColor(t[0].numpy(), cv2.COLOR_RGB2HSV) for t in (image, output)]
            delta = (after[..., 0] - before[..., 0] + 180) % 360 - 180
            self.assertLessEqual(float(np.max(np.abs(delta))), 8.001)
            self.assertGreater(float(np.max(np.abs(delta))), 0.5)
            ratio = after[..., 1] / before[..., 1]
            self.assertGreaterEqual(float(ratio.min()), .89999)
            self.assertLessEqual(float(ratio.max()), 1.10001)
            self.assertTrue(torch.equal(output[0][mask == 0], image[0][mask == 0]))
            self.assertTrue(torch.equal(image, original))
            np.testing.assert_allclose(after[..., 2], before[..., 2], atol=2e-6)

    def test_empty_mask_gray_and_small_colored_badge_are_noops(self):
        for image, mask in [(fixture(), torch.zeros(96, 128)), (fixture(0, 0), torch.ones(96, 128))]:
            self.assertTrue(torch.equal(image, Calibrator().calibrate(image, mask, True, 60, 80)[0]))
        gray = fixture(0, 0)
        gray[:, 40:44, 50:54] = fixture(0)[0, 40:44, 50:54]
        self.assertTrue(torch.equal(gray, Calibrator().calibrate(gray, torch.ones(96, 128), True, 60, 80)[0]))

    def test_incoherent_two_tone_wrap_does_not_guess(self):
        hsv = np.zeros((96, 128, 3), np.float32)
        hsv[..., 0] = 300; hsv[:, 64:, 0] = 120
        hsv[..., 1] = .7; hsv[..., 2] = .35
        image = torch.from_numpy(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))[None]
        self.assertTrue(torch.equal(image, Calibrator().calibrate(image, torch.ones(96, 128), True, 10, 80)[0]))

    def test_batch_resize_and_invalid_inputs(self):
        images = torch.cat([fixture(30), fixture(210)])
        masks = torch.ones(2, 48, 64); masks[1] = 0
        result = Calibrator().calibrate(images, masks, True, 90, 70)[0]
        self.assertEqual(result.shape, images.shape)
        self.assertEqual(result.dtype, images.dtype)
        self.assertTrue(torch.equal(result[1], images[1]))
        self.assertFalse(torch.equal(result[0], images[0]))
        for hue, sat in [(float('nan'), 50), (20, 101), (-1, 50)]:
            with self.assertRaises(ValueError):
                Calibrator().calibrate(images, masks, True, hue, sat)
        with self.assertRaises(ValueError):
            Calibrator().calibrate(images, torch.ones(3, 96, 128), True, 90, 70)


if __name__ == '__main__':
    unittest.main()
