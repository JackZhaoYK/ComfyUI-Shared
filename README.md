# ComfyUI-Extracted-Nodes

Custom ComfyUI nodes for vehicle detection, image cropping/placement, and masked blending. Designed for car image processing pipelines.

## Nodes

### YOLO Car

Detects the largest vehicle (car/truck/bus) in an image using YOLOv8-seg, crops and optionally resizes it to an optimal resolution.

**Inputs:**

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Input image containing a vehicle |
| use_resolution_matching | BOOLEAN | If True, resizes crop to nearest optimal resolution from a preset list. If False, returns raw bounding box crop. Default: True |

**Outputs:**

| Name | Type | Description |
|------|------|-------------|
| mask | MASK | Segmentation mask of the detected vehicle |
| image | IMAGE | Cropped (and optionally resized) vehicle image |
| x1 | INT | Left coordinate of the crop region |
| y1 | INT | Top coordinate of the crop region |
| x2 | INT | Right coordinate of the crop region |
| y2 | INT | Bottom coordinate of the crop region |
| debug_image | IMAGE | Original image with all detection bounding boxes drawn |

**Notes:**
- Uses YOLOv8m-seg model for instance segmentation
- Automatically tries RGB and BGR color spaces if initial detection fails
- When `use_resolution_matching` is enabled, the output is resized to the closest aspect ratio from a list of preferred resolutions (useful for diffusion model inputs)
- Adds 50px padding around the detected vehicle bounding box

---

### Place Cropped Image Back

Places a cropped/processed image back onto its original position in a background image using pixel coordinates.

**Inputs:**

| Name | Type | Description |
|------|------|-------------|
| background_image | IMAGE | The original full-size background image |
| cropped_image | IMAGE | The processed crop to place back |
| x1 | INT | Left coordinate of target region |
| y1 | INT | Top coordinate of target region |
| x2 | INT | Right coordinate of target region |
| y2 | INT | Bottom coordinate of target region |

**Outputs:**

| Name | Type | Description |
|------|------|-------------|
| IMAGE | IMAGE | Background image with the cropped image placed at the specified coordinates |

**Notes:**
- Automatically resizes the cropped image to fit the target rectangle (x1,y1)-(x2,y2)
- Uses Lanczos resampling for high-quality resize
- Coordinates are clamped to image boundaries

---

### Apply Masked Blend

Blends an input image onto a background image using a mask with smooth alpha blending.

**Inputs:**

| Name | Type | Description |
|------|------|-------------|
| background_image | IMAGE | The base background image |
| input_image | IMAGE | The image to blend on top |
| mask | MASK | Blend mask (0 = background, 1 = input). Supports soft/feathered edges. |
| blend_percentage | FLOAT | Global blend strength (0.0-1.0). Default: 1.0 |

**Outputs:**

| Name | Type | Description |
|------|------|-------------|
| IMAGE | IMAGE | The final blended result |

**Notes:**
- Supports soft masks with feathered edges for smooth transitions
- Automatically resizes input image and mask to match background dimensions
- `blend_percentage` scales the entire mask, useful for fading the effect

---

## Typical Workflow

```
Load Image -> YOLO Car -> [process vehicle crop] -> Place Cropped Image Back -> Apply Masked Blend -> Save Image
```

1. **YOLO Car** detects and crops the vehicle from the input image, outputting coordinates
2. Process the cropped vehicle (e.g., color correction, inpainting, style transfer)
3. **Place Cropped Image Back** puts the processed crop back at the original coordinates
4. **Apply Masked Blend** uses the vehicle segmentation mask to smoothly blend the result with the original background

---

### Masked Area Color Adjuster

Nudges the colour of the masked (car paint) area toward a target hue/saturation by a **bounded** amount. This is a calibration pass, not a repaint: it measures the colour actually present in the paint area, works out the delta to the target, clamps it, and applies it. Every pixel keeps its own shading — highlights, reflections and shadow structure survive, and **Value is never touched**.

**Inputs:**

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Image to calibrate |
| mask | MASK | Car-paint mask (e.g. from YOLO Car) |
| target_hue | FLOAT | Target hue in degrees, 0-360 |
| target_saturation | FLOAT | Target saturation, 0-1 |
| max_hue_shift | FLOAT | Hard cap on the hue rotation, degrees. Default 8 |
| max_saturation_shift | FLOAT | Hard cap on the saturation change, as a fraction (0.10 = +/-10%). Default 0.10 |
| strength | FLOAT | Overall blend, 0-1. Default 1.0 |
| chroma_floor | FLOAT | *(optional)* Pixels below this saturation are excluded from measurement. Default 0.10 |
| min_chroma_coverage | FLOAT | *(optional)* Chromatic pixels must be at least this fraction of the sampled paint, else no calibration. Default 0.15 |
| min_hue_concentration | FLOAT | *(optional)* Minimum circular resultant length R; below this the hue is judged incoherent and nothing is applied. Default 0.50 |
| filter_highlights | BOOLEAN | *(optional)* Exclude highlights/shadows from the measurement. Default True |
| highlight_threshold | FLOAT | *(optional)* Upper luminance bound for sampling. Default 0.37 |
| shadow_threshold | FLOAT | *(optional)* Lower luminance bound for sampling. Default 0.15 |
| transition_width | FLOAT | *(optional)* Soft ramp below the highlight threshold. Default 0.2 |
| edge_feather | FLOAT | *(optional)* Inward feather from the mask edge, px. Default 20 |

**Outputs:**

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Calibrated image |
| measured_hue | FLOAT | Average hue found in the paint area (degrees) |
| measured_saturation | FLOAT | Average saturation found in the paint area |
| applied_hue_shift | FLOAT | Hue delta actually applied, after clamping |
| applied_saturation_scale | FLOAT | Saturation multiplier actually applied, after clamping |
| effective_mask | MASK | The weighting mask the correction was applied through |

**Notes:**
- **Measurement hygiene.** The mask is thresholded at >0.5 then feathered *inward* via a distance transform, so pixels near the edge count for little — this cancels the background bleed you get from a grown/blurred YOLO silhouette. Only mid-tone pixels are sampled, so blown specular highlights and dark tyres/shadow can't skew the average.
- **Achromatic pixels are excluded** (`chroma_floor`). Glass, chrome and tyres have no hue, and in HSV they report hue = 0 = *red* — including them would drag a blue or green car's measurement toward red.
- **Hue is averaged as a circular mean.** A plain arithmetic mean is wrong on a wrap-around channel: a red car clustering at 355 deg and 5 deg would average to 180 deg (cyan).
- **Saturation is scaled, not offset.** An offset hits a nearly-grey pixel far harder in relative terms than a saturated one (+0.06 on s=0.05 more than doubles it), which shows up as blotchy glass and trim.
- **The measurement is refused unless it is trustworthy**, and the image is then returned **unchanged** rather than guessed at. Three conditions must all hold, each added because violating it produced a visibly wrong correction in testing:
  - enough sampled paint in absolute terms;
  - chromatic pixels are a real *fraction* of the paint area (`min_chroma_coverage`) — otherwise a silver or white car, which is almost entirely below `chroma_floor`, would have its "paint colour" decided by a badge or a taillight and that accent's hue rotated across the whole car;
  - the hue distribution is *concentrated* (`min_hue_concentration`) — a circular mean over a two-tone wrap, or paint plus background bleeding through the dilated mask, cancels out and returns an arbitrary angle with full confidence. Magenta paint plus green foliage measured 30° (orange), a hue present nowhere in the image.
- Batches are processed frame by frame; the FLOAT outputs report the first frame.
- A calibration pass must never be the thing that breaks a render, so every uncertain case is a no-op.

### Car Wrap Gentle HSV Calibration

Use `CarWrapGentleHSVCalibration` for the management-controlled final render
pass. Its unique node ID avoids the unrelated `MaskedAreaColorAdjuster` node
provided by older color-change packages.

- Required inputs: `image`, `mask`, `enabled` (default false), `target_hue`
  (0–360 degrees), and `target_saturation` (0–100 percent).
- Returns one IMAGE. Disabled processing returns the input tensor unchanged.
- Enabled processing uses the calibrator above with a fixed hue limit of 8
  degrees and saturation multiplier of 0.90–1.10. HSV Value is preserved.
- Connect the generated vehicle crop and its unexpanded vehicle mask before
  masked blending and placement back into the original photograph. A vehicle
  silhouette is not a semantic paint-only mask; check colored trim and reflections
  on representative real photographs before enabling colors in production.
- Deploy `nodes.py`, `render_calibration.py`, and `__init__.py` together, restart
  ComfyUI, and verify that `CarWrapGentleHSVCalibration` is registered. Backend
  settings and saved workflows must be deployed separately.

Run numerical regression checks without loading YOLO or a ComfyUI server:

```bash
python -B tests/test_render_calibration.py
```

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repo-url> ComfyUI-Extracted-Nodes
   ```

2. Install dependencies:
   ```bash
   cd ComfyUI-Extracted-Nodes
   pip install -r requirements.txt
   ```

3. The YOLOv8m-seg model (`models/yolov8m-seg.pt`) is included in the repository.

4. Restart ComfyUI.

## Requirements

- Python 3.10+
- PyTorch
- ultralytics (YOLOv8)
- opencv-contrib-python-headless
- Pillow
- numpy
