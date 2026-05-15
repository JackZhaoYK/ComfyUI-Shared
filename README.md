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
