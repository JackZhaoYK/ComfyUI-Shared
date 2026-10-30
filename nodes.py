import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import comfy.model_management as model_management

here = Path(__file__).parent.resolve()


# ============================================================
# YOLO Vehicle Detection Utilities
# ============================================================

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (1248, 832),
    (1104, 944),
    (1024, 1024),
    (944, 1104),
    (1184, 880),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def find_best_resolution(width, height):
    """Find the best matching resolution from PREFERED_KONTEXT_RESOLUTIONS"""
    aspect_ratio = width / height
    _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
    return width, height


def _detect_largest_vehicle(image, yolo_model):
    """Run YOLO detection on a single image and return the largest car detection and all detections for debug."""
    results = yolo_model(image)

    largest_car_detection = None
    largest_car_detection_seg = None
    largest_area = 0
    all_detections = []

    for result in results:
        if result.boxes is None:
            continue
        has_masks = result.masks is not None
        for i, box in enumerate(result.boxes):
            class_name = result.names.get(box.cls.item(), "Unknown")
            conf = box.conf.item()
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            is_vehicle = class_name in ["car", "truck", "bus"]
            all_detections.append((bx1, by1, bx2, by2, class_name, conf, is_vehicle))

            if is_vehicle and conf > 0.25 and has_masks:
                seg = result.masks[i]
                area = (bx2 - bx1) * (by2 - by1)
                if area > largest_area:
                    largest_area = area
                    largest_car_detection = box
                    largest_car_detection_seg = seg

    return largest_car_detection, largest_car_detection_seg, largest_area, all_detections


def _draw_debug_image(image, all_detections):
    """Draw all detection boxes on the image for debugging."""
    debug_img = image.copy()
    for (x1, y1, x2, y2, class_name, conf, is_vehicle) in all_detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if is_vehicle:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        font_scale = 0.6
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(debug_img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(debug_img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    if not all_detections:
        h, w = debug_img.shape[:2]
        cv2.putText(debug_img, "NO DETECTIONS", (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    return debug_img


def chop_largest_vehicle(image, yolo_model, crop_car_image):
    """Detect and crop the largest vehicle from an image using YOLO."""
    largest_car_detection, largest_car_detection_seg, largest_area, all_detections = _detect_largest_vehicle(image, yolo_model)

    # If no car found, try with swapped color space (RGB<->BGR)
    if largest_car_detection is None:
        image_swapped = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        det_swapped, seg_swapped, area_swapped, all_detections_swapped = _detect_largest_vehicle(image_swapped, yolo_model)
        if det_swapped is not None:
            largest_car_detection = det_swapped
            largest_car_detection_seg = seg_swapped
            largest_area = area_swapped
            all_detections = all_detections_swapped
            print(f"[YOLO] Car detected after color space swap (RGB<->BGR)")
        else:
            all_detections = all_detections + all_detections_swapped

    debug_image = _draw_debug_image(image, all_detections)

    if largest_car_detection is not None:
        x1, y1, x2, y2 = largest_car_detection.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        padding = 50
        height, width, _ = image.shape
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        seg = largest_car_detection_seg.data.cpu().numpy()[0]
        seg = cv2.resize(seg, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        if crop_car_image:
            seg = seg[y1:y2, x1:x2]

        return image[y1:y2, x1:x2], x1, y1, x2, y2, seg, debug_image
    else:
        if all_detections:
            detected_classes = [f"{name}({conf:.2f})" for (_, _, _, _, name, conf, _) in all_detections]
            raise Exception(
                f"No car/truck/bus detections found. "
                f"YOLO detected other objects: {', '.join(detected_classes)}. "
                f"Tried both original and swapped (RGB<->BGR) color spaces. "
                f"Check the debug_image output for all detection boxes."
            )
        else:
            raise Exception(
                "No objects detected at all by YOLO. "
                "The image may be too dark, too blurry, or not contain recognizable objects. "
                "Tried both original and swapped (RGB<->BGR) color spaces. "
                "Check the debug_image output to verify."
            )


def chop_largest_vehicle_yolo(image, use_resolution_matching=True):
    """
    Main function that finds the largest vehicle and crops original image.
    
    Args:
        image: Input image
        use_resolution_matching: If True, finds best resolution and resizes. 
                                If False, just crops to vehicle bounding box.
    
    Returns:
        Cropped/resized image and crop coordinates
    """
    yolo_model = YOLO(str(Path(here, "./models/yolov8m-seg.pt"))).to(model_management.get_torch_device())
    
    _, x1, y1, x2, y2, seg, debug_image = chop_largest_vehicle(image, yolo_model, False)
    
    img_height, img_width = image.shape[:2]
    
    if use_resolution_matching:
        vehicle_width = x2 - x1
        vehicle_height = y2 - y1
        
        target_width, target_height = find_best_resolution(vehicle_width, vehicle_height)
        target_ratio = target_width / target_height
        
        crop_width = vehicle_width
        crop_height = vehicle_height
        current_ratio = crop_width / crop_height
        
        if current_ratio > target_ratio:
            crop_height = int(crop_width / target_ratio)
        else:
            crop_width = int(crop_height * target_ratio)
        
        vehicle_center_x = (x1 + x2) // 2
        vehicle_center_y = (y1 + y2) // 2
        
        crop_x1 = max(0, vehicle_center_x - crop_width // 2)
        crop_y1 = max(0, vehicle_center_y - crop_height // 2)
        crop_x2 = min(img_width, crop_x1 + crop_width)
        crop_y2 = min(img_height, crop_y1 + crop_height)
        
        if crop_x2 - crop_x1 < crop_width:
            crop_x1 = max(0, crop_x2 - crop_width)
        if crop_y2 - crop_y1 < crop_height:
            crop_y1 = max(0, crop_y2 - crop_height)
        
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height
        
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        resized_image = cv2.resize(cropped_image, (target_width, target_height), 
                                   interpolation=cv2.INTER_LANCZOS4)
        
        if seg is not None:
            seg = seg[crop_y1:crop_y2, crop_x1:crop_x2]
            seg = cv2.resize(seg, (target_width, target_height), 
                                   interpolation=cv2.INTER_LANCZOS4)
        
        return resized_image, crop_x1, crop_y1, crop_x2, crop_y2, seg, debug_image

    else:
        crop_x1, crop_y1, crop_x2, crop_y2 = x1, y1, x2, y2
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if seg is not None:
            seg = seg[crop_y1:crop_y2, crop_x1:crop_x2]
        return cropped_image, crop_x1, crop_y1, crop_x2, crop_y2, seg, debug_image


# ============================================================
# Node: YOLO Car (YOLO_Preprocessor)
# ============================================================

class YOLO_Preprocessor:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_resolution_matching": ("BOOLEAN", {"default": True})
            },
        }

    RETURN_TYPES = ("MASK","IMAGE","INT","INT","INT","INT","IMAGE",)
    RETURN_NAMES = ("mask","image","x1","y1","x2","y2","debug_image",)
    FUNCTION = "execute"

    CATEGORY = "custom"

    def execute(self, image, use_resolution_matching):
        batch_size = image.shape[0]
        out_tensor = None
        car_crop_tensor = None
        debug_tensor = None
        for i, single_image in enumerate(image):
            np_image = np.asarray(single_image.cpu() * 255., dtype=np.uint8)
            chopped_image, x1, y1, x2, y2, car_mask, debug_image = chop_largest_vehicle_yolo(np_image, use_resolution_matching)

            out = torch.from_numpy(np.where(car_mask > 0, 255, 0).astype(np.float32) / 255.0)
            car_crop_out = torch.from_numpy(chopped_image.astype(np.float32) / 255.0)
            debug_out = torch.from_numpy(debug_image.astype(np.float32) / 255.0)

            if out_tensor is None:
                out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
            out_tensor[i] = out

            if car_crop_tensor is None:
                car_crop_tensor = torch.zeros(batch_size, *car_crop_out.shape, dtype=torch.float32)
            car_crop_tensor[i] = car_crop_out

            if debug_tensor is None:
                debug_tensor = torch.zeros(batch_size, *debug_out.shape, dtype=torch.float32)
            debug_tensor[i] = debug_out
        return (out_tensor, car_crop_tensor, x1, y1, x2, y2, debug_tensor,)


# ============================================================
# Node: Place Cropped Image Back (ImagePlacementNode)
# ============================================================

class ImagePlacementNode:
    """
    A ComfyUI node that places a cropped image back onto its original location
    in the background image using pixel coordinates (x1, y1, x2, y2).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "x1": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "y1": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "x2": ("INT", {"default": 100, "min": 0, "max": 9999}),
                "y2": ("INT", {"default": 100, "min": 0, "max": 9999}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "place_image"
    CATEGORY = "image/transform"
    
    def place_image(self, background_image, cropped_image, x1, y1, x2, y2):
        """
        Place cropped image back to its original location on background.
        Simple approach: resize cropped image to exact coordinate dimensions and paste.
        """
        bg_pil = self.tensor_to_pil(background_image)
        crop_pil = self.tensor_to_pil(cropped_image)
        
        bg_width, bg_height = bg_pil.size
        
        x1 = max(0, min(x1, bg_width))
        y1 = max(0, min(y1, bg_height))
        x2 = max(x1, min(x2, bg_width))
        y2 = max(y1, min(y2, bg_height))
        
        target_width = x2 - x1
        target_height = y2 - y1
        
        if target_width <= 0 or target_height <= 0:
            print(f"Warning: Invalid target dimensions ({target_width}x{target_height}), returning background unchanged")
            return (self.pil_to_tensor(bg_pil),)
        
        resized_crop = crop_pil.resize((target_width, target_height), Image.LANCZOS)
        
        result = bg_pil.copy()
        result.paste(resized_crop, (x1, y1))
        
        result_tensor = self.pil_to_tensor(result)
        
        print(f"Placed image at ({x1}, {y1}) with size {target_width}x{target_height}")
        
        return (result_tensor,)
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        tensor = tensor.astype(np.uint8)
        return Image.fromarray(tensor)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        np_image = np.array(pil_image)
        np_image = np_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        return tensor


# ============================================================
# Node: Apply Masked Blend (ApplyMaskedBlendNode)
# ============================================================

class ApplyMaskedBlendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "input_image": ("IMAGE",),
                "mask": ("MASK",),
                "blend_percentage": ("FLOAT", {"default": 1.0, "min": 0, "max": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_masked_blend"
    CATEGORY = "custom"

    def apply_masked_blend(self, background_image, input_image, mask, blend_percentage):
        """
        Apply the masked area of the input image onto the background image,
        blending the masked region smoothly based on the mask.
        """
        if background_image.dim() == 4:
            background_image = background_image.squeeze(0)
        if input_image.dim() == 4:
            input_image = input_image.squeeze(0)
        
        while mask.dim() > 2:
            mask = mask.squeeze(0)

        background_np = background_image.cpu().numpy()
        input_np = input_image.cpu().numpy()
        mask_np = mask.cpu().numpy()

        print(f"Background shape: {background_np.shape}")
        print(f"Input shape: {input_np.shape}")
        print(f"Mask shape: {mask_np.shape}")
        print(f"Mask range: {mask_np.min()} to {mask_np.max()}")

        target_height, target_width = background_np.shape[:2]

        if input_np.shape[:2] != (target_height, target_width):
            print(f"Resizing input image from {input_np.shape[:2]} to {(target_height, target_width)}")
            input_np = cv2.resize(input_np, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        if mask_np.shape != (target_height, target_width):
            print(f"Resizing mask from {mask_np.shape} to {(target_height, target_width)}")
            mask_np = cv2.resize(mask_np, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        mask_np = np.clip(mask_np.astype(np.float32), 0, 1)
        mask_np *= blend_percentage

        mask_expanded = mask_np[..., None]
        blended_image_np = background_np * (1 - mask_expanded) + input_np * mask_expanded
        blended_image_np = np.clip(blended_image_np, 0, 1)

        blended_image_tensor = torch.from_numpy(blended_image_np).float().unsqueeze(0)

        return (blended_image_tensor,)


# ============================================================
# Masked Area Color Adjuster (HSV calibration of the car paint)
# ============================================================

class MaskedAreaColorAdjuster:
    """
    Nudge the colour of the masked (car paint) area toward a target hue and
    saturation, by a deliberately BOUNDED amount.

    This is a calibration pass, not a repaint: it MEASURES the average hue and
    saturation actually present in the paint area, works out the delta to the
    target, clamps that delta to `max_hue_shift` / `max_saturation_shift`, and
    applies the clamped delta to every pixel. Each pixel keeps its own shading,
    so highlights, reflections and shadow structure survive; only the overall
    colour cast moves. Value/luminance is never touched.

    Measurement hygiene (this is what makes the average trustworthy):
      * the mask is thresholded at >0.5, then feathered INWARD via a distance
        transform, so pixels near the mask edge count for little - that cancels
        the background bleed you get from a grown/blurred YOLO silhouette;
      * only mid-tone pixels are sampled (between shadow_threshold and
        highlight_threshold), so blown specular highlights and dark tyres /
        shadow do not drag the average;
      * only pixels with real chroma (saturation >= chroma_floor) contribute to
        the colour statistics. This matters: glass, chrome and tyres are
        achromatic, and in HSV an achromatic pixel reports hue = 0 = RED, so
        including them would drag a blue or green car's measurement toward red.
      * hue is averaged as a CIRCULAR mean (vector sum). A plain arithmetic mean
        is wrong on a wrap-around channel: a red car clustering at 355 deg and
        5 deg would average to 180 deg, i.e. cyan.

    The measurement is REFUSED (image returned unchanged) unless it is actually
    trustworthy. Three independent conditions must hold, and each one exists
    because violating it produced a visibly wrong correction in testing:
      * enough sampled paint in absolute terms (`MIN_SAMPLE_WEIGHT`);
      * the chromatic pixels must be a real FRACTION of the sampled paint
        (`min_chroma_coverage`). Without this, a silver or white car - which is
        almost entirely below `chroma_floor` - has its "paint colour" decided by
        a handful of accent pixels (a badge, a taillight, a red brake caliper),
        and that accent's hue is then rotated across the whole car;
      * the hue distribution must be CONCENTRATED (`min_hue_concentration`,
        the circular resultant length R). A circular mean over a bimodal region
        - a colour-shift wrap, or paint plus a chunk of contrasting background
        inside the dilated mask - cancels out and returns an essentially
        arbitrary angle with full confidence. Magenta paint plus green foliage
        measured 30 deg (orange), a hue present nowhere in the image.

    A calibration pass must never be the thing that breaks a render, so every
    uncertain case is a no-op rather than a guess.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "target_hue": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 0.1}),
                "target_saturation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_hue_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 60.0, "step": 0.5}),
                "max_saturation_shift": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "chroma_floor": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_chroma_coverage": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_hue_concentration": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "filter_highlights": ("BOOLEAN", {"default": True}),
                "highlight_threshold": ("FLOAT", {"default": 0.37, "min": 0.3, "max": 1.0, "step": 0.01}),
                "shadow_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.8, "step": 0.01}),
                "transition_width": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 0.5, "step": 0.01}),
                "edge_feather": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "MASK")
    RETURN_NAMES = ("image", "measured_hue", "measured_saturation",
                    "applied_hue_shift", "applied_saturation_scale", "effective_mask")
    FUNCTION = "adjust"
    CATEGORY = "custom"

    # Minimum total sampling weight (in pixels) before we trust the average.
    MIN_SAMPLE_WEIGHT = 64.0

    def adjust(self, image, mask, target_hue, target_saturation,
               max_hue_shift, max_saturation_shift, strength,
               chroma_floor=0.10, min_chroma_coverage=0.15,
               min_hue_concentration=0.50, filter_highlights=True,
               highlight_threshold=0.37, shadow_threshold=0.15,
               transition_width=0.2, edge_feather=20.0):

        # Normalise to a batch of HWC frames and a batch of HW masks. Note the
        # explicit reshape rather than a squeeze loop: torch.squeeze(0) is a
        # NO-OP when dim 0 has size != 1, so `while m.dim() > 2: m.squeeze(0)`
        # spins forever on a batched (B,H,W) mask and wedges the worker.
        images = image.unsqueeze(0) if image.dim() == 3 else image
        masks = mask.reshape(-1, mask.shape[-2], mask.shape[-1]) if mask.dim() != 2 \
            else mask.unsqueeze(0)

        out_images, out_masks = [], []
        first = None
        for i in range(images.shape[0]):
            # A single mask is broadcast across the batch; otherwise pair them up.
            m = masks[i] if masks.shape[0] > 1 else masks[0]
            res = self._adjust_one(
                images[i], m, target_hue, target_saturation,
                max_hue_shift, max_saturation_shift, strength,
                chroma_floor, min_chroma_coverage, min_hue_concentration,
                filter_highlights, highlight_threshold, shadow_threshold,
                transition_width, edge_feather)
            out_images.append(res[0])
            out_masks.append(res[4])
            if first is None:
                first = res

        # The FLOAT outputs are scalars, so they report the first frame; the
        # batch itself is fully processed.
        return (torch.stack(out_images, dim=0),
                float(first[1]), float(first[2]), float(first[3]), float(first[5]),
                torch.stack(out_masks, dim=0))

    # ------------------------------------------------------------------
    def _adjust_one(self, image_hwc, mask_hw, target_hue, target_saturation,
                    max_hue_shift, max_saturation_shift, strength,
                    chroma_floor, min_chroma_coverage, min_hue_concentration,
                    filter_highlights, highlight_threshold, shadow_threshold,
                    transition_width, edge_feather):
        """Returns (image_hwc, measured_hue, measured_sat, hue_delta, mask_hw, sat_scale)."""
        # nan_to_num BEFORE the clip: np.clip passes NaN straight through, and a
        # single NaN pixel would otherwise reach the output (and, on some OpenCV
        # dispatch paths, poison the whole measurement).
        image_np = np.nan_to_num(image_hwc.cpu().numpy().astype(np.float32),
                                 nan=0.0, posinf=1.0, neginf=0.0)
        image_np = np.clip(image_np, 0.0, 1.0)
        mask_np = np.nan_to_num(mask_hw.cpu().numpy().astype(np.float32),
                                nan=0.0, posinf=1.0, neginf=0.0)

        h_img, w_img = image_np.shape[:2]
        if mask_np.shape != (h_img, w_img):
            print(f"[ColorAdjust] resizing mask {mask_np.shape} -> {(h_img, w_img)}")
            mask_np = cv2.resize(mask_np, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        mask_np = np.clip(mask_np, 0.0, 1.0)

        sampling_mask, mixing_mask = self._build_masks(
            image_np, mask_np, filter_highlights,
            highlight_threshold, shadow_threshold, transition_width, edge_feather)

        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)   # H 0..360, S 0..1, V 0..1
        hue_ch, sat_ch = hsv[:, :, 0], hsv[:, :, 1]

        measured_hue, measured_sat, why = self._measure(
            hue_ch, sat_ch, sampling_mask, chroma_floor,
            min_chroma_coverage, min_hue_concentration)

        mask_out = torch.from_numpy(mixing_mask).float()
        if measured_hue is None:
            print(f"[ColorAdjust] not calibrating - {why}")
            # Return the SANITISED copy, not the caller's tensor: even on the
            # no-op path this node should never hand a NaN/inf downstream.
            return (torch.from_numpy(image_np).float(), 0.0, 0.0, 0.0, mask_out, 1.0)

        # Hue is a rotation: take the SHORTEST way round the circle, then clamp.
        hue_delta = ((target_hue - measured_hue + 180.0) % 360.0) - 180.0
        hue_delta = float(np.clip(hue_delta, -max_hue_shift, max_hue_shift))

        # Saturation is scaled, not offset: an offset would hit a nearly-grey
        # pixel far harder in relative terms than a saturated one (+0.06 on
        # s=0.05 more than doubles it), which shows up as blotchy glass and
        # trim. A clamped ratio moves every pixel proportionally.
        sat_scale = target_saturation / measured_sat if measured_sat > 1e-4 else 1.0
        sat_scale = float(np.clip(sat_scale, 1.0 - max_saturation_shift,
                                  1.0 + max_saturation_shift))

        print(f"[ColorAdjust] measured H={measured_hue:.1f} S={measured_sat:.3f} ({why}) "
              f"| target H={target_hue:.1f} S={target_saturation:.3f} "
              f"| applying dH={hue_delta:+.2f} sat x{sat_scale:.3f} @ strength {strength:.2f}")

        weight = np.clip(mixing_mask * float(strength), 0.0, 1.0)
        hsv[:, :, 0] = (hue_ch + hue_delta * weight) % 360.0
        hsv[:, :, 1] = np.clip(sat_ch * (1.0 + (sat_scale - 1.0) * weight), 0.0, 1.0)
        # hsv[:, :, 2] (value) deliberately untouched - shading must not move.

        result = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)
        return (torch.from_numpy(result).float(),
                float(measured_hue), float(measured_sat), hue_delta, mask_out, sat_scale)

    # ------------------------------------------------------------------
    def _build_masks(self, image_np, mask_np, filter_highlights,
                     highlight_threshold, shadow_threshold, transition_width, edge_feather):
        """Strict sampling mask + smooth mixing mask."""
        luminance = (0.299 * image_np[:, :, 0]
                     + 0.587 * image_np[:, :, 1]
                     + 0.114 * image_np[:, :, 2])

        mask_binary = (mask_np > 0.5).astype(np.float32)

        # Feather INWARD from the mask edge: pixels within `edge_feather` px of
        # the boundary lose weight, which keeps a dilated/blurred YOLO
        # silhouette from dragging background into the measurement.
        # The 1px zero border matters: distanceTransform measures distance to
        # the nearest zero pixel and does NOT treat the frame edge as
        # background, so a mask that fills the frame would otherwise report a
        # huge distance everywhere and the feather would silently do nothing.
        if edge_feather > 0:
            padded = cv2.copyMakeBorder((mask_binary * 255).astype(np.uint8),
                                        1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
            mask_feathered = mask_binary * np.clip(dist / edge_feather, 0.0, 1.0)
        else:
            mask_feathered = mask_binary

        if not filter_highlights:
            return mask_feathered, mask_feathered

        # Sampling: mid-tones only, so specular blowouts and shadow don't skew.
        luminance_filter = ((luminance >= shadow_threshold)
                            & (luminance < highlight_threshold)).astype(np.float32)
        sampling_mask = mask_feathered * luminance_filter

        # Mixing: full strength below the transition, ramping to zero at the
        # highlight threshold, so the correction doesn't tide-mark across glare.
        transition_start = highlight_threshold - transition_width
        full_mix = (luminance < transition_start).astype(np.float32)
        in_transition = ((luminance >= transition_start) & (luminance < highlight_threshold))
        ramp = np.where(in_transition,
                        1.0 - (luminance - transition_start) / max(transition_width, 1e-6),
                        0.0)
        mixing_mask = mask_feathered * np.clip(full_mix + ramp, 0.0, 1.0)

        return sampling_mask, mixing_mask

    # ------------------------------------------------------------------
    def _measure(self, hue_ch, sat_ch, sampling_mask, chroma_floor,
                 min_chroma_coverage, min_hue_concentration):
        """
        Chroma-gated circular mean hue + mean saturation over the sampling area.
        Returns (hue_deg, saturation, reason). hue_deg is None when the paint
        cannot be measured trustworthily, and `reason` says why.
        """
        sampled_weight = float(sampling_mask.sum())
        if sampled_weight < self.MIN_SAMPLE_WEIGHT:
            return None, 0.0, (f"only {sampled_weight:.0f} px of sampled paint, "
                               f"need {self.MIN_SAMPLE_WEIGHT:.0f}")

        # Achromatic pixels carry no hue information (and report hue 0 = red),
        # so they are excluded outright rather than merely down-weighted.
        selected = (sampling_mask > 0.0) & (sat_ch >= chroma_floor)
        weights = sampling_mask[selected]
        chroma_weight = float(weights.sum())

        # A few chromatic accent pixels on an otherwise achromatic car must not
        # be allowed to define "the paint colour" for the whole mask.
        coverage = chroma_weight / sampled_weight if sampled_weight > 0 else 0.0
        if chroma_weight < self.MIN_SAMPLE_WEIGHT or coverage < min_chroma_coverage:
            return None, 0.0, (f"only {coverage * 100:.1f}% of the paint area has "
                               f"usable chroma (need {min_chroma_coverage * 100:.0f}%) "
                               f"- car reads as achromatic")

        angles = np.deg2rad(hue_ch[selected].astype(np.float64))
        # Weight the hue vectors by chroma too: a strongly-coloured body panel
        # should say more about the paint than a barely-tinted one.
        hue_weights = weights * sat_ch[selected]
        if hue_weights.sum() <= 1e-6:
            hue_weights = weights
        wsum = float(hue_weights.sum())
        x = float(np.sum(hue_weights * np.cos(angles)))
        y = float(np.sum(hue_weights * np.sin(angles)))

        # Resultant length: 1 = all pixels agree on a hue, 0 = they cancel out.
        # A low R means the circular mean is meaningless (bimodal paint, or
        # background bleeding into the mask), so refuse rather than invent.
        concentration = float(np.hypot(x, y) / wsum) if wsum > 1e-9 else 0.0
        if concentration < min_hue_concentration:
            return None, 0.0, (f"hue is not concentrated (R={concentration:.2f}, "
                               f"need {min_hue_concentration:.2f}) - the masked area "
                               f"holds more than one colour")

        measured_hue = float(np.rad2deg(np.arctan2(y, x)) % 360.0)
        # Saturation is averaged with the plain mask weights - weighting it by
        # saturation as well would bias the estimate upward.
        measured_sat = float(np.sum(weights * sat_ch[selected]) / chroma_weight)

        return measured_hue, measured_sat, (f"{coverage * 100:.0f}% chroma coverage, "
                                            f"R={concentration:.2f}")


# ============================================================
# Node Registration
# ============================================================

NODE_CLASS_MAPPINGS = {
    "YoloPreprocessor": YOLO_Preprocessor,
    "ImagePlacementNode": ImagePlacementNode,
    "ApplyMaskedBlendNode": ApplyMaskedBlendNode,
    "MaskedAreaColorAdjuster": MaskedAreaColorAdjuster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloPreprocessor": "YOLO Car",
    "ImagePlacementNode": "Place Cropped Image Back",
    "ApplyMaskedBlendNode": "Apply Masked Blend",
    "MaskedAreaColorAdjuster": "Masked Area Color Adjuster",
}
