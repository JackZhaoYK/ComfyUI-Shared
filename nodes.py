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
# Node Registration
# ============================================================

NODE_CLASS_MAPPINGS = {
    "YoloPreprocessor": YOLO_Preprocessor,
    "ImagePlacementNode": ImagePlacementNode,
    "ApplyMaskedBlendNode": ApplyMaskedBlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloPreprocessor": "YOLO Car",
    "ImagePlacementNode": "Place Cropped Image Back",
    "ApplyMaskedBlendNode": "Apply Masked Blend",
}
