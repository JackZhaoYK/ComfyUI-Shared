import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

from .render_calibration import CarWrapGentleHSVCalibration
NODE_CLASS_MAPPINGS["CarWrapGentleHSVCalibration"] = CarWrapGentleHSVCalibration
NODE_DISPLAY_NAME_MAPPINGS["CarWrapGentleHSVCalibration"] = "Car Wrap Gentle HSV Calibration"
