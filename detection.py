import time

import cv2
import numpy as np

from config import (
    CUBE_MATCH_DISTANCE_PX,
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    DEFAULT_MAX_CUBE_AREA_PX,
    DEFAULT_MIN_CUBE_AREA_PX,
    DEFAULT_YOLO_MODEL_NAME,
    HSV_COLOR_RANGES,
    HSV_DRAW_COLORS,
    HSV_SELECTED_CLASS_NAME,
    SHOW_LIVE_DEBUG_TEXT,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
    YOLO_MAX_FPS,
    YOLO_MODELS,
    YOLO_TARGET_CLASS,
)

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    torch = None
    YOLO = None


class CubeDetector:
    def __init__(
        self,
        lower_hsv=DEFAULT_HSV_LOWER,
        upper_hsv=DEFAULT_HSV_UPPER,
        color_ranges=HSV_COLOR_RANGES,
        min_area=DEFAULT_MIN_CUBE_AREA_PX,
        max_area=DEFAULT_MAX_CUBE_AREA_PX,
    ):
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.selected_profile = HSV_SELECTED_CLASS_NAME
        self.color_ranges = self._normalize_color_ranges(color_ranges)
        self.target_class = self._normalize_class_name(HSV_SELECTED_CLASS_NAME)
        self.min_area = min_area
        self.max_area = max_area

    def _normalize_class_name(self, class_name):
        return str(class_name).strip().lower().lstrip(".")

    def set_target_class(self, class_name):
        self.target_class = self._normalize_class_name(class_name)

    def _is_target_class(self, class_name):
        return self._normalize_class_name(class_name) == self.target_class

    def _normalize_color_ranges(self, color_ranges):
        normalized = {}

        for name, values in color_ranges.items():
            normalized[str(name).lower()] = {
                "lower": np.array(values["lower"], dtype=np.uint8),
                "upper": np.array(values["upper"], dtype=np.uint8),
            }

        return normalized

    def update_settings(
        self,
        lower_hsv=None,
        upper_hsv=None,
        min_area=None,
        max_area=None,
        profile_name=None,
    ):
        if profile_name is not None:
            self.selected_profile = str(profile_name).lower()

        if lower_hsv is not None:
            self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        if upper_hsv is not None:
            self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        if lower_hsv is not None or upper_hsv is not None:
            self.color_ranges[self.selected_profile] = {
                "lower": self.lower_hsv.copy(),
                "upper": self.upper_hsv.copy(),
            }
        if min_area is not None:
            self.min_area = int(min_area)
        if max_area is not None:
            self.max_area = int(max_area)

    def set_selected_profile(self, profile_name):
        self.selected_profile = str(profile_name).lower()
        profile = self.color_ranges[self.selected_profile]
        self.lower_hsv = profile["lower"].copy()
        self.upper_hsv = profile["upper"].copy()

    def detect_blocks(self, frame, calibration_ready=False, pixel_to_world_func=None,
        is_inside_workspace_func=None, is_in_detection_zone_func=None, world_to_robot_func=None,
        include_placement_points=False):

        blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        kernel = np.ones((5, 5), np.uint8)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        detected_points = []

        for class_name, hsv_range in self.color_ranges.items():
            mask = cv2.inRange(hsv, hsv_range["lower"], hsv_range["upper"])
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=4)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            draw_color = HSV_DRAW_COLORS.get(class_name, (0, 255, 0))
            is_target = self._is_target_class(class_name)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < self.min_area or area > self.max_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)

                if not 0.7 < aspect_ratio < 1.3:
                    continue

                cx = x + w // 2
                cy = y + h // 2
                block_info = {
                    "pixel": (cx, cy),
                    "bbox": (x, y, w, h),
                    "class_name": class_name,
                    "confidence": None,
                }
                text = f"{class_name}: {cx},{cy}"

                if calibration_ready:
                    xw, yw = pixel_to_world_func(cx, cy)
                    block_info["world"] = (xw, yw)

                    if not is_inside_workspace_func(xw, yw):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                        if SHOW_LIVE_DEBUG_TEXT:
                            cv2.putText( frame, f"{class_name} | Out of workspace", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, draw_color, 2)
                        continue

                    if not is_in_detection_zone_func(xw, yw):
                        block_info["zone"] = "placement"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                        if SHOW_LIVE_DEBUG_TEXT:
                            cv2.putText( frame, f"{class_name} | Placement zone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                draw_color, 2)
                        if include_placement_points:
                            detected_points.append(block_info)
                        continue

                    xr, yr, zr = world_to_robot_func(xw, yw)
                    block_info["robot"] = (xr, yr, zr)
                    block_info["zone"] = "detection"
                    text = class_name

                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                cv2.circle(frame, (cx, cy), 5, draw_color, -1)
                if SHOW_LIVE_DEBUG_TEXT:
                    cv2.putText( frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, draw_color, 2)

                if is_target:
                    detected_points.append(block_info)

        return frame, combined_mask, detected_points


class YoloCubeDetector:
    """YOLO cube detector integration reused from Myproject/test1.py."""

    def __init__(
        self,
        models=YOLO_MODELS,
        default_model_name=DEFAULT_YOLO_MODEL_NAME,
        confidence=YOLO_CONFIDENCE,
        image_size=YOLO_IMAGE_SIZE,
        target_class=YOLO_TARGET_CLASS,
        max_fps=YOLO_MAX_FPS,
    ):
        self.models = models
        self.model_name = default_model_name
        self.confidence = confidence
        self.image_size = image_size
        self.target_class = self._normalize_class_name(target_class)
        self.max_fps = max_fps
        self.min_frame_time = 1.0 / max_fps if max_fps > 0 else 0
        self.prev_time = time.time()
        self.model = None
        self.device = 0 if torch is not None and torch.cuda.is_available() else "cpu"

    def set_model(self, model_name):
        if model_name == self.model_name and self.model is not None:
            return

        self.model_name = model_name
        self.model = None

    def update_settings(self, *args, **kwargs):
        pass

    def set_target_class(self, class_name):
        self.target_class = self._normalize_class_name(class_name)

    def _normalize_class_name(self, class_name):
        return str(class_name).strip().lower().lstrip(".")

    def _is_target_class(self, class_name):
        return self._normalize_class_name(class_name) == self.target_class

    def _class_name_from_id(self, names, class_id):
        if isinstance(names, dict):
            return names.get(class_id, class_id)
        try:
            return names[class_id]
        except (IndexError, TypeError):
            return class_id

    def load_model(self):
        if YOLO is None:
            raise RuntimeError("YOLO dependencies are missing. Install ultralytics and torch.")

        if self.model_name not in self.models:
            raise RuntimeError(f"Unknown YOLO model: {self.model_name}")

        if self.model is None:
            self.model = YOLO(self.models[self.model_name])
            print("YOLO model loaded:", self.models[self.model_name])
            print("YOLO classes:", self.model.names)

        return self.model

    def detect_blocks(
        self,
        frame,
        calibration_ready=False,
        pixel_to_world_func=None,
        is_inside_workspace_func=None,
        is_in_detection_zone_func=None,
        world_to_robot_func=None,
        include_placement_points=False,
    ):
        if self.min_frame_time > 0:
            elapsed = time.time() - self.prev_time
            if elapsed < self.min_frame_time:
                time.sleep(self.min_frame_time - elapsed)

        start_time = time.time()
        model = self.load_model()
        results = model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detected_points = []
        result = results[0]
        names = result.names
        annotated = frame

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x = int(round(x1))
            y = int(round(y1))
            w = int(round(x2 - x1))
            h = int(round(y2 - y1))
            cx = x + w // 2
            cy = y + h // 2
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self._normalize_class_name(self._class_name_from_id(names, class_id))
            is_target = self._is_target_class(class_name)
            draw_color = (0, 255, 0) if is_target else HSV_DRAW_COLORS.get(class_name, (255, 255, 0))
            thickness = 3 if is_target else 2

            block_info = {
                "pixel": (cx, cy),
                "bbox": (x, y, w, h),
                "class_name": class_name,
                "confidence": confidence,
            }
            label = f"{class_name} {confidence:.2f}"
            xw = yw = None

            if calibration_ready:
                xw, yw = pixel_to_world_func(cx, cy)
                block_info["world"] = (xw, yw)

                if not is_inside_workspace_func(xw, yw):
                    continue

            cv2.rectangle(annotated, (x, y), (x + w, y + h), draw_color, thickness)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                draw_color,
                2,
            )

            if calibration_ready:
                if not is_in_detection_zone_func(xw, yw):
                    block_info["zone"] = "placement"
                    cv2.putText(
                        annotated,
                        "Placement zone",
                        (x, min(y + h + 22, annotated.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                    )
                    if include_placement_points:
                        detected_points.append(block_info)
                    continue

                if not is_target:
                    continue

                xr, yr, zr = world_to_robot_func(xw, yw)
                block_info["robot"] = (xr, yr, zr)
                block_info["zone"] = "detection"
                cv2.putText(
                    annotated,
                    f"TARGET | W:{xw:.1f},{yw:.1f} R:{xr:.1f},{yr:.1f}",
                    (x, min(y + h + 22, annotated.shape[0] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            else:
                if not is_target:
                    continue

                cv2.putText(
                    annotated,
                    f"TARGET {self.target_class}",
                    (x, min(y + h + 22, annotated.shape[0] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            detected_points.append(block_info)

        end_time = time.time()
        self.prev_time = end_time
        fps = 1.0 / max(end_time - start_time, 1e-6)
        fps = min(fps, float(self.max_fps)) if self.max_fps > 0 else fps

        overlay_x = 10
        overlay_y = annotated.shape[0] - 55
        cv2.rectangle(
            annotated,
            (overlay_x - 5, overlay_y - 25),
            (overlay_x + 350, overlay_y + 40),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            f"Model: {self.model_name} | Target: {self.target_class}",
            (overlay_x, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f} / cap {self.max_fps}",
            (overlay_x, overlay_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        return annotated, None, detected_points


class SavedCubeTracker:
    def __init__(self):
        self.detected_cube_points = []
        self.next_cube_id = 1
        self.last_detection_update = 0

    def reset(self):
        self.detected_cube_points = []
        self.next_cube_id = 1
        self.last_detection_update = 0

    def update_saved_cube_points(self, live_points):
        matched_saved_ids = set()
        updated_points = []

        for point in live_points:
            px, py = point["pixel"]
            point_class = point.get("class_name")
            closest_saved = None
            closest_distance = CUBE_MATCH_DISTANCE_PX

            for saved in self.detected_cube_points:
                if saved["id"] in matched_saved_ids:
                    continue
                if point_class is not None and saved.get("class_name") != point_class:
                    continue

                sx, sy = saved["pixel"]
                distance = np.hypot(px - sx, py - sy)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_saved = saved

            if closest_saved is None:
                cube_id = self.next_cube_id
                self.next_cube_id += 1
            else:
                cube_id = closest_saved["id"]
                matched_saved_ids.add(cube_id)

            saved_point = point.copy()
            saved_point["id"] = cube_id
            updated_points.append(saved_point)

        updated_points.sort(key=lambda item: item["id"])
        self.detected_cube_points = updated_points
        self.last_detection_update = time.monotonic()


def draw_saved_cube_points(frame, detected_cube_points, show_text=SHOW_LIVE_DEBUG_TEXT):
    for point in detected_cube_points:
        px, py = point["pixel"]
        class_name = point.get("class_name", "cube")

        cv2.circle(frame, (px, py), 9, (255, 0, 0), 2)
        if show_text:
            cv2.putText(
                frame,
                f"Saved {point['id']} {class_name}",
                (px + 10, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
