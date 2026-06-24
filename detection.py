import time

import cv2
import numpy as np

from config import CUBE_MATCH_DISTANCE_PX


class CubeDetector:
    def detect_blocks(
        self,
        frame,
        calibration_ready=False,
        pixel_to_world_func=None,
        is_in_detection_zone_func=None,
        world_to_robot_func=None,
    ):
        blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        lower_color = np.array([38, 28, 26])
        upper_color = np.array([84, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=4)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detected_points = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area <= 15:
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
            }
            text = f"Pixel: {cx},{cy}"

            if calibration_ready:
                xw, yw = pixel_to_world_func(cx, cy)

                if not is_in_detection_zone_func(xw, yw):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(
                        frame,
                        "Placement zone",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2,
                    )
                    continue

                xr, yr, zr = world_to_robot_func(xw, yw)
                block_info["world"] = (xw, yw)
                block_info["robot"] = (xr, yr, zr)
                text = f"W: {xw:.1f},{yw:.1f} mm | R: {xr:.1f},{yr:.1f}"

            detected_points.append(block_info)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame, mask, detected_points


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
            closest_saved = None
            closest_distance = CUBE_MATCH_DISTANCE_PX

            for saved in self.detected_cube_points:
                if saved["id"] in matched_saved_ids:
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


def draw_saved_cube_points(frame, detected_cube_points):
    for point in detected_cube_points:
        px, py = point["pixel"]

        cv2.circle(frame, (px, py), 9, (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"Saved {point['id']}",
            (px + 10, py + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

