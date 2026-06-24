import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import calibration
from config import CAMERA_ID, DETECTION_UPDATE_SECONDS, PLACE_POINT_WORLD
from detection import CubeDetector, SavedCubeTracker, draw_saved_cube_points
from robot_commands import RobotController


class RobotVisionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Vision Calibration UI")

        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FPS, 5)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Camera could not be opened.")
            self.root.destroy()
            return

        self.calibration_mode = False
        self.calibration_done = False
        self.zone_calibration_mode = False
        self.division_line_points = []
        self.division_line_world_points = []
        self.zones_ready = False
        self.clicked_points = []
        self.H_img_to_world = None
        self.PLACE_POINT_WORLD = PLACE_POINT_WORLD
        self.detector = CubeDetector()
        self.cube_tracker = SavedCubeTracker()
        self.robot = RobotController(
            status_callback=self.set_status,
            idle_callback=self.root.update_idletasks,
        )

        self.build_layout()
        self.video_label.bind("<Button-1>", self.mouse_click)

        self.refresh_detection_log()
        self.update_frame()

    @property
    def detected_cube_points(self):
        return self.cube_tracker.detected_cube_points

    @detected_cube_points.setter
    def detected_cube_points(self, value):
        self.cube_tracker.detected_cube_points = value

    def build_layout(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        self.video_label = tk.Label(main_frame)
        self.video_label.grid(row=0, column=0, sticky="n")

        log_frame = tk.Frame(main_frame)
        log_frame.grid(row=0, column=1, padx=(12, 0), sticky="ns")

        log_title = tk.Label(
            log_frame,
            text="Detected Blocks",
            font=("Arial", 12, "bold"),
        )
        log_title.pack(anchor="w")

        self.block_log = tk.Text(
            log_frame,
            width=38,
            height=34,
            font=("Consolas", 10),
            state=tk.DISABLED,
        )
        self.block_log.pack(fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.calibrate_button = tk.Button(
            button_frame,
            text="Calibrate New Map",
            width=20,
            command=self.start_calibration,
        )
        self.calibrate_button.grid(row=0, column=0, padx=5)

        self.manual_button = tk.Button(
            button_frame,
            text="Manual Operation",
            width=20,
            command=self.manual_operation,
        )
        self.manual_button.grid(row=0, column=1, padx=5)

        self.auto_button = tk.Button(
            button_frame,
            text="Automatic Operation",
            width=20,
            command=self.automatic_operation,
        )
        self.auto_button.grid(row=0, column=2, padx=5)

        self.exit_button = tk.Button(
            button_frame,
            text="Exit",
            width=20,
            command=self.exit_program,
        )
        self.exit_button.grid(row=0, column=3, padx=5)

        self.status_label = tk.Label(
            self.root,
            text="Status: Not calibrated",
            font=("Arial", 12),
        )
        self.status_label.pack(pady=5)

    def set_status(self, text):
        self.status_label.config(text=text)

    def start_calibration(self):
        self.calibration_mode = True
        self.calibration_done = False
        self.zone_calibration_mode = False
        self.division_line_points = []
        self.division_line_world_points = []
        self.zones_ready = False
        self.clicked_points = []
        self.H_img_to_world = None
        self.cube_tracker.reset()
        self.refresh_detection_log()

        self.set_status(
            "Calibration mode: Click 4 workspace corners: TL, TR, BR, BL"
        )

        print("\nCalibration started.")
        print("Click 4 corners in this order:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")

    def manual_operation(self):
        if not self.calibration_done or not self.zones_ready:
            messagebox.showwarning(
                "Calibration Required",
                "Please calibrate the workspace and division zones first.",
            )
            return

        if not self.detected_cube_points:
            messagebox.showwarning(
                "No Blocks",
                "No blocks are currently saved in the detection zone.",
            )
            return

        self.set_status("Status: Manual pick running.")
        block = self.detected_cube_points[0]

        try:
            print("Manual operation selected.")
            print(f"Picking Block {block['id']}...")
            self.robot.pick_and_place_block(block, self.PLACE_POINT_WORLD)

            self.detected_cube_points.pop(0)
            self.refresh_detection_log()

            self.set_status("Status: Manual pick completed.")
            print("Manual operation completed.")
        except Exception as exc:
            self.set_status("Status: Manual pick failed.")
            messagebox.showerror("Robot Error", str(exc))
            print(f"Manual operation failed: {exc}")

    def automatic_operation(self):
        if not self.calibration_done or not self.zones_ready:
            messagebox.showwarning(
                "Calibration Required",
                "Please calibrate the workspace and division zones first.",
            )
            return

        if not self.detected_cube_points:
            messagebox.showwarning(
                "No Blocks",
                "No blocks are currently saved in the detection zone.",
            )
            return

        self.set_status("Status: Automatic pick running.")
        print("Automatic operation started.")

        try:
            while self.detected_cube_points:
                block = self.detected_cube_points[0]
                print(f"Picking Block {block['id']}...")
                self.set_status(f"Status: Picking Block {block['id']}.")
                self.root.update_idletasks()

                self.robot.pick_and_place_block(block, self.PLACE_POINT_WORLD)
                self.detected_cube_points.pop(0)
                self.refresh_detection_log()
                self.root.update_idletasks()

            self.set_status(
                "Status: Automatic operation completed. No blocks remaining."
            )
            print("Automatic operation completed. No blocks remaining.")
        except Exception as exc:
            self.set_status("Status: Automatic operation failed.")
            messagebox.showerror("Robot Error", str(exc))
            print(f"Automatic operation failed: {exc}")

    def exit_program(self):
        self.robot.disconnect_robot()
        self.cap.release()
        self.root.destroy()

    def mouse_click(self, event):
        if not self.calibration_mode and not self.zone_calibration_mode:
            return

        x = event.x
        y = event.y

        if self.zone_calibration_mode:
            self.division_line_points.append([x, y])
            xw, yw = self.pixel_to_world(x, y)
            self.division_line_world_points.append((xw, yw))

            print(
                f"Division line point {len(self.division_line_points)}: "
                f"pixel=({x}, {y}), world=({xw:.1f}, {yw:.1f})"
            )

            if len(self.division_line_points) == 2:
                self.zone_calibration_mode = False
                self.zones_ready = True
                self.cube_tracker.last_detection_update = 0
                self.set_status(
                    "Status: Calibration + zones completed. Detection active."
                )

                print("\nZone calibration completed.")
                print("Detection active.")

            return

        self.clicked_points.append([x, y])
        print(f"Calibration point {len(self.clicked_points)}: pixel=({x}, {y})")

        if len(self.clicked_points) == 4:
            self.H_img_to_world, status = calibration.find_workspace_homography(
                self.clicked_points
            )

            if self.H_img_to_world is None:
                messagebox.showerror(
                    "Calibration Error",
                    "Homography failed. Try clicking points again.",
                )
                self.clicked_points = []
                return

            self.calibration_done = True
            self.calibration_mode = False
            self.zone_calibration_mode = True
            self.set_status("Status: Click 2 division-line points.")

            print("\nCalibration completed.")
            print("Homography Matrix:")
            print(self.H_img_to_world)
            print("Click 2 points to define detection/placement division line.")

    def pixel_to_world(self, u, v):
        return calibration.pixel_to_world(self.H_img_to_world, u, v)

    def world_to_robot(self, xw, yw, zw=0):
        return calibration.world_to_robot(xw, yw, zw)

    def is_in_detection_zone(self, xw, yw):
        if not self.zones_ready:
            return False
        return calibration.is_in_detection_zone(
            self.division_line_world_points,
            xw,
            yw,
        )

    def update_saved_cube_points(self, live_points):
        self.cube_tracker.update_saved_cube_points(live_points)
        self.refresh_detection_log()

    def refresh_detection_log(self):
        if not hasattr(self, "block_log"):
            return

        lines = []
        lines.append(f"Saved blocks: {len(self.detected_cube_points)}")

        if (
            self.calibration_done
            and self.zones_ready
            and self.cube_tracker.last_detection_update
        ):
            elapsed = time.monotonic() - self.cube_tracker.last_detection_update
            next_update = max(0, DETECTION_UPDATE_SECONDS - elapsed)
            lines.append(f"Next update: {next_update:0.1f}s")
        elif self.calibration_done and self.zones_ready:
            lines.append("Next update: now")
        elif self.calibration_done:
            lines.append("Define zones to start")
        else:
            lines.append("Calibrate to start")

        lines.append("")

        if not self.detected_cube_points:
            lines.append("No saved blocks yet.")
        else:
            for point in self.detected_cube_points:
                px, py = point["pixel"]
                lines.append(f"Block {point['id']}")
                lines.append(f"  Pixel: {px}, {py}")

                if "world" in point:
                    xw, yw = point["world"]
                    lines.append(f"  World: {xw:0.1f}, {yw:0.1f} mm")

                if "robot" in point:
                    xr, yr, zr = point["robot"]
                    lines.append(f"  Robot: {xr:0.1f}, {yr:0.1f}, {zr:0.1f}")

                lines.append("")

        self.block_log.config(state=tk.NORMAL)
        self.block_log.delete("1.0", tk.END)
        self.block_log.insert(tk.END, "\n".join(lines))
        self.block_log.config(state=tk.DISABLED)

    def draw_calibration_points(self, frame):
        for i, pt in enumerate(self.clicked_points):
            x, y = pt
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"P{i + 1}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if 2 <= len(self.clicked_points) < 4:
            pts = np.array(self.clicked_points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (255, 0, 0), 2)

        if self.calibration_done:
            pts = np.array(self.clicked_points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    def draw_division_line(self, frame):
        if not self.division_line_points:
            return

        line_pts = np.array(self.division_line_points, dtype=np.int32)

        for i, pt in enumerate(line_pts):
            x, y = pt
            cv2.circle(frame, (x, y), 6, (255, 255, 0), -1)
            cv2.putText(
                frame,
                f"D{i + 1}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        if len(line_pts) == 2:
            cv2.line(frame, tuple(line_pts[0]), tuple(line_pts[1]), (255, 255, 0), 3)

    def draw_status_overlay(self, frame):
        if self.calibration_mode:
            cv2.putText(
                frame,
                f"Calibration: {len(self.clicked_points)}/4 points",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        elif self.zone_calibration_mode:
            cv2.putText(
                frame,
                "Click 2 points to define detection/placement division line",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        elif self.calibration_done and self.zones_ready:
            cv2.putText(
                frame,
                "Calibrated + Zones Ready: Detection Active",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Not Calibrated",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.set_status("Camera frame failed.")
            return

        frame = cv2.resize(frame, (800, 600))
        self.draw_calibration_points(frame)
        self.draw_division_line(frame)

        if self.calibration_done and self.zones_ready:
            frame, mask, detected_points = self.detector.detect_blocks(
                frame,
                calibration_ready=True,
                pixel_to_world_func=self.pixel_to_world,
                is_in_detection_zone_func=self.is_in_detection_zone,
                world_to_robot_func=self.world_to_robot,
            )

            should_update_saved_points = (
                self.cube_tracker.last_detection_update == 0
                or time.monotonic() - self.cube_tracker.last_detection_update
                >= DETECTION_UPDATE_SECONDS
            )

            if should_update_saved_points:
                self.update_saved_cube_points(detected_points)
            else:
                self.refresh_detection_log()

            draw_saved_cube_points(frame, self.detected_cube_points)

        self.draw_status_overlay(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(20, self.update_frame)

