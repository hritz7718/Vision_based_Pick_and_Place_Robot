import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import calibration
from config import CAMERA_ID, DETECTION_UPDATE_SECONDS
from detection import CubeDetector, SavedCubeTracker, draw_saved_cube_points
from placement import PlacementPlanner
from robot_commands import RobotController


APP_BG = "#d8d8d4"
APP_SURFACE = "#efefec"
PANEL_BG = "#f8f8f5"
CAMERA_BG = "#171918"
BORDER = "#b8bbb6"
TEXT_DARK = "#191b1d"
TEXT_MUTED = "#666a6d"
ACCENT = "#00a6a6"
ACCENT_DARK = "#087777"
CLASSIC_RED = "#b13b3b"


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

        self.frame_width = 800
        self.frame_height = 600
        self.calibration_mode = False
        self.calibration_done = False
        self.zone_calibration_mode = False
        self.workspace_calibration_mode = False
        self.division_line_points = []
        self.division_line_world_points = []
        self.workspace_points = []
        self.workspace_world_points = []
        self.workspace_ready = False
        self.zones_ready = False
        self.clicked_points = []
        self.H_img_to_world = None
        self.detector = CubeDetector()
        self.cube_tracker = SavedCubeTracker()
        self.placement_planner = PlacementPlanner()
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
        self.root.configure(bg=APP_BG)
        self.root.geometry("1240x820")
        self.root.minsize(1100, 760)

        title_bar = tk.Frame(self.root, bg=APP_SURFACE, height=44)
        title_bar.pack(fill=tk.X, padx=18, pady=(14, 0))
        title_bar.pack_propagate(False)

        window_dots = tk.Frame(title_bar, bg=APP_SURFACE)
        window_dots.pack(side=tk.LEFT, padx=(14, 0))
        for color in ("#9b9b96", "#b6b6b0", "#d0d0ca"):
            tk.Label(
                window_dots,
                text="●",
                fg=color,
                bg=APP_SURFACE,
                font=("Arial", 12, "bold"),
            ).pack(side=tk.LEFT, padx=2)

        tk.Label(
            title_bar,
            text="ROBOT VISION CALIBRATION SYSTEM",
            bg=APP_SURFACE,
            fg=TEXT_DARK,
            font=("Arial", 12, "bold"),
        ).pack(expand=True)

        shell = tk.Frame(
            self.root,
            bg=APP_SURFACE,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        shell.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 12))

        main_frame = tk.Frame(shell, bg=APP_SURFACE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        camera_frame = tk.Frame(
            main_frame,
            bg=CAMERA_BG,
            highlightbackground="#2f3333",
            highlightthickness=2,
        )
        camera_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 18))

        self.video_label = tk.Label(
            camera_frame,
            bg=CAMERA_BG,
            bd=0,
            highlightthickness=0,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        log_frame = tk.Frame(
            main_frame,
            bg=PANEL_BG,
            width=300,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        log_frame.grid(row=0, column=1, sticky="ns")
        log_frame.grid_propagate(False)

        self.log_title = tk.Label(
            log_frame,
            text="Detected Objects: 0",
            bg=PANEL_BG,
            fg=TEXT_DARK,
            font=("Arial", 13, "bold"),
            anchor="w",
        )
        self.log_title.pack(fill=tk.X, padx=14, pady=(14, 8))

        tk.Frame(log_frame, height=1, bg="#d4d4cf").pack(fill=tk.X)

        self.block_log = tk.Text(
            log_frame,
            width=34,
            height=31,
            font=("Consolas", 10),
            bg=PANEL_BG,
            fg=TEXT_DARK,
            insertbackground=TEXT_DARK,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=12,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.block_log.pack(fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(shell, bg=APP_SURFACE)
        button_frame.pack(fill=tk.X, padx=18, pady=(0, 12))

        for column in range(5):
            button_frame.grid_columnconfigure(column, weight=1)

        button_options = {
            "width": 20,
            "height": 2,
            "font": ("Arial", 10, "bold"),
            "relief": tk.FLAT,
            "bd": 0,
            "highlightthickness": 1,
            "highlightbackground": "#8e938d",
            "bg": APP_SURFACE,
            "fg": TEXT_DARK,
            "activebackground": "#dde9e8",
            "activeforeground": ACCENT_DARK,
            "disabledforeground": "#969a96",
            "cursor": "hand2",
        }

        self.calibrate_button = tk.Button(
            button_frame,
            text="Calibrate New Map",
            command=self.start_calibration,
            **button_options,
        )
        self.calibrate_button.grid(row=0, column=0, padx=6, sticky="ew")

        self.manual_button = tk.Button(
            button_frame,
            text="Manual Operation",
            command=self.manual_operation,
            **button_options,
        )
        self.manual_button.grid(row=0, column=1, padx=6, sticky="ew")

        self.auto_button = tk.Button(
            button_frame,
            text="Automatic Operation",
            command=self.automatic_operation,
            **button_options,
        )
        self.auto_button.grid(row=0, column=2, padx=6, sticky="ew")

        self.home_button = tk.Button(
            button_frame,
            text="Home Operation",
            command=self.home_operation,
            state=tk.DISABLED,
            **button_options,
        )
        self.home_button.grid(row=0, column=3, padx=6, sticky="ew")

        self.exit_button = tk.Button(
            button_frame,
            text="Exit",
            command=self.exit_program,
            **button_options,
        )
        self.exit_button.config(
            activebackground="#f0dddd",
            activeforeground=CLASSIC_RED,
        )
        self.exit_button.grid(row=0, column=4, padx=6, sticky="ew")

        self.status_label = tk.Label(
            shell,
            text="SYSTEM STATUS: AWAITING CALIBRATION",
            bg=APP_SURFACE,
            fg=TEXT_DARK,
            font=("Arial", 14, "bold"),
        )
        self.status_label.pack(pady=(0, 4))

        self.telemetry_label = tk.Label(
            shell,
            text="FEED: 800x600   •   FPS: 5   •   MODE: LIVE VISION",
            bg=APP_SURFACE,
            fg=TEXT_MUTED,
            font=("Arial", 9),
        )
        self.telemetry_label.pack(pady=(0, 12))

    def set_status(self, text):
        display_text = text
        if text.lower().startswith("status:"):
            display_text = "SYSTEM STATUS:" + text.split(":", 1)[1]
        self.status_label.config(text=display_text.upper())

    def update_home_button_state(self):
        if not hasattr(self, "home_button"):
            return

        if self.robot.can_home():
            self.home_button.config(state=tk.NORMAL)
        else:
            self.home_button.config(state=tk.DISABLED)

    def start_calibration(self):
        self.calibration_mode = True
        self.calibration_done = False
        self.zone_calibration_mode = False
        self.workspace_calibration_mode = False
        self.division_line_points = []
        self.division_line_world_points = []
        self.workspace_points = []
        self.workspace_world_points = []
        self.workspace_ready = False
        self.zones_ready = False
        self.clicked_points = []
        self.H_img_to_world = None
        self.cube_tracker.reset()
        self.placement_planner.reset()
        self.refresh_detection_log()

        self.set_status(
            "Calibration mode: Click 4 map corners: TL, TR, BR, BL"
        )

        print("\nCalibration started.")
        print("Click 4 map corners in this order:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")

    def manual_operation(self):
        if not self.calibration_done or not self.zones_ready or not self.workspace_ready:
            messagebox.showwarning(
                "Calibration Required",
                "Please calibrate the map, division zones, and workspace first.",
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
            place_point_world = self.placement_planner.next_place_point()
            block["place_world"] = place_point_world
            self.robot.pick_and_place_block(block, place_point_world)

            self.detected_cube_points.pop(0)
            self.refresh_detection_log()

            self.set_status("Status: Manual pick completed.")
            print("Manual operation completed.")
        except Exception as exc:
            self.set_status("Status: Manual pick failed.")
            messagebox.showerror("Robot Error", str(exc))
            print(f"Manual operation failed: {exc}")
        finally:
            self.update_home_button_state()

    def automatic_operation(self):
        if not self.calibration_done or not self.zones_ready or not self.workspace_ready:
            messagebox.showwarning(
                "Calibration Required",
                "Please calibrate the map, division zones, and workspace first.",
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

                place_point_world = self.placement_planner.next_place_point()
                block["place_world"] = place_point_world
                self.robot.pick_and_place_block(block, place_point_world)
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
        finally:
            self.update_home_button_state()

    def home_operation(self):
        if not self.robot.can_home():
            messagebox.showinfo(
                "Home Unavailable",
                "Home operation is only available when the robot is idle and not already at home.",
            )
            self.update_home_button_state()
            return

        try:
            self.home_button.config(state=tk.DISABLED)
            self.robot.home_robot()
        except Exception as exc:
            self.set_status("Status: Home operation failed.")
            messagebox.showerror("Robot Error", str(exc))
            print(f"Home operation failed: {exc}")
        finally:
            self.update_home_button_state()

    def exit_program(self):
        self.robot.disconnect_robot()
        self.cap.release()
        self.root.destroy()

    def get_frame_click_position(self, event):
        label_width = max(1, self.video_label.winfo_width())
        label_height = max(1, self.video_label.winfo_height())

        image_x_offset = max(0, (label_width - self.frame_width) // 2)
        image_y_offset = max(0, (label_height - self.frame_height) // 2)

        x = event.x - image_x_offset
        y = event.y - image_y_offset

        if x < 0 or y < 0 or x >= self.frame_width or y >= self.frame_height:
            return None, None

        return int(x), int(y)

    def mouse_click(self, event):
        if (
            not self.calibration_mode
            and not self.zone_calibration_mode
            and not self.workspace_calibration_mode
        ):
            return

        x, y = self.get_frame_click_position(event)
        if x is None or y is None:
            return

        if self.workspace_calibration_mode:
            self.workspace_points.append([x, y])
            xw, yw = self.pixel_to_world(x, y)
            self.workspace_world_points.append((xw, yw))

            print(
                f"Workspace point {len(self.workspace_points)}: "
                f"pixel=({x}, {y}), world=({xw:.1f}, {yw:.1f})"
            )

            if len(self.workspace_points) == 8:
                self.workspace_calibration_mode = False
                self.workspace_ready = True
                self.cube_tracker.last_detection_update = 0
                self.placement_planner.configure(
                    self.workspace_world_points,
                    self.division_line_world_points,
                    self.is_inside_workspace,
                    self.is_in_placement_zone,
                )
                self.set_status(
                    "Status: Calibration + zones + workspace completed. Detection active."
                )

                print("\nWorkspace plotting completed.")
                print(f"Placement points ready: {len(self.placement_planner.candidates)}")
                print("Detection active.")

            return

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
                self.workspace_calibration_mode = True
                self.set_status(
                    "Status: Click 8 workspace boundary points clockwise from top-left."
                )

                print("\nZone calibration completed.")
                print("Click 8 workspace boundary points in this order:")
                print("1. Top-left")
                print("2. Top edge middle")
                print("3. Top-right")
                print("4. Right edge middle")
                print("5. Bottom-right")
                print("6. Bottom edge middle")
                print("7. Bottom-left")
                print("8. Left edge middle")

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

    def is_in_placement_zone(self, xw, yw):
        if not self.zones_ready:
            return False
        return calibration.is_in_placement_zone(
            self.division_line_world_points,
            xw,
            yw,
        )

    def is_inside_workspace(self, xw, yw):
        if not self.workspace_ready:
            return False
        return calibration.is_inside_workspace(
            self.workspace_world_points,
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
        object_count = len(self.detected_cube_points)
        if hasattr(self, "log_title"):
            self.log_title.config(text=f"Detected Objects: {object_count}")

        lines.append(f"Saved blocks: {object_count}")

        if (
            self.calibration_done
            and self.zones_ready
            and self.workspace_ready
            and self.cube_tracker.last_detection_update
        ):
            elapsed = time.monotonic() - self.cube_tracker.last_detection_update
            next_update = max(0, DETECTION_UPDATE_SECONDS - elapsed)
            lines.append(f"Next update: {next_update:0.1f}s")
        elif self.calibration_done and self.zones_ready and not self.workspace_ready:
            lines.append("Define workspace to start")
        elif self.calibration_done and self.zones_ready:
            lines.append("Next update: now")
        elif self.calibration_done:
            lines.append("Define zones to start")
        else:
            lines.append("Calibrate to start")

        if self.calibration_done and self.zones_ready and self.workspace_ready:
            remaining_slots = max(
                0,
                len(self.placement_planner.candidates) - self.placement_planner.next_index,
            )
            lines.append(f"Placement slots: {remaining_slots}")

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

                if "place_world" in point:
                    xw, yw, zw = point["place_world"]
                    lines.append(f"  Place: {xw:0.1f}, {yw:0.1f}, {zw:0.1f}")

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

    def draw_workspace_points(self, frame):
        if not self.workspace_points:
            return

        for i, pt in enumerate(self.workspace_points):
            x, y = pt
            cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)
            cv2.putText(
                frame,
                f"W{i + 1}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )

        if len(self.workspace_points) >= 2:
            pts = np.array(self.workspace_points, dtype=np.int32)
            cv2.polylines(frame, [pts], self.workspace_ready, (255, 0, 255), 2)

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
                f"Map calibration: {len(self.clicked_points)}/4 points",
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
        elif self.workspace_calibration_mode:
            cv2.putText(
                frame,
                f"Workspace: {len(self.workspace_points)}/8 boundary points",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )
        elif self.calibration_done and self.zones_ready and self.workspace_ready:
            cv2.putText(
                frame,
                "Calibrated + Zones + Workspace Ready: Detection Active",
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

        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        self.draw_calibration_points(frame)
        self.draw_division_line(frame)
        self.draw_workspace_points(frame)

        if self.calibration_done and self.zones_ready and self.workspace_ready:
            frame, mask, detected_points = self.detector.detect_blocks(
                frame,
                calibration_ready=True,
                pixel_to_world_func=self.pixel_to_world,
                is_inside_workspace_func=self.is_inside_workspace,
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
