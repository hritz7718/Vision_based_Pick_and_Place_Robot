import json
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageTk

import calibration
from config import (
    CAMERA_ID,
    CUBE_CLASS_TO_LINE,
    CUBE_MATCH_DISTANCE_PX,
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    DEFAULT_MAX_CUBE_AREA_PX,
    DEFAULT_MIN_CUBE_AREA_PX,
    DETECTION_UPDATE_SECONDS,
    HSV_COLOR_RANGES,
    HSV_SELECTED_CLASS_NAME,
    PLACEMENT_HISTORY_FILE,
    PLACEMENT_POINT_RADIUS_MM,
    PLACEMENT_VERIFY_DISTANCE_MM,
    SHOW_LIVE_DEBUG_TEXT,
    YOLO_TARGET_CLASS,
)
from detection import CubeDetector, SavedCubeTracker, YoloCubeDetector, draw_saved_cube_points
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
        self.display_width = 760
        self.display_height = 570
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
        self.yolo_detector = YoloCubeDetector()
        self.yolo_error_message = None
        self.detector.set_target_class(HSV_SELECTED_CLASS_NAME)
        self.yolo_detector.set_target_class(YOLO_TARGET_CLASS)
        self.tuning_mode = False
        self.area_drag_start = None
        self.area_drag_end = None
        self.latest_frame = None
        self.hsv_tuner_window = None
        self.hsv_tuner_preview = None
        self.hsv_tuner_vars = {}
        self.active_hsv_profile = next(iter(HSV_COLOR_RANGES.keys()))
        self.hsv_target_class = HSV_SELECTED_CLASS_NAME
        self.yolo_target_class = YOLO_TARGET_CLASS
        self.cube_tracker = SavedCubeTracker()
        self.live_detection_points = []
        self.placement_detections = []
        self.placed_blocks = []
        self.placement_history_path = Path(__file__).with_name(PLACEMENT_HISTORY_FILE)
        self.placement_planner = PlacementPlanner()
        self.robot = RobotController(
            status_callback=self.set_status,
            idle_callback=self.root.update_idletasks,
        )

        self.build_layout()
        self.video_label.bind("<Button-1>", self.mouse_click)
        self.video_label.bind("<B1-Motion>", self.mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self.mouse_release)

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
        self.root.geometry("1240x860")
        self.root.minsize(1100, 800)

        title_bar = tk.Frame(self.root, bg=APP_SURFACE, height=36)
        title_bar.pack(fill=tk.X, padx=12, pady=(8, 0))
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
        shell.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        main_frame = tk.Frame(shell, bg=APP_SURFACE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
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
        self.video_label.pack(padx=6, pady=6)

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
            height=14,
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
        self.build_detection_controls(log_frame)

        button_frame = tk.Frame(shell, bg=APP_SURFACE)
        button_frame.pack(fill=tk.X, padx=12, pady=(0, 8))

        for column in range(5):
            button_frame.grid_columnconfigure(column, weight=1)

        button_options = {
            "width": 20,
            "height": 1,
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
        self.telemetry_label.pack(pady=(0, 6))

    def build_detection_controls(self, parent):
        controls = tk.Frame(parent, bg=PANEL_BG)
        controls.pack(fill=tk.X, padx=12, pady=(0, 12))

        tk.Frame(controls, height=1, bg="#d4d4cf").pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            controls,
            text="Detection Tuning",
            bg=PANEL_BG,
            fg=TEXT_DARK,
            font=("Arial", 11, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        mode_row = tk.Frame(controls, bg=PANEL_BG)
        mode_row.pack(fill=tk.X, pady=(6, 2))

        tk.Label(
            mode_row,
            text="Mode",
            width=8,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Arial", 8),
            anchor="w",
        ).pack(side=tk.LEFT)

        self.detection_mode_var = tk.StringVar(value="HSV Mode")
        self.mode_menu = tk.OptionMenu(
            mode_row,
            self.detection_mode_var,
            "HSV Mode",
            "YOLO Mode",
            command=self.on_detection_mode_changed,
        )
        self.mode_menu.config(
            bg=APP_SURFACE,
            fg=TEXT_DARK,
            activebackground="#dde9e8",
            activeforeground=ACCENT_DARK,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=("Arial", 9, "bold"),
        )
        self.mode_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

        target_row = tk.Frame(controls, bg=PANEL_BG)
        target_row.pack(fill=tk.X, pady=(2, 6))

        tk.Label(
            target_row,
            text="Target",
            width=8,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Arial", 8),
            anchor="w",
        ).pack(side=tk.LEFT)

        self.target_class_var = tk.StringVar(value=self.hsv_target_class)
        self.target_class_menu = tk.OptionMenu(
            target_row,
            self.target_class_var,
            *self.get_target_class_options(),
            command=self.on_target_class_changed,
        )
        self.target_class_menu.config(
            bg=APP_SURFACE,
            fg=TEXT_DARK,
            activebackground="#dde9e8",
            activeforeground=ACCENT_DARK,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=("Arial", 9, "bold"),
        )
        self.target_class_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

        button_row = tk.Frame(controls, bg=PANEL_BG)
        button_row.pack(fill=tk.X, pady=(8, 0))

        self.tune_button = tk.Button(
            button_row,
            text="Open HSV Tuner",
            command=self.open_hsv_tuner,
            relief=tk.FLAT,
            bd=0,
            bg="#e5eeed",
            fg=TEXT_DARK,
            activebackground="#d0e7e5",
            activeforeground=ACCENT_DARK,
            font=("Arial", 9, "bold"),
            cursor="hand2",
        )
        self.tune_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        tk.Button(
            button_row,
            text="Reset HSV",
            command=self.reset_detection_settings,
            relief=tk.FLAT,
            bd=0,
            bg="#eee8e5",
            fg=TEXT_DARK,
            activebackground="#f0dddd",
            activeforeground=CLASSIC_RED,
            font=("Arial", 9, "bold"),
            cursor="hand2",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        tk.Button(
            button_row,
            text="Reset Blocks",
            command=self.reset_all_blocks,
            relief=tk.FLAT,
            bd=0,
            bg="#f0dddd",
            fg=TEXT_DARK,
            activebackground="#e8caca",
            activeforeground=CLASSIC_RED,
            font=("Arial", 9, "bold"),
            cursor="hand2",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

    def create_tuning_scale(self, parent, label, variable, from_, to):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill=tk.X, pady=1)

        tk.Label(
            row,
            text=label,
            width=8,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Arial", 8),
            anchor="w",
        ).pack(side=tk.LEFT)

        tk.Scale(
            row,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=self.on_hsv_tuner_changed,
            bg=PANEL_BG,
            fg=TEXT_DARK,
            troughcolor="#d8ddda",
            activebackground=ACCENT,
            highlightthickness=0,
            bd=0,
            length=150,
            showvalue=True,
            sliderlength=14,
            width=8,
            font=("Arial", 8),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def set_status(self, text):
        display_text = text
        if text.lower().startswith("status:"):
            display_text = "SYSTEM STATUS:" + text.split(":", 1)[1]
        self.status_label.config(text=display_text.upper())

    def get_detection_mode(self):
        if not hasattr(self, "detection_mode_var"):
            return "HSV Mode"
        return self.detection_mode_var.get()

    def get_target_class_options(self):
        if self.get_detection_mode() == "YOLO Mode":
            yolo_options = [
                class_name for class_name in CUBE_CLASS_TO_LINE
                if str(class_name).lower().endswith("_cube")
            ]
            if self.yolo_target_class not in yolo_options:
                yolo_options.insert(0, self.yolo_target_class)
            return yolo_options

        return list(HSV_COLOR_RANGES.keys())

    def refresh_target_class_menu(self):
        if not hasattr(self, "target_class_menu"):
            return

        options = self.get_target_class_options()
        selected = self.yolo_target_class if self.get_detection_mode() == "YOLO Mode" else self.hsv_target_class
        if selected not in options and options:
            selected = options[0]

        self.target_class_var.set(selected)
        menu = self.target_class_menu["menu"]
        menu.delete(0, "end")
        for option in options:
            menu.add_command(
                label=option,
                command=tk._setit(self.target_class_var, option, self.on_target_class_changed),
            )

        self.apply_target_class(selected, reset_saved=False)

    def apply_target_class(self, class_name, reset_saved=True):
        class_name = str(class_name).strip().lower()
        if self.get_detection_mode() == "YOLO Mode":
            self.yolo_target_class = class_name
            self.yolo_detector.set_target_class(class_name)
        else:
            self.hsv_target_class = class_name
            self.detector.set_target_class(class_name)
            if class_name in self.detector.color_ranges:
                self.active_hsv_profile = class_name
                self.detector.set_selected_profile(class_name)
                if hasattr(self, "hsv_tuner_profile_var"):
                    self.hsv_tuner_profile_var.set(class_name)
                if self.hsv_tuner_vars:
                    self.hsv_tuner_vars["h_min"].set(int(self.detector.lower_hsv[0]))
                    self.hsv_tuner_vars["h_max"].set(int(self.detector.upper_hsv[0]))
                    self.hsv_tuner_vars["s_min"].set(int(self.detector.lower_hsv[1]))
                    self.hsv_tuner_vars["s_max"].set(int(self.detector.upper_hsv[1]))
                    self.hsv_tuner_vars["v_min"].set(int(self.detector.lower_hsv[2]))
                    self.hsv_tuner_vars["v_max"].set(int(self.detector.upper_hsv[2]))

        if reset_saved:
            self.cube_tracker.reset()
            self.live_detection_points = []
            self.placement_detections = []
            self.refresh_detection_log()

    def on_target_class_changed(self, class_name):
        self.apply_target_class(class_name)
        self.set_status(f"Status: Target class set to {class_name}.")

    def on_detection_mode_changed(self, _value=None):
        if self.tuning_mode and self.get_detection_mode() == "YOLO Mode":
            self.tuning_mode = False
            self.tune_button.config(text="Tune Color/Area", bg="#e5eeed")

        self.refresh_target_class_menu()
        self.cube_tracker.reset()
        self.yolo_error_message = None
        self.refresh_detection_log()
        self.set_status(f"Status: {self.get_detection_mode()} active.")

    def open_hsv_tuner(self):
        if self.hsv_tuner_window is not None and self.hsv_tuner_window.winfo_exists():
            self.hsv_tuner_window.lift()
            return

        self.detection_mode_var.set("HSV Mode")
        self.on_detection_mode_changed()

        self.hsv_tuner_window = tk.Toplevel(self.root)
        self.hsv_tuner_window.title("HSV Mask Tuner")
        self.hsv_tuner_window.configure(bg=APP_SURFACE)
        self.hsv_tuner_window.geometry("980x690")
        self.hsv_tuner_window.protocol("WM_DELETE_WINDOW", self.close_hsv_tuner)

        preview_frame = tk.Frame(self.hsv_tuner_window, bg=CAMERA_BG)
        preview_frame.pack(side=tk.LEFT, padx=12, pady=12)

        self.hsv_tuner_preview = tk.Label(preview_frame, bg=CAMERA_BG)
        self.hsv_tuner_preview.pack(padx=8, pady=8)

        controls = tk.Frame(self.hsv_tuner_window, bg=PANEL_BG, width=290)
        controls.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=12)
        controls.pack_propagate(False)

        tk.Label(
            controls,
            text="HSV Mask Tuner",
            bg=PANEL_BG,
            fg=TEXT_DARK,
            font=("Arial", 14, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(
            controls,
            text="Choose a cube color, tune the mask, then click Set.",
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Arial", 9),
            wraplength=250,
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(0, 10))

        self.hsv_tuner_profile_var = tk.StringVar(value=self.active_hsv_profile)
        profile_menu = tk.OptionMenu(
            controls,
            self.hsv_tuner_profile_var,
            *HSV_COLOR_RANGES.keys(),
            command=self.on_hsv_tuner_profile_changed,
        )
        profile_menu.config(
            bg=APP_SURFACE,
            fg=TEXT_DARK,
            activebackground="#dde9e8",
            activeforeground=ACCENT_DARK,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            font=("Arial", 10, "bold"),
        )
        profile_menu.pack(fill=tk.X, padx=12, pady=(0, 8))

        selected = self.detector.color_ranges[self.active_hsv_profile]
        self.hsv_tuner_vars = {
            "h_min": tk.IntVar(value=int(selected["lower"][0])),
            "h_max": tk.IntVar(value=int(selected["upper"][0])),
            "s_min": tk.IntVar(value=int(selected["lower"][1])),
            "s_max": tk.IntVar(value=int(selected["upper"][1])),
            "v_min": tk.IntVar(value=int(selected["lower"][2])),
            "v_max": tk.IntVar(value=int(selected["upper"][2])),
            "min_area": tk.IntVar(value=self.detector.min_area),
            "max_area": tk.IntVar(value=self.detector.max_area),
        }

        self.create_hsv_tuner_scale(controls, "H Min", "h_min", 0, 179)
        self.create_hsv_tuner_scale(controls, "H Max", "h_max", 0, 179)
        self.create_hsv_tuner_scale(controls, "S Min", "s_min", 0, 255)
        self.create_hsv_tuner_scale(controls, "S Max", "s_max", 0, 255)
        self.create_hsv_tuner_scale(controls, "V Min", "v_min", 0, 255)
        self.create_hsv_tuner_scale(controls, "V Max", "v_max", 0, 255)
        self.create_hsv_tuner_scale(controls, "Min Area", "min_area", 0, 30000)
        self.create_hsv_tuner_scale(controls, "Max Area", "max_area", 1, 500000)

        button_row = tk.Frame(controls, bg=PANEL_BG)
        button_row.pack(fill=tk.X, padx=12, pady=(12, 0))

        tk.Button(
            button_row,
            text="Set HSV",
            command=self.apply_hsv_tuner_settings,
            relief=tk.FLAT,
            bd=0,
            bg="#d0e7e5",
            fg=TEXT_DARK,
            activebackground="#bfe0dd",
            activeforeground=ACCENT_DARK,
            font=("Arial", 10, "bold"),
            cursor="hand2",
        ).pack(fill=tk.X, pady=(0, 6))

        tk.Button(
            button_row,
            text="Reset This Color",
            command=self.reset_hsv_tuner_profile,
            relief=tk.FLAT,
            bd=0,
            bg="#eee8e5",
            fg=TEXT_DARK,
            activebackground="#f0dddd",
            activeforeground=CLASSIC_RED,
            font=("Arial", 10, "bold"),
            cursor="hand2",
        ).pack(fill=tk.X, pady=(0, 6))

        tk.Button(
            button_row,
            text="Cancel",
            command=self.close_hsv_tuner,
            relief=tk.FLAT,
            bd=0,
            bg=APP_SURFACE,
            fg=TEXT_DARK,
            activebackground="#dde9e8",
            activeforeground=ACCENT_DARK,
            font=("Arial", 10, "bold"),
            cursor="hand2",
        ).pack(fill=tk.X)

        self.set_status("Status: HSV tuner open. Preview shows the selected color mask.")

    def create_hsv_tuner_scale(self, parent, label, key, from_, to):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(
            row,
            text=label,
            width=8,
            bg=PANEL_BG,
            fg=TEXT_MUTED,
            font=("Arial", 9),
            anchor="w",
        ).pack(side=tk.LEFT)

        tk.Scale(
            row,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=self.hsv_tuner_vars[key],
            command=self.on_hsv_tuner_changed,
            bg=PANEL_BG,
            fg=TEXT_DARK,
            troughcolor="#d8ddda",
            activebackground=ACCENT,
            highlightthickness=0,
            bd=0,
            length=180,
            showvalue=True,
            sliderlength=16,
            width=10,
            font=("Arial", 8),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def on_hsv_tuner_profile_changed(self, profile_name):
        self.active_hsv_profile = profile_name
        self.detector.set_selected_profile(profile_name)
        self.hsv_tuner_vars["h_min"].set(int(self.detector.lower_hsv[0]))
        self.hsv_tuner_vars["h_max"].set(int(self.detector.upper_hsv[0]))
        self.hsv_tuner_vars["s_min"].set(int(self.detector.lower_hsv[1]))
        self.hsv_tuner_vars["s_max"].set(int(self.detector.upper_hsv[1]))
        self.hsv_tuner_vars["v_min"].set(int(self.detector.lower_hsv[2]))
        self.hsv_tuner_vars["v_max"].set(int(self.detector.upper_hsv[2]))
        self.cube_tracker.reset()
        self.refresh_detection_log()
        self.set_status(f"Status: Editing HSV profile {profile_name}.")

    def on_hsv_tuner_changed(self, _value=None):
        if not self.hsv_tuner_vars:
            return

        h_min = min(self.hsv_tuner_vars["h_min"].get(), self.hsv_tuner_vars["h_max"].get())
        h_max = max(self.hsv_tuner_vars["h_min"].get(), self.hsv_tuner_vars["h_max"].get())
        s_min = min(self.hsv_tuner_vars["s_min"].get(), self.hsv_tuner_vars["s_max"].get())
        s_max = max(self.hsv_tuner_vars["s_min"].get(), self.hsv_tuner_vars["s_max"].get())
        v_min = min(self.hsv_tuner_vars["v_min"].get(), self.hsv_tuner_vars["v_max"].get())
        v_max = max(self.hsv_tuner_vars["v_min"].get(), self.hsv_tuner_vars["v_max"].get())
        min_area = min(self.hsv_tuner_vars["min_area"].get(), self.hsv_tuner_vars["max_area"].get())
        max_area = max(self.hsv_tuner_vars["min_area"].get(), self.hsv_tuner_vars["max_area"].get())

        self.detector.update_settings(
            lower_hsv=(h_min, s_min, v_min),
            upper_hsv=(h_max, s_max, v_max),
            min_area=min_area,
            max_area=max_area,
            profile_name=self.active_hsv_profile,
        )

        self.cube_tracker.reset()
        self.refresh_detection_log()

    def reset_hsv_tuner_profile(self):
        profile_name = self.active_hsv_profile
        defaults = HSV_COLOR_RANGES[profile_name]
        self.hsv_tuner_vars["h_min"].set(defaults["lower"][0])
        self.hsv_tuner_vars["h_max"].set(defaults["upper"][0])
        self.hsv_tuner_vars["s_min"].set(defaults["lower"][1])
        self.hsv_tuner_vars["s_max"].set(defaults["upper"][1])
        self.hsv_tuner_vars["v_min"].set(defaults["lower"][2])
        self.hsv_tuner_vars["v_max"].set(defaults["upper"][2])
        self.hsv_tuner_vars["min_area"].set(DEFAULT_MIN_CUBE_AREA_PX)
        self.hsv_tuner_vars["max_area"].set(DEFAULT_MAX_CUBE_AREA_PX)
        self.area_drag_start = None
        self.area_drag_end = None
        self.on_hsv_tuner_changed()
        self.set_status("Status: Detection tuning reset.")

    def reset_detection_settings(self):
        for profile_name, defaults in HSV_COLOR_RANGES.items():
            self.detector.update_settings(
                lower_hsv=defaults["lower"],
                upper_hsv=defaults["upper"],
                min_area=DEFAULT_MIN_CUBE_AREA_PX,
                max_area=DEFAULT_MAX_CUBE_AREA_PX,
                profile_name=profile_name,
            )
        self.cube_tracker.reset()
        self.refresh_detection_log()
        self.set_status("Status: HSV profiles reset.")

    def reset_all_blocks(self):
        self.cube_tracker.reset()
        self.live_detection_points = []
        self.placement_detections = []
        self.placed_blocks = []
        self.placement_planner.reset_usage()
        self.save_placement_history()
        self.refresh_detection_log()
        self.set_status("Status: Detected and placed blocks reset.")

    def apply_hsv_tuner_settings(self):
        self.on_hsv_tuner_changed()
        self.close_hsv_tuner()
        self.detection_mode_var.set("HSV Mode")
        self.on_detection_mode_changed()
        self.resume_after_hsv_tuning()

    def resume_after_hsv_tuning(self):
        if self.calibration_done and self.zones_ready and self.workspace_ready:
            self.cube_tracker.reset()
            self.refresh_detection_log()
            self.set_status("Status: HSV settings applied. Detection active.")
            return

        if self.calibration_done and self.zones_ready:
            self.calibration_mode = False
            self.zone_calibration_mode = False
            self.workspace_calibration_mode = True
            self.refresh_detection_log()
            remaining = max(0, 8 - len(self.workspace_points))
            self.set_status(f"Status: HSV settings applied. Click {remaining} workspace points.")
            return

        if self.calibration_done:
            self.calibration_mode = False
            self.zone_calibration_mode = True
            self.workspace_calibration_mode = False
            self.refresh_detection_log()
            remaining = max(0, 2 - len(self.division_line_points))
            self.set_status(f"Status: HSV settings applied. Click {remaining} division-line points.")
            return

        self.start_calibration()

    def close_hsv_tuner(self):
        if self.hsv_tuner_window is not None and self.hsv_tuner_window.winfo_exists():
            self.hsv_tuner_window.destroy()
        self.hsv_tuner_window = None
        self.hsv_tuner_preview = None
        self.hsv_tuner_vars = {}

    def update_hsv_tuner_preview(self):
        if (
            self.hsv_tuner_window is None
            or not self.hsv_tuner_window.winfo_exists()
            or self.hsv_tuner_preview is None
            or self.latest_frame is None
        ):
            return

        profile = self.detector.color_ranges[self.active_hsv_profile]
        blurred = cv2.GaussianBlur(self.latest_frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        mask = cv2.inRange(hsv, profile["lower"], profile["upper"])

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=4)

        masked = cv2.bitwise_and(self.latest_frame, self.latest_frame, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.detector.min_area or area > self.detector.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(masked, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                masked,
                f"{self.active_hsv_profile} {int(area)}px",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            masked,
            "MASK PREVIEW - click Set HSV when ready",
            (18, self.frame_height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        preview_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(preview_rgb)
        image = image.resize((640, 480), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(image=image)
        self.hsv_tuner_preview.imgtk = image_tk
        self.hsv_tuner_preview.configure(image=image_tk)

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
            class_name = block.get("class_name", "green")
            _, placement_points = self.detect_current_workspace_points()
            place_point_world = self.reserve_free_place_point_for_class(class_name, placement_points)
            block["place_world"] = place_point_world
            block["sorting_line"] = self.placement_planner.get_line_for_class(class_name)
            print(
                "Reserved placement point "
                f"({place_point_world[0]:.1f}, {place_point_world[1]:.1f}) "
                f"for {class_name}."
            )
            self.robot.pick_and_place_block(block, place_point_world)

            pickup_points, placement_points = self.detect_current_workspace_points()
            self.update_placement_verification(placement_points)
            if self.block_still_on_detection_side(block, pickup_points):
                self.set_status("Status: Operation unsuccessful. Block still on detection side. Run again.")
                messagebox.showwarning(
                    "Operation Unsuccessful",
                    "The block is still detected on the detection side. Run the operation again.",
                )
                self.update_saved_cube_points(pickup_points)
                return

            placement_match = self.find_placement_match(block, place_point_world, placement_points)
            if placement_match is None:
                self.set_status("Status: Operation unsuccessful. Block not verified in placement zone. Run again.")
                messagebox.showwarning(
                    "Operation Unsuccessful",
                    "The block was not detected at the placement point. Run the operation again.",
                )
                self.update_saved_cube_points(pickup_points)
                return

            self.record_placed_block(
                block,
                place_point_world,
                block["sorting_line"],
                verified_point=placement_match,
            )

            self.detected_cube_points = [
                point for point in self.detected_cube_points
                if point.get("id") != block.get("id")
            ]
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
            self.set_status("Status: Calibrate map, zones, and workspace before automatic mode.")
            print("Automatic operation skipped: calibration, zones, or workspace not ready.")
            return

        if not self.detected_cube_points:
            self.set_status("Status: No saved target blocks for automatic mode.")
            print("Automatic operation skipped: no saved target blocks.")
            return

        self.set_status("Status: Automatic pick running.")
        print("Automatic operation started.")

        try:
            _, initial_placement_points = self.detect_current_workspace_points()
            occupied_placement_points = list(initial_placement_points)

            while self.detected_cube_points:
                block = self.detected_cube_points[0]
                print(f"Picking Block {block['id']}...")
                self.set_status(f"Status: Picking Block {block['id']}.")
                self.root.update_idletasks()

                class_name = block.get("class_name", "green")
                place_point_world = self.reserve_free_place_point_for_class(
                    class_name,
                    occupied_placement_points,
                )
                block["place_world"] = place_point_world
                block["sorting_line"] = self.placement_planner.get_line_for_class(class_name)
                print(
                    "Reserved placement point "
                    f"({place_point_world[0]:.1f}, {place_point_world[1]:.1f}) "
                    f"for {class_name}."
                )
                self.robot.pick_and_place_block(block, place_point_world)

                self.record_placed_block(
                    block,
                    place_point_world,
                    block["sorting_line"],
                    verified_point=None,
                )
                occupied_placement_points.append({"world": place_point_world[:2]})
                self.detected_cube_points = [
                    point for point in self.detected_cube_points
                    if point.get("id") != block.get("id")
                ]
                self.refresh_detection_log()
                self.root.update_idletasks()

            if not self.detected_cube_points:
                final_pickup_points, final_placement_points = self.detect_current_workspace_points()
                self.update_placement_verification(final_placement_points)
                unverified_count = sum(
                    1 for block in self.placed_blocks
                    if not block["verified"]
                )
                if final_pickup_points:
                    self.set_status(
                        "Status: Automatic operation completed. Some target blocks may remain."
                    )
                    print(
                        "Automatic operation completed for saved blocks, but "
                        f"{len(final_pickup_points)} target block(s) are still detected on the detection side."
                    )
                elif unverified_count:
                    self.set_status(
                        "Status: Automatic operation completed. Some placements are not verified."
                    )
                    print(
                        "Automatic operation completed. "
                        f"{unverified_count} placed block record(s) are still unverified."
                    )
                else:
                    self.set_status(
                        "Status: Automatic operation completed. No blocks remaining."
                    )
                    print("Automatic operation completed. No blocks remaining.")
        except Exception as exc:
            self.set_status("Status: Automatic operation failed.")
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

        image_x_offset = max(0, (label_width - self.display_width) // 2)
        image_y_offset = max(0, (label_height - self.display_height) // 2)

        display_x = event.x - image_x_offset
        display_y = event.y - image_y_offset

        if (
            display_x < 0
            or display_y < 0
            or display_x >= self.display_width
            or display_y >= self.display_height
        ):
            return None, None

        x = display_x * self.frame_width / self.display_width
        y = display_y * self.frame_height / self.display_height

        return int(x), int(y)

    def mouse_drag(self, event):
        if not self.tuning_mode:
            return

        x, y = self.get_frame_click_position(event)
        if x is None or y is None:
            return

        if self.area_drag_start is None:
            self.area_drag_start = (x, y)
        self.area_drag_end = (x, y)

    def mouse_release(self, event):
        if not self.tuning_mode or self.area_drag_start is None:
            return

        x, y = self.get_frame_click_position(event)
        if x is None or y is None:
            return

        self.area_drag_end = (x, y)
        x1, y1 = self.area_drag_start
        x2, y2 = self.area_drag_end
        area = abs(x2 - x1) * abs(y2 - y1)

        if area > 0:
            if "max_area" in self.hsv_tuner_vars:
                self.hsv_tuner_vars["max_area"].set(area)
            self.on_hsv_tuner_changed()
            self.set_status(f"Status: Max cube area set to {area} px.")

    def mouse_click(self, event):
        if self.tuning_mode:
            x, y = self.get_frame_click_position(event)
            if x is None or y is None:
                return
            self.area_drag_start = (x, y)
            self.area_drag_end = (x, y)
            return

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

    def world_to_pixel(self, xw, yw):
        return calibration.world_to_pixel(self.H_img_to_world, xw, yw)

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

    def detect_current_workspace_points(self):
        if not (self.calibration_done and self.zones_ready and self.workspace_ready):
            return [], []

        time.sleep(0.2)
        ret, frame = self.cap.read()
        if not ret:
            return [], []

        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        mode = self.get_detection_mode()

        try:
            if mode == "YOLO Mode":
                _, _, detected_points = self.yolo_detector.detect_blocks(
                    frame,
                    calibration_ready=True,
                    pixel_to_world_func=self.pixel_to_world,
                    is_inside_workspace_func=self.is_inside_workspace,
                    is_in_detection_zone_func=self.is_in_detection_zone,
                    world_to_robot_func=self.world_to_robot,
                    include_placement_points=True,
                )
                self.yolo_error_message = None
            else:
                _, _, detected_points = self.detector.detect_blocks(
                    frame,
                    calibration_ready=True,
                    pixel_to_world_func=self.pixel_to_world,
                    is_inside_workspace_func=self.is_inside_workspace,
                    is_in_detection_zone_func=self.is_in_detection_zone,
                    world_to_robot_func=self.world_to_robot,
                    include_placement_points=True,
                )
        except Exception as exc:
            self.yolo_error_message = str(exc)
            return [], []

        pickup_points = [
            point for point in detected_points
            if point.get("zone", "detection") == "detection"
        ]
        placement_points = [
            point for point in detected_points
            if point.get("zone") == "placement"
        ]
        self.live_detection_points = pickup_points
        self.placement_detections = placement_points
        return pickup_points, placement_points

    def block_still_on_detection_side(self, block, pickup_points):
        block_class = str(block.get("class_name", "")).lower()
        source_world = block.get("world")
        source_pixel = block.get("pixel")

        for point in pickup_points:
            point_class = str(point.get("class_name", "")).lower()
            if block_class and point_class and point_class != block_class:
                continue

            if source_world is not None and "world" in point:
                xw, yw = point["world"]
                distance = np.hypot(xw - source_world[0], yw - source_world[1])
                if distance <= PLACEMENT_VERIFY_DISTANCE_MM:
                    return True

            if source_pixel is not None and "pixel" in point:
                px, py = point["pixel"]
                distance = np.hypot(px - source_pixel[0], py - source_pixel[1])
                if distance <= CUBE_MATCH_DISTANCE_PX:
                    return True

        return False

    def reserve_free_place_point_for_class(self, class_name, occupied_points):
        min_clearance = PLACEMENT_POINT_RADIUS_MM * 2

        while True:
            line, line_index, place_point_world = self.placement_planner.peek_place_point_for_class(class_name)
            if not self.is_place_point_occupied(place_point_world, occupied_points, min_clearance):
                self.placement_planner.commit_place_point_for_class(class_name)
                return place_point_world

            print(
                "Skipping occupied placement point "
                f"line {line} index {line_index}: "
                f"({place_point_world[0]:.1f}, {place_point_world[1]:.1f})."
            )
            self.placement_planner.commit_place_point_for_class(class_name)

    def is_place_point_occupied(self, place_point_world, occupied_points, min_clearance):
        expected_x, expected_y = place_point_world[0], place_point_world[1]
        all_occupied_points = list(occupied_points)
        all_occupied_points.extend(
            {"world": point.get("place_world")}
            for point in self.placed_blocks
        )

        for point in all_occupied_points:
            world = point.get("world")
            if world is None:
                continue

            xw, yw = world[0], world[1]
            if np.hypot(xw - expected_x, yw - expected_y) < min_clearance:
                return True

        return False

    def find_placement_match(self, block, place_point_world, placement_points):
        block_class = str(block.get("class_name", "")).lower()
        expected_x, expected_y = place_point_world[0], place_point_world[1]

        for point in placement_points:
            point_class = str(point.get("class_name", "")).lower()
            if block_class and point_class and point_class != block_class:
                continue
            if "world" not in point:
                continue

            xw, yw = point["world"]
            distance = np.hypot(xw - expected_x, yw - expected_y)
            if distance <= PLACEMENT_VERIFY_DISTANCE_MM:
                return point

        return None

    def summarize_points_by_class(self, points):
        counts = Counter(str(point.get("class_name", "cube")).lower() for point in points)
        if not counts:
            return "0"
        return ", ".join(
            f"{name}: {count}" for name, count in sorted(counts.items())
        )

    def record_placed_block(self, block, place_point_world, sorting_line, verified_point=None):
        source_world = block.get("world")
        if source_world is not None:
            source_world = tuple(float(value) for value in source_world)

        verified_world = None
        if verified_point is not None and "world" in verified_point:
            verified_world = tuple(float(value) for value in verified_point["world"])

        record = {
            "id": len(self.placed_blocks) + 1,
            "source_block_id": block.get("id"),
            "class_name": block.get("class_name", "cube"),
            "source_pixel": block.get("pixel"),
            "source_world": source_world,
            "place_world": tuple(float(value) for value in place_point_world),
            "sorting_line": sorting_line,
            "verified": verified_point is not None,
            "verified_pixel": verified_point.get("pixel") if verified_point is not None else None,
            "verified_world": verified_world,
            "placed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if verified_point is not None:
            record["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.placed_blocks.append(record)
        self.save_placement_history()
        return record

    def save_placement_history(self):
        try:
            self.placement_history_path.write_text(
                json.dumps(self.placed_blocks, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"Could not save placement history: {exc}")

    def update_placement_verification(self, placement_points):
        self.placement_detections = placement_points
        changed = False

        for record in self.placed_blocks:
            if record["verified"]:
                continue

            expected = record.get("place_world")
            if not expected:
                continue

            expected_x, expected_y = expected[0], expected[1]
            expected_class = str(record.get("class_name", "")).lower()

            for point in placement_points:
                point_class = str(point.get("class_name", "")).lower()
                if expected_class and point_class and point_class != expected_class:
                    continue

                if "world" not in point:
                    continue

                xw, yw = point["world"]
                distance = np.hypot(xw - expected_x, yw - expected_y)

                if distance <= PLACEMENT_VERIFY_DISTANCE_MM:
                    record["verified"] = True
                    record["verified_pixel"] = point.get("pixel")
                    record["verified_world"] = tuple(float(value) for value in point["world"])
                    record["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    changed = True
                    break

        if changed:
            self.save_placement_history()

    def refresh_detection_log(self):
        if not hasattr(self, "block_log"):
            return

        lines = []
        object_count = len(self.detected_cube_points)
        if hasattr(self, "log_title"):
            self.log_title.config(text=f"Detected Objects: {object_count}")

        mode = self.get_detection_mode()
        lines.append(f"Saved blocks: {object_count}")
        lines.append(f"Mode: {mode}")
        if hasattr(self, "target_class_var"):
            lines.append(f"Target: {self.target_class_var.get()}")
        lines.append(
            f"Detection side: {len(self.live_detection_points)} "
            f"({self.summarize_points_by_class(self.live_detection_points)})"
        )
        lines.append(
            f"Placement side: {len(self.placement_detections)} "
            f"({self.summarize_points_by_class(self.placement_detections)})"
        )
        if mode == "YOLO Mode" and self.yolo_error_message:
            lines.append(f"YOLO error: {self.yolo_error_message}")

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
            remaining_slots = sum(
                max(0, len(candidates) - self.placement_planner.next_index_by_line.get(line, 0))
                for line, candidates in self.placement_planner.candidates_by_line.items()
            )
            lines.append(f"Placement slots: {remaining_slots}")
            for line, candidates in self.placement_planner.candidates_by_line.items():
                used = self.placement_planner.next_index_by_line.get(line, 0)
                lines.append(f"  Line {line}: {max(0, len(candidates) - used)}")

        if self.placed_blocks:
            verified_count = sum(1 for block in self.placed_blocks if block["verified"])
            lines.append(f"Placed history: {verified_count}/{len(self.placed_blocks)} verified")
            lines.append(f"Placement detections: {len(self.placement_detections)}")
            for record in self.placed_blocks[-6:]:
                status = "OK" if record["verified"] else "WAIT"
                lines.append(
                    f"  {status} #{record['id']} {record['class_name']} "
                    f"line {record['sorting_line']}"
                )

        if mode == "HSV Mode":
            lines.append(f"Editing HSV: {self.active_hsv_profile}")
            for name, hsv_range in self.detector.color_ranges.items():
                lower = hsv_range["lower"]
                upper = hsv_range["upper"]
                lines.append(
                    f"{name}: [{lower[0]}, {lower[1]}, {lower[2]}] "
                    f"- [{upper[0]}, {upper[1]}, {upper[2]}]"
                )
            lines.append(f"Area: {self.detector.min_area}-{self.detector.max_area} px")

        lines.append("")

        if not self.detected_cube_points:
            lines.append("No saved blocks yet.")
        else:
            for point in self.detected_cube_points:
                px, py = point["pixel"]
                lines.append(f"Block {point['id']}")
                if "class_name" in point:
                    lines.append(f"  Class: {point['class_name']}")
                if point.get("confidence") is not None:
                    lines.append(f"  Confidence: {point['confidence']:.2f}")
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

                if "sorting_line" in point:
                    lines.append(f"  Sorting line: {point['sorting_line']}")

                lines.append("")

        self.block_log.config(state=tk.NORMAL)
        self.block_log.delete("1.0", tk.END)
        self.block_log.insert(tk.END, "\n".join(lines))
        self.block_log.config(state=tk.DISABLED)

    def draw_calibration_points(self, frame):
        for i, pt in enumerate(self.clicked_points):
            x, y = pt
            cv2.circle(frame, (x, y), 3, (0, 0, 0), -1)
            if self.calibration_mode or SHOW_LIVE_DEBUG_TEXT:
                cv2.putText(
                    frame,
                    f"P{i + 1}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

        if 2 <= len(self.clicked_points) < 4:
            pts = np.array(self.clicked_points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 0, 0), 2)

        if self.calibration_done:
            pts = np.array(self.clicked_points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 0), 2)

    def draw_workspace_points(self, frame):
        if not self.workspace_points:
            return

        for i, pt in enumerate(self.workspace_points):
            x, y = pt
            cv2.circle(frame, (x, y), 3, (0, 0, 0), -1)
            if self.workspace_calibration_mode or SHOW_LIVE_DEBUG_TEXT:
                cv2.putText(
                    frame,
                    f"W{i + 1}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

        if len(self.workspace_points) >= 2:
            pts = np.array(self.workspace_points, dtype=np.int32)
            cv2.polylines(frame, [pts], self.workspace_ready, (0, 0, 0), 2)

    def draw_division_line(self, frame):
        if not self.division_line_points:
            return

        line_pts = np.array(self.division_line_points, dtype=np.int32)

        for i, pt in enumerate(line_pts):
            x, y = pt
            cv2.circle(frame, (x, y), 3, (0, 0, 0), -1)
            if self.zone_calibration_mode or SHOW_LIVE_DEBUG_TEXT:
                cv2.putText(
                    frame,
                    f"D{i + 1}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

        if len(line_pts) == 2:
            cv2.line(frame, tuple(line_pts[0]), tuple(line_pts[1]), (0, 0, 0), 2)

    def draw_zone_side_labels(self, frame):
        if not self.zones_ready or len(self.division_line_points) != 2:
            return

        p1 = np.array(self.division_line_points[0], dtype=np.float32)
        p2 = np.array(self.division_line_points[1], dtype=np.float32)
        direction = p2 - p1
        length = float(np.linalg.norm(direction))
        if length < 1:
            return

        midpoint = (p1 + p2) / 2.0
        normal = np.array([-direction[1], direction[0]], dtype=np.float32) / length
        candidates = [midpoint + normal * 95, midpoint - normal * 95]

        for candidate in candidates:
            u = int(round(np.clip(candidate[0], 20, self.frame_width - 220)))
            v = int(round(np.clip(candidate[1], 45, self.frame_height - 20)))

            try:
                xw, yw = self.pixel_to_world(u, v)
                label = "Detection Side" if self.is_in_detection_zone(xw, yw) else "Placement Side"
            except Exception:
                continue

            cv2.putText(
                frame,
                label,
                (u, v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                2,
            )

    def draw_placement_slots(self, frame):
        if not self.workspace_ready:
            return

        for line, candidates in self.placement_planner.candidates_by_line.items():
            used = self.placement_planner.next_index_by_line.get(line, 0)

            for index, point in enumerate(candidates):
                xw, yw, zw = point
                u, v = self.world_to_pixel(xw, yw)
                u = int(round(u))
                v = int(round(v))

                if not (0 <= u < self.frame_width and 0 <= v < self.frame_height):
                    continue

                color = (0, 0, 0)
                if index < used:
                    color = (80, 80, 80)

                radius_u, radius_v = self.world_to_pixel(xw + PLACEMENT_POINT_RADIUS_MM, yw)
                radius_px = max(3, int(round(np.hypot(radius_u - u, radius_v - v))))
                cv2.circle(frame, (u, v), radius_px, color, 1)
                cv2.circle(frame, (u, v), 3, color, 1)
                if index == 0 and SHOW_LIVE_DEBUG_TEXT:
                    cv2.putText(
                        frame,
                        f"Line {line}",
                        (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )

        for record in self.placed_blocks:
            xw, yw, zw = record["place_world"]
            u, v = self.world_to_pixel(xw, yw)
            u = int(round(u))
            v = int(round(v))

            if not (0 <= u < self.frame_width and 0 <= v < self.frame_height):
                continue

            color = (0, 255, 0) if record["verified"] else (0, 255, 255)
            label = "OK" if record["verified"] else "WAIT"
            cv2.circle(frame, (u, v), 11, color, 2)
            if SHOW_LIVE_DEBUG_TEXT:
                cv2.putText(
                    frame,
                    f"{label} {record['class_name']}",
                    (u + 12, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

    def draw_area_selection(self, frame):
        if self.area_drag_start is None or self.area_drag_end is None:
            return

        x1, y1 = self.area_drag_start
        x2, y2 = self.area_drag_end
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        area = (right - left) * (bottom - top)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"Max area: {area} px",
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

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
            mode = self.get_detection_mode()
            cv2.putText(
                frame,
                f"{mode}: Ready",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
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

        if self.tuning_mode:
            cv2.putText(
                frame,
                "TUNING MODE: adjust HSV sliders or drag cube to set max area",
                (20, self.frame_height - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.set_status("Camera frame failed.")
            return

        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        self.latest_frame = frame.copy()
        self.update_hsv_tuner_preview()

        self.draw_calibration_points(frame)
        self.draw_division_line(frame)
        self.draw_zone_side_labels(frame)
        self.draw_workspace_points(frame)
        self.draw_placement_slots(frame)

        mode = self.get_detection_mode()

        if self.tuning_mode:
            frame, mask, detected_points = self.detector.detect_blocks(
                frame,
                calibration_ready=False,
            )
        elif self.calibration_done and self.zones_ready and self.workspace_ready:
            if mode == "YOLO Mode":
                try:
                    frame, mask, detected_points = self.yolo_detector.detect_blocks(
                        frame,
                        calibration_ready=True,
                        pixel_to_world_func=self.pixel_to_world,
                        is_inside_workspace_func=self.is_inside_workspace,
                        is_in_detection_zone_func=self.is_in_detection_zone,
                        world_to_robot_func=self.world_to_robot,
                        include_placement_points=True,
                    )
                    self.yolo_error_message = None
                except Exception as exc:
                    detected_points = []
                    self.yolo_error_message = str(exc)
                    cv2.putText(
                        frame,
                        f"YOLO error: {self.yolo_error_message}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
            else:
                frame, mask, detected_points = self.detector.detect_blocks(
                    frame,
                    calibration_ready=True,
                    pixel_to_world_func=self.pixel_to_world,
                    is_inside_workspace_func=self.is_inside_workspace,
                    is_in_detection_zone_func=self.is_in_detection_zone,
                    world_to_robot_func=self.world_to_robot,
                    include_placement_points=True,
                )

            pickup_points = [
                point for point in detected_points
                if point.get("zone", "detection") == "detection"
            ]
            placement_points = [
                point for point in detected_points
                if point.get("zone") == "placement"
            ]
            self.live_detection_points = pickup_points
            self.placement_detections = placement_points
            self.update_placement_verification(placement_points)

            should_update_saved_points = (
                self.cube_tracker.last_detection_update == 0
                or time.monotonic() - self.cube_tracker.last_detection_update
                >= DETECTION_UPDATE_SECONDS
            )

            if should_update_saved_points:
                self.update_saved_cube_points(pickup_points)
            else:
                self.refresh_detection_log()

            draw_saved_cube_points(frame, self.detected_cube_points, show_text=SHOW_LIVE_DEBUG_TEXT)

        self.draw_area_selection(frame)
        self.draw_status_overlay(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((self.display_width, self.display_height), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(20, self.update_frame)
