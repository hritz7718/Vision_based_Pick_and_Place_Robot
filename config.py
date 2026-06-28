import numpy as np


CAMERA_ID = 1

WORKSPACE_LENGTH_MM = 400
WORKSPACE_WIDTH_MM = 620

ROBOT_COM_PORT = "COM17"
ROBOT_BAUD_RATE = 115200
ROBOT_ADDRESS = -1
ROBOT_SPEED = 30
ROBOT_PICKUP_SPEED = 10
ROBOT_PLACE_SPEED = 15

TOUCH_Z_MM = 44.7
SAFE_TRAVEL_Z_OFFSET_MM = 50
MOVE_RX_DEG = 0
MOVE_RY_DEG = 0
MOVE_RZ_DEG = 0
PUMP_SETTLE_SECONDS = 0.5

WORLD_POINTS = np.array([
    [310, 310],
    [310, -310],
    [-90, -310],
    [-90, 310],
], dtype=np.float32)

WORKSPACE_WORLD_POINTS = np.array([
    [310, 310],
    [310, 0],
    [310, -310],
    [110, -310],
    [-90, -310],
    [-90, 0],
    [-90, 310],
    [110, 310],
], dtype=np.float32)

ROBOT_BASE_WORLD = np.array([0, 0, 0], dtype=np.float32)
PLACE_POINT_WORLD = np.array([0, 0, 0], dtype=np.float32)

DETECTION_UPDATE_SECONDS = 10
CUBE_MATCH_DISTANCE_PX = 60
PLACEMENT_MARGIN_MM = 25
PLACEMENT_SPACING_MM = 25
PLACEMENT_MIN_SPACING_MM = 18
PLACEMENT_POINT_RADIUS_MM = 10
PLACEMENT_MIN_SLOTS_PER_LINE = 8
PLACEMENT_VERIFY_DISTANCE_MM = 45
PLACEMENT_HISTORY_FILE = "placement_history.json"
PLACEMENT_LINE_CLEARANCE_MM = 70
PLACEMENT_DEPTH_MIN_RATIO = 0.35
PLACEMENT_DEPTH_MAX_RATIO = 0.80

# Keep placement slots inside a comfortable robot XY range.
# These are robot coordinates after subtracting ROBOT_BASE_WORLD.
ROBOT_REACH_X_MIN_MM = 40
ROBOT_REACH_X_MAX_MM = 240
ROBOT_REACH_Y_MIN_MM = -180
ROBOT_REACH_Y_MAX_MM = 180

DEFAULT_HSV_LOWER = (38, 28, 26)
DEFAULT_HSV_UPPER = (84, 255, 255)
DEFAULT_MIN_CUBE_AREA_PX = 15
DEFAULT_MAX_CUBE_AREA_PX = 10000
HSV_SELECTED_CLASS_NAME = "green"

# HSV mode loops over every profile here. Tune these values for your lighting.
HSV_COLOR_RANGES = {
    "green": {
        "lower": (38, 28, 26),
        "upper": (84, 255, 255),
    },
    "purple": {
        "lower": (120, 40, 40),
        "upper": (165, 255, 255),
    },
    "orange": {
        "lower": (0, 80, 80),
        "upper": (25, 255, 255),
    },
}

HSV_DRAW_COLORS = {
    "green": (0, 255, 0),
    "purple": (255, 0, 255),
    "orange": (0, 165, 255),
}

YOLO_MODELS = {
    "YOLOv8": r"C:\Users\Administrator\Desktop\Open CV\Cube Detector\Myproject\runs\detect\train\weights\best.pt",
}
DEFAULT_YOLO_MODEL_NAME = "YOLOv8"
YOLO_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 640
YOLO_TARGET_CLASS = "green_cube"
YOLO_MAX_FPS = 20

# Configure which placement line each YOLO class/color should use.
# Line 1 is the first generated placement row, line 2 is the next row, etc.
CUBE_CLASS_TO_LINE = {
    "green": 1,
    "green_cube": 1,
    "purple": 2,
    "purple_cube": 2,
    "orange": 3,
    "orange_cube": 3,
}
DEFAULT_SORTING_LINE = 1
SORTING_LINE_COUNT = 3

# Keep the live camera readable. Set True if you need all debug labels again.
SHOW_LIVE_DEBUG_TEXT = False
