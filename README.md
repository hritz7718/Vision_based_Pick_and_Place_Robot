# Robot Vision Calibration System

User guide for the cube detection and robot sorting application.

This project uses a camera, OpenCV detection, map calibration, workspace zones, and a Wlkata Mirobot arm to pick colored cubes from the detection side and place them into sorted lines on the placement side.

## What This App Does

- Shows a live camera feed of the workspace.
- Calibrates the camera view to real world robot coordinates.
- Lets you draw:
  - the map calibration area,
  - the detection/placement divider,
  - the valid workspace boundary.
- Detects colored cubes using HSV or YOLO.
- Sorts cubes by color/class into placement lines.
- Verifies whether a block was actually placed.
- Keeps a placement history.
- Provides reset controls for detection and placement state.

## Main Controls

`Calibrate New Map`

Starts the full calibration process again. Use this when the camera, board, or robot setup has moved.

`Manual Operation`

Picks and places the first saved block from the detection side.

`Automatic Operation`

Runs through all saved blocks on the detection side and places them one by one.

`Home Operation`

Homes the robot. This button is available only when the robot is idle and not already at home.

`Open HSV Tuner`

Opens a separate masked preview window where you can tune HSV ranges and area limits for each cube color.

`Reset HSV`

Restores HSV profiles to the default values from `config.py`.

`Reset Blocks`

Clears saved detected blocks, placed block history, and placement line occupancy. Use this when the robot failed to pick/place a block but the UI still thinks the line is occupied.

## Detection Modes

### HSV Mode

HSV mode detects cubes using color thresholds and contour area filtering.

Current HSV profiles are configured in `config.py`:

- green
- purple
- orange

HSV mode is useful when lighting is stable and you want fast color-based detection.

### YOLO Mode

YOLO mode uses the configured YOLO model path in `config.py`.

YOLO detects cube classes such as:

- green / green_cube
- purple / purple_cube
- orange / orange_cube

YOLO mode is useful when HSV color thresholds are unreliable or when multiple cube classes should be detected at once.

## Normal Workflow

1. Start the app:

   ```powershell
   python main.py
   ```

2. Click `Calibrate New Map`.

3. Click the 4 map corners in this order:

   ```text
   1. Top-left
   2. Top-right
   3. Bottom-right
   4. Bottom-left
   ```

4. Click 2 points to draw the line that separates:

   ```text
   Detection Side
   Placement Side
   ```

5. Click 8 workspace boundary points clockwise from the top-left.

6. Confirm the live feed shows:

   ```text
   HSV Mode: Ready
   ```

   or:

   ```text
   YOLO Mode: Ready
   ```

7. Place cubes only inside the detection side.

8. Wait for the sidebar to show detected blocks.

9. Run `Manual Operation` or `Automatic Operation`.

## Placement Behavior

The placement side is divided into 3 sorting lines.

Default mapping in `config.py`:

```python
CUBE_CLASS_TO_LINE = {
    "green": 1,
    "green_cube": 1,
    "purple": 2,
    "purple_cube": 2,
    "orange": 3,
    "orange_cube": 3,
}
```

The app generates placement points inside the calibrated placement zone. Current config targets at least 8 slots per line when the physical placement zone is large enough:

```python
PLACEMENT_MIN_SLOTS_PER_LINE = 8
PLACEMENT_SPACING_MM = 25
PLACEMENT_MIN_SPACING_MM = 18
```

If the placement zone is too small, the app uses the best valid layout it can generate.

## Pick And Place Sequence

For each block, the robot should:

1. Move above the cube at safe height.
2. Move down to pickup Z.
3. Turn suction on.
4. Lift the cube by the safe travel height.
5. Move to the placement side while lifted.
6. Move down to the placement point.
7. Turn suction off.
8. Lift again before moving to the next block.

Important Z values are in `config.py`:

```python
TOUCH_Z_MM = 44.7
SAFE_TRAVEL_Z_OFFSET_MM = 50
```

## Placement Verification

The app does not mark a block as placed just because the robot command ran.

After movement, it checks:

- whether the block is still visible on the detection side,
- whether the correct class/color is visible near the expected placement point.

If the cube is still on the detection side, the UI reports the operation as unsuccessful and keeps the placement slot free.

If the cube is not detected at the placement target, the UI also reports the operation as unsuccessful.

Use `Reset Blocks` if the physical state and UI state become mismatched.

## Sidebar Information

The right-side log shows:

- saved detection-side blocks,
- detection mode,
- live detection-side block count by class,
- live placement-side block count by class,
- remaining placement slots,
- placed history and verification status,
- HSV values when HSV mode is active.

This is the quickest place to check whether the app still sees cubes on the detection side after a pick attempt.

## HSV Tuning

Click `Open HSV Tuner`.

In the tuner:

1. Choose the color profile.
2. Adjust H, S, V min/max sliders.
3. Adjust min/max cube area if needed.
4. Use the masked preview to confirm only the target cube is visible.
5. Click `Set HSV`.

If calibration is already complete, HSV tuning does not force you to recalibrate the map again.

## Important Configuration

Most user-adjustable values are in `config.py`.

Camera:

```python
CAMERA_ID = 1
```

Robot connection:

```python
ROBOT_COM_PORT = "COM17"
ROBOT_BAUD_RATE = 115200
ROBOT_SPEED = 30
```

Placement capacity:

```python
PLACEMENT_MIN_SLOTS_PER_LINE = 8
PLACEMENT_SPACING_MM = 25
PLACEMENT_MIN_SPACING_MM = 18
```

Robot reach limits:

```python
ROBOT_REACH_X_MIN_MM = 40
ROBOT_REACH_X_MAX_MM = 240
ROBOT_REACH_Y_MIN_MM = -180
ROBOT_REACH_Y_MAX_MM = 180
```

Set `SHOW_LIVE_DEBUG_TEXT = True` only when you need detailed overlay labels. Keep it `False` for normal use.

## Troubleshooting

### Camera does not open

Check `CAMERA_ID` in `config.py`. Try `0`, `1`, or `2` depending on your connected camera.

### Blocks are detected outside the workspace

Recalibrate the map and workspace. Make sure the 4 map points and 8 workspace points are clicked accurately.

### Purple/orange/green is detected incorrectly

Open the HSV tuner and adjust that color profile. Lighting changes can affect HSV detection a lot.

### Robot picks the block but does not place it

Watch the status label. The expected sequence is:

```text
Suction activating...
Lifting picked block...
Moving to placement side...
Releasing block...
Lifting after placement...
```

If it stops at suction, check the pump command and robot idle state.

### UI says placement line is occupied but the block is not there

Click `Reset Blocks`. This clears the placement history and frees all placement slots.

### Automatic operation says unsuccessful

That means at least one block is still detected on the detection side, or the block was not verified at the placement point. Correct the physical block position and run the operation again.

## Files

```text
main.py                 Starts the app
ui.py                   Tkinter user interface and workflow
detection.py            HSV and YOLO detection
calibration.py          Pixel/world/robot coordinate conversion
placement.py            Placement point generation and line assignment
robot_commands.py       Robot movement, suction, homing
config.py               User-adjustable settings
placement_history.json  Saved placement records
```

## Safety Notes

- Keep the robot workspace clear before running automatic operation.
- Use safe Z height values that avoid collisions.
- Do not place cubes outside the calibrated detection side.
- Recalibrate after moving the camera, board, robot, or workspace.
- Use manual operation first after any major calibration change.
