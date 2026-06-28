import time

from config import (
    MOVE_RX_DEG,
    MOVE_RY_DEG,
    MOVE_RZ_DEG,
    PUMP_SETTLE_SECONDS,
    ROBOT_ADDRESS,
    ROBOT_BAUD_RATE,
    ROBOT_COM_PORT,
    ROBOT_PICKUP_SPEED,
    ROBOT_PLACE_SPEED,
    ROBOT_SPEED,
    SAFE_TRAVEL_Z_OFFSET_MM,
    TOUCH_Z_MM,
)

MOVE_START_SETTLE_SECONDS = 0.35
MOVE_TIMEOUT_SECONDS = 45
POSITION_TOLERANCE_MM = 3.0

try:
    import serial
    import wlkatapython
except ImportError:
    serial = None
    wlkatapython = None


class RobotController:
    def __init__(self, status_callback=None, idle_callback=None):
        self.status_callback = status_callback
        self.idle_callback = idle_callback
        self.robot_serial = None
        self.robot_arm = None
        self.at_home = False
        self.current_position = None
        self.current_speed = None

    def set_status(self, text):
        if self.status_callback is not None:
            self.status_callback(text)

    def update_idle(self):
        if self.idle_callback is not None:
            self.idle_callback()

    def get_state(self):
        if self.robot_arm is None:
            return "Disconnected"
        try:
            return self.robot_arm.getState()
        except Exception:
            return "Unknown"

    def is_idle(self):
        return self.get_state() == "Idle"

    def is_at_home(self):
        return self.robot_arm is not None and self.at_home

    def can_home(self):
        return self.robot_arm is not None and self.is_idle() and not self.at_home

    def wait_until_idle(self, waiting_text, status_text):
        while self.get_state() != "Idle":
            print(waiting_text)
            self.set_status(status_text)
            self.update_idle()
            time.sleep(0.5)

    def read_robot_status(self):
        if self.robot_arm is None:
            return None

        try:
            status = self.robot_arm.getStatus()
        except Exception:
            return None

        if isinstance(status, dict):
            return status
        return None

    def status_position(self, status):
        try:
            return (
                float(status["coordinate_X"]),
                float(status["coordinate_Y"]),
                float(status["coordinate_Z"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def wait_until_position(self, x, y, z):
        target = (float(x), float(y), float(z))
        deadline = time.time() + MOVE_TIMEOUT_SECONDS
        last_position = None
        last_state = None

        time.sleep(MOVE_START_SETTLE_SECONDS)

        while time.time() < deadline:
            status = self.read_robot_status()
            if status is not None:
                last_state = status.get("state")
                last_position = self.status_position(status)
                if last_state == "Idle" and last_position is not None:
                    dx = abs(last_position[0] - target[0])
                    dy = abs(last_position[1] - target[1])
                    dz = abs(last_position[2] - target[2])
                    if max(dx, dy, dz) <= POSITION_TOLERANCE_MM:
                        return

            if last_position is None:
                print("Moving...")
            else:
                print(
                    "Moving... current="
                    f"({last_position[0]:.1f}, {last_position[1]:.1f}, {last_position[2]:.1f})"
                )
            self.set_status("Status: Robot moving...")
            self.update_idle()
            time.sleep(0.2)

        if last_position is None:
            last_position_text = "unknown"
        else:
            last_position_text = (
                f"({last_position[0]:.1f}, {last_position[1]:.1f}, {last_position[2]:.1f})"
            )
        raise RuntimeError(
            "Robot did not reach target "
            f"({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) "
            f"within {MOVE_TIMEOUT_SECONDS}s. "
            f"Last state={last_state}, last position={last_position_text}."
        )

    def connect_robot(self):
        if self.robot_arm is not None:
            return self.robot_arm

        if serial is None or wlkatapython is None:
            raise RuntimeError(
                "Robot libraries are missing. Install wlkatapython and pyserial "
                "inside opencvEnv."
            )

        print(f"Connecting robot on {ROBOT_COM_PORT} at {ROBOT_BAUD_RATE}...")
        self.robot_serial = serial.Serial(ROBOT_COM_PORT, ROBOT_BAUD_RATE, timeout=2)
        time.sleep(2)

        self.robot_arm = wlkatapython.Mirobot_UART(
            block_flag=False,
            message_flag=False,
        )
        self.robot_arm.init(self.robot_serial, ROBOT_ADDRESS)

        # Same homing style as Wlkata.py.
        self.robot_arm.homing()
        self.wait_until_idle("Homing...", "Status: Robot homing...")
        self.at_home = True
        print("Homing complete")

        self.robot_arm.speed(ROBOT_SPEED)
        self.current_speed = ROBOT_SPEED
        self.robot_arm.pump(0)

        print("Robot connected and homed.")
        return self.robot_arm

    def disconnect_robot(self):
        if self.robot_arm is not None:
            try:
                self.robot_arm.pump(0)
            except Exception:
                pass

        if self.robot_serial is not None and self.robot_serial.is_open:
            self.robot_serial.close()

        self.robot_arm = None
        self.robot_serial = None
        self.at_home = False
        self.current_position = None
        self.current_speed = None

    def home_robot(self):
        arm = self.connect_robot()

        if self.at_home:
            print("Robot is already at home.")
            return

        if not self.is_idle():
            raise RuntimeError("Robot must be idle before homing.")

        print("Home operation selected.")
        self.set_status("Status: Robot homing...")
        arm.homing()
        self.wait_until_idle("Homing...", "Status: Robot homing...")
        self.at_home = True
        print("Homing complete")
        self.set_status("Status: Robot homed.")

    def set_robot_speed(self, speed):
        arm = self.connect_robot()
        speed = int(speed)

        if self.current_speed == speed:
            return

        arm.speed(speed)
        self.wait_until_idle(
            f"Setting speed to {speed}...",
            "Status: Robot speed changing...",
        )
        self.current_speed = speed

    def move_robot_to(self, x, y, z=TOUCH_Z_MM, speed=None):
        arm = self.connect_robot()

        if speed is not None:
            self.set_robot_speed(speed)

        # Same absolute coordinate command style as Wlkata.py.
        arm.writecoordinate(
            0,
            0,
            round(float(x), 2),
            round(float(y), 2),
            round(float(z), 2),
            MOVE_RX_DEG,
            MOVE_RY_DEG,
            MOVE_RZ_DEG,
        )
        self.at_home = False

        self.wait_until_position(x, y, z)
        self.current_position = (float(x), float(y), float(z))

        print("Reached target point")

    def lift_to_safe_height_if_needed(self, safe_z):
        if self.current_position is None:
            return

        current_x, current_y, current_z = self.current_position
        if current_z < safe_z:
            print(f"Safety lift before travel to Z={safe_z:.1f}")
            self.move_robot_to(current_x, current_y, safe_z)

    def pick_block_at_map_coordinate(self, xw, yw, zw=0):
        print(f"Pick absolute map coordinate=({xw:.1f}, {yw:.1f}, {TOUCH_Z_MM})")
        self.move_robot_to(xw, yw, TOUCH_Z_MM, speed=ROBOT_PICKUP_SPEED)
        self.connect_robot().pump(1)
        self.wait_until_idle("Waiting for suction...", "Status: Suction activating...")
        time.sleep(PUMP_SETTLE_SECONDS)

    def place_block_at_map_coordinate(self, xw, yw, zw=0):
        print(f"Place absolute map coordinate=({xw:.1f}, {yw:.1f}, {TOUCH_Z_MM})")
        self.move_robot_to(xw, yw, TOUCH_Z_MM, speed=ROBOT_PLACE_SPEED)
        self.connect_robot().pump(0)
        self.wait_until_idle("Waiting for release...", "Status: Suction releasing...")
        time.sleep(PUMP_SETTLE_SECONDS)

    def pick_and_place_block(self, block, place_point_world):
        xw, yw = block["world"]
        place_xw = place_point_world[0]
        place_yw = place_point_world[1]
        safe_z = TOUCH_Z_MM + SAFE_TRAVEL_Z_OFFSET_MM

        print(
            "Pick/place sequence: approach high, descend to pick, "
            "lift high, travel high, descend to place, lift high."
        )

        self.lift_to_safe_height_if_needed(safe_z)

        print(f"Approach block above=({xw:.1f}, {yw:.1f}, {safe_z:.1f})")
        self.move_robot_to(xw, yw, safe_z, speed=ROBOT_SPEED)

        print(f"Descend to pick=({xw:.1f}, {yw:.1f}, {TOUCH_Z_MM:.1f})")
        self.move_robot_to(xw, yw, TOUCH_Z_MM, speed=ROBOT_PICKUP_SPEED)
        self.set_status("Status: Suction activating...")
        self.connect_robot().pump(1)
        self.wait_until_idle("Waiting for suction...", "Status: Suction activating...")
        time.sleep(PUMP_SETTLE_SECONDS)

        print(f"Lift picked block to Z={safe_z:.1f}")
        self.set_status("Status: Lifting picked block...")
        self.move_robot_to(xw, yw, safe_z, speed=ROBOT_PICKUP_SPEED)

        print(f"Travel high to placement=({place_xw:.1f}, {place_yw:.1f}, {safe_z:.1f})")
        self.set_status("Status: Moving to placement side...")
        self.move_robot_to(place_xw, place_yw, safe_z, speed=ROBOT_SPEED)

        print(f"Descend to place=({place_xw:.1f}, {place_yw:.1f}, {TOUCH_Z_MM:.1f})")
        self.move_robot_to(place_xw, place_yw, TOUCH_Z_MM, speed=ROBOT_PLACE_SPEED)
        self.set_status("Status: Releasing block...")
        self.connect_robot().pump(0)
        self.wait_until_idle("Waiting for release...", "Status: Releasing block...")
        time.sleep(PUMP_SETTLE_SECONDS)

        print(f"Lift after release to Z={safe_z:.1f}")
        self.set_status("Status: Lifting after placement...")
        self.move_robot_to(place_xw, place_yw, safe_z, speed=ROBOT_SPEED)
