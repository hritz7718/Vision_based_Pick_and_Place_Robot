import time

from config import (
    MOVE_RX_DEG,
    MOVE_RY_DEG,
    MOVE_RZ_DEG,
    PUMP_SETTLE_SECONDS,
    ROBOT_ADDRESS,
    ROBOT_BAUD_RATE,
    ROBOT_COM_PORT,
    ROBOT_SPEED,
    TOUCH_Z_MM,
)

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

    def set_status(self, text):
        if self.status_callback is not None:
            self.status_callback(text)

    def update_idle(self):
        if self.idle_callback is not None:
            self.idle_callback()

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
        while self.robot_arm.getState() != "Idle":
            print("Homing...")
            self.set_status("Status: Robot homing...")
            self.update_idle()
            time.sleep(0.5)
        print("Homing complete")

        self.robot_arm.speed(ROBOT_SPEED)
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

    def move_robot_to(self, x, y, z=TOUCH_Z_MM):
        arm = self.connect_robot()

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

        while arm.getState() != "Idle":
            print("Moving...")
            self.set_status("Status: Robot moving...")
            self.update_idle()
            time.sleep(0.5)

        print("Reached target point")

    def pick_block_at_map_coordinate(self, xw, yw, zw=0):
        print(f"Pick absolute map coordinate=({xw:.1f}, {yw:.1f}, {TOUCH_Z_MM})")
        self.move_robot_to(xw, yw, TOUCH_Z_MM)
        self.connect_robot().pump(1)
        time.sleep(PUMP_SETTLE_SECONDS)

    def place_block_at_map_coordinate(self, xw, yw, zw=0):
        print(f"Place absolute map coordinate=({xw:.1f}, {yw:.1f}, {TOUCH_Z_MM})")
        self.move_robot_to(xw, yw, TOUCH_Z_MM)
        self.connect_robot().pump(0)
        time.sleep(PUMP_SETTLE_SECONDS)

    def pick_and_place_block(self, block, place_point_world):
        xw, yw = block["world"]

        self.pick_block_at_map_coordinate(xw, yw, 0)
        self.place_block_at_map_coordinate(
            place_point_world[0],
            place_point_world[1],
            place_point_world[2],
        )

