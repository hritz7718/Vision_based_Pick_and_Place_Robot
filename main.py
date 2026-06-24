import tkinter as tk

from ui import RobotVisionUI


def run_app():
    root = tk.Tk()
    app = RobotVisionUI(root)
    root.mainloop()
    return app


run_app()

