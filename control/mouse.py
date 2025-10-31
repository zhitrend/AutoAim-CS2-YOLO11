import ctypes
import atexit  # 注册程序结束时自动执行的函数

# # 常量
# SPI_GETMOUSE = 0x0003
# SPI_SETMOUSE = 0x0004

# # 获取原始鼠标设置
# original_params = (ctypes.c_int * 3)()
# ctypes.windll.user32.SystemParametersInfoW(SPI_GETMOUSE, 0, original_params, 0)

# # 禁用鼠标加速度
# disable_params = (ctypes.c_int * 3)(0, 0, 0)
# ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, disable_params, 0)

# # 程序退出时还原鼠标设置
# def restore_mouse_settings():
#     ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, original_params, 0)
#     print("已恢复鼠标加速度设置")

# 定义结构体
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUT)]
    _anonymous_ = ("union",)

# 常量
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001  # 注意：不加 ABSOLUTE

# 相对移动函数
def move_mouse_relative(dx, dy):
    inp = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(
        dx=dx,
        dy=dy,
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE,  # 相对移动
        time=0,
        dwExtraInfo=None
    ))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long),
                ("y", ctypes.c_long)]

def get_mouse_position():
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

# 移动到指定位置
def move_mouse_to(x, y):
    curr_x, curr_y = get_mouse_position()
    dx = x - curr_x
    dy = y - curr_y
    move_mouse_relative(dx, dy)


def move_mouse_to_smooth(x, y, gain: float = 0.35, max_step: int = 25, deadzone: int = 1):
    """Move the cursor smoothly towards (x, y) using proportional control.

    - gain: fraction of remaining distance to move this frame
    - max_step: clamp per-frame pixel step to avoid spikes
    - deadzone: within this radius, do not move to reduce jitter
    """
    curr_x, curr_y = get_mouse_position()
    dx = x - curr_x
    dy = y - curr_y

    # Deadzone to prevent micro jitter
    if abs(dx) <= deadzone and abs(dy) <= deadzone:
        return

    step_x = int(max(-max_step, min(max_step, dx * gain)))
    step_y = int(max(-max_step, min(max_step, dy * gain)))
    if step_x == 0 and dx != 0:
        step_x = 1 if dx > 0 else -1
    if step_y == 0 and dy != 0:
        step_y = 1 if dy > 0 else -1
    move_mouse_relative(step_x, step_y)


def move_mouse_to_duration(x, y, duration_s: float = 0.1, steps: int = 10):
    """Move the cursor to (x, y) over a fixed duration, split into N steps.

    This gives deterministic speed: total time ~= duration_s, with evenly spaced steps.
    """
    if steps <= 0:
        steps = 1
    if duration_s < 0:
        duration_s = 0.0

    import time as _t

    per_step_sleep = duration_s / float(steps) if steps > 0 else 0.0
    for _ in range(steps):
        curr_x, curr_y = get_mouse_position()
        dx = x - curr_x
        dy = y - curr_y
        # If already at target, stop early
        if dx == 0 and dy == 0:
            break
        # Move a fraction each step; recompute to remain robust if external forces move the mouse
        step_x = int(round(dx / max(1, (steps))))
        step_y = int(round(dy / max(1, (steps))))
        # Ensure we make progress even for tiny residuals
        if step_x == 0 and dx != 0:
            step_x = 1 if dx > 0 else -1
        if step_y == 0 and dy != 0:
            step_y = 1 if dy > 0 else -1
        move_mouse_relative(step_x, step_y)
        if per_step_sleep > 0:
            _t.sleep(per_step_sleep)


# # 示例：获取当前鼠标位置
# px, py = get_mouse_position()
# print(f"当前鼠标位置: ({px}, {py})")

# # 示例：向右移动10像素，向下移动10像素
# move_mouse_relative(10, 10)

# x, y = get_mouse_position()
# dx = x - px
# dy = y - py
# print(f"鼠标相对移动: dx={dx}, dy={dy}")
