import time
from typing import Optional, Tuple


class TargetPredictor:
    """Simple constant-velocity predictor with EMA smoothing on position and velocity.

    - Call update_and_predict with the latest raw target (x, y) to get a smoothed, predicted point.
    - Uses exponential smoothing for stability and clamps velocity to avoid spikes.
    """

    def __init__(self, pos_alpha: float = 0.5, vel_alpha: float = 0.5, max_speed_px_per_s: float = 3000.0, lead_ms: int = 60):
        self.pos_alpha = pos_alpha
        self.vel_alpha = vel_alpha
        self.max_speed = max_speed_px_per_s
        self.lead_ms = lead_ms
        self._pos_ema: Optional[Tuple[float, float]] = None
        self._vel_ema: Optional[Tuple[float, float]] = None
        self._last_ts: Optional[float] = None

    def _clamp_speed(self, vx: float, vy: float) -> Tuple[float, float]:
        speed = (vx * vx + vy * vy) ** 0.5
        if speed <= self.max_speed or speed == 0:
            return vx, vy
        scale = self.max_speed / speed
        return vx * scale, vy * scale

    def update_and_predict(self, x: int, y: int, now_s: Optional[float] = None) -> Tuple[int, int]:
        if now_s is None:
            now_s = time.time()

        # Initialize on first call
        if self._pos_ema is None:
            self._pos_ema = (float(x), float(y))
            self._vel_ema = (0.0, 0.0)
            self._last_ts = now_s
            return int(x), int(y)

        dt = max(1e-3, now_s - (self._last_ts or now_s))
        self._last_ts = now_s

        # Position EMA
        px, py = self._pos_ema
        px = self.pos_alpha * float(x) + (1.0 - self.pos_alpha) * px
        py = self.pos_alpha * float(y) + (1.0 - self.pos_alpha) * py
        self._pos_ema = (px, py)

        # Instantaneous velocity from raw delta
        inst_vx = (float(x) - px) / dt
        inst_vy = (float(y) - py) / dt

        # Velocity EMA with clamp
        vx, vy = self._vel_ema or (0.0, 0.0)
        vx = self.vel_alpha * inst_vx + (1.0 - self.vel_alpha) * vx
        vy = self.vel_alpha * inst_vy + (1.0 - self.vel_alpha) * vy
        vx, vy = self._clamp_speed(vx, vy)
        self._vel_ema = (vx, vy)

        # Predict forward by lead_ms
        lead_s = max(0.0, self.lead_ms) / 1000.0
        pred_x = px + vx * lead_s
        pred_y = py + vy * lead_s

        return int(pred_x), int(pred_y)


