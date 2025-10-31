import yaml
import torch
from capture.grabber import grab
from preprocess.transform import preprocess
from model.yolo_wrapper import YOLOModel
from logic.targeting import select_target, get_mouse_position
from logic.predictor import TargetPredictor
from control.mouse import move_mouse_to_smooth
from control.mouse import move_mouse_to_duration
from utils.logger import setup_logger
import time


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    # log = setup_logger(log_file=cfg["log_file"])
    model = YOLOModel(cfg["model_path"])
    predictor = TargetPredictor(
        pos_alpha=cfg.get("smoothing_alpha", 0.5),
        vel_alpha=cfg.get("velocity_alpha", 0.6),
        max_speed_px_per_s=cfg.get("max_target_speed", 3000.0),
        lead_ms=cfg.get("prediction_ms", 60),
    )

    while True:
        frame = grab(region=cfg["screen_region"])
        img_tensor = preprocess(frame)
        detections = model.infer(img_tensor, conf = cfg["confidence_threshold"])
        mouse_position = get_mouse_position()
        target = select_target(detections, cfg["preferred_cls"], cfg["screen_region"], cfg["confidence_threshold"], mouse_xy=mouse_position)
        if target:
            if cfg.get("prediction_enabled", True):
                pred_x, pred_y = predictor.update_and_predict(*target)
            else:
                pred_x, pred_y = target
            if cfg.get("direct_move_enabled", False):
                move_mouse_to_duration(
                    pred_x,
                    pred_y,
                    duration_s=float(cfg.get("move_duration", 0.1)),
                    steps=int(cfg.get("move_steps", 10)),
                )
            else:
                for _ in range(int(cfg.get("repeat_moves", 1))):
                    move_mouse_to_smooth(
                        pred_x,
                        pred_y,
                        gain=cfg.get("mouse_gain", 0.35),
                        max_step=cfg.get("mouse_max_step", 25),
                        deadzone=cfg.get("mouse_deadzone", 1),
                    )
            time.sleep(cfg.get("loop_sleep", 0.01))
        # time.sleep(0.05)


if __name__ == "__main__":
    main()
