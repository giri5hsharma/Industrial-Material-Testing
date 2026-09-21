import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import Resize

from anomalib.models import EfficientAd


# ============================================================
# CONFIGURATION
# ============================================================

# EfficientAD input size
IMAGE_SIZE = (256, 256)

# ------------------------------------------------------------
# Inspection ROI
#
# Change these values to position the green box around
# the object you want to inspect.
# ------------------------------------------------------------

ROI_X = 170
ROI_Y = 70
ROI_W = 300
ROI_H = 300

# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------

CALIBRATION_FRAMES = 30

# Threshold = mean + CALIBRATION_STD_MULTIPLIER * std
CALIBRATION_STD_MULTIPLIER = 3.0

# Never allow threshold to become too low
MIN_THRESHOLD = 0.45

# ------------------------------------------------------------
# Temporal filtering
# ------------------------------------------------------------

HISTORY_SIZE = 7
DEFECT_REQUIRED = 5

# ============================================================
# MODEL
# ============================================================


def load_model(checkpoint_path: str, device: torch.device):

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")

    model = EfficientAd.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    model.to(device)
    model.eval()

    resize = Resize(
        IMAGE_SIZE,
        antialias=True,
    )

    return model, resize


# ============================================================
# PREPROCESSING
# ============================================================


def preprocess(frame, resize, device):

    # OpenCV uses BGR
    # EfficientAD expects RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    # HWC -> CHW
    tensor = torch.from_numpy(
        rgb
    ).permute(2, 0, 1)

    # uint8 -> float32 [0, 1]
    tensor = tensor.float() / 255.0

    # Resize to EfficientAD input size
    tensor = resize(tensor)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


# ============================================================
# INFERENCE
# ============================================================


@torch.inference_mode()
def predict(
    model,
    tensor,
):

    # EfficientAD forward pass
    prediction = model.model(tensor)

    # Anomalib post-processing
    prediction = model.post_processor(
        prediction
    )

    score = float(
        prediction.pred_score[0]
        .detach()
        .cpu()
    )

    # This is the anomaly map after
    # Anomalib post-processing.
    anomaly_map = (
        prediction.anomaly_map[0]
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    return score, anomaly_map


# ============================================================
# ROI
# ============================================================


def get_roi(frame):

    return frame[
        ROI_Y : ROI_Y + ROI_H,
        ROI_X : ROI_X + ROI_W,
    ]


# ============================================================
# HEATMAP
# ============================================================


def make_heatmap(
    anomaly_map,
    roi_shape,
):

    # Keep values in [0, 1]
    anomaly_map = np.clip(
        anomaly_map,
        0.0,
        1.0,
    )

    # Convert to 8-bit
    heatmap = (
        anomaly_map * 255
    ).astype(np.uint8)

    # Resize to ROI dimensions
    heatmap = cv2.resize(
        heatmap,
        (
            roi_shape[1],
            roi_shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # OpenCV color map
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    return heatmap_color


# ============================================================
# CALIBRATION
# ============================================================


def calibrate_threshold(
    model,
    resize,
    device,
    camera,
):

    print()
    print("=" * 55)
    print("STARTING CALIBRATION")
    print("=" * 55)
    print()
    print("Place a GOOD / DEFECT-FREE object inside")
    print("the green inspection box.")
    print()
    print("Keep the object and camera as stable as possible.")
    print()

    scores = []

    for i in range(
        CALIBRATION_FRAMES
    ):

        success, frame = camera.read()

        if not success:
            print(
                "Failed to read frame during calibration."
            )
            continue

        # Get only the inspection area
        roi = get_roi(frame)

        # Run inference
        tensor = preprocess(
            roi,
            resize,
            device,
        )

        score, _ = predict(
            model,
            tensor,
        )

        scores.append(score)

        # ------------------------------------------------
        # Calibration display
        # ------------------------------------------------

        display = frame.copy()

        cv2.rectangle(
            display,
            (ROI_X, ROI_Y),
            (
                ROI_X + ROI_W,
                ROI_Y + ROI_H,
            ),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"CALIBRATING {i + 1}/{CALIBRATION_FRAMES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"Current score: {score:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Industrial Defect Detection",
            display,
        )

        # Needed so OpenCV continues updating
        cv2.waitKey(1)

    # ----------------------------------------------------
    # Validate calibration data
    # ----------------------------------------------------

    if len(scores) < 10:

        raise RuntimeError(
            "Not enough calibration samples."
        )

    scores = np.array(
        scores,
        dtype=np.float32,
    )

    mean_score = float(
        np.mean(scores)
    )

    std_score = float(
        np.std(scores)
    )

    threshold = (
        mean_score
        + CALIBRATION_STD_MULTIPLIER
        * std_score
    )

    threshold = max(
        threshold,
        MIN_THRESHOLD,
    )

    print()
    print("=" * 55)
    print("CALIBRATION COMPLETE")
    print("=" * 55)
    print()
    print(
        f"Normal mean score : {mean_score:.4f}"
    )
    print(
        f"Standard deviation: {std_score:.4f}"
    )
    print(
        f"Detection threshold: {threshold:.4f}"
    )
    print()

    return threshold


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to EfficientAD checkpoint",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index",
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=2,
        help="Run inference every N frames",
    )

    parser.add_argument(
        "--start-on",
        action="store_true",
        help="Start with detection enabled",
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    if torch.backends.mps.is_available():

        device = torch.device("mps")

    elif torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, resize = load_model(
        args.checkpoint,
        device,
    )

    print()
    print("Model loaded successfully.")
    print()

    # ========================================================
    # CAMERA
    # ========================================================

    camera = cv2.VideoCapture(
        args.camera
    )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open camera."
        )

    # Lower camera resolution for better FPS
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    # ========================================================
    # STATE
    # ========================================================

    detection_enabled = args.start_on

    # Custom threshold obtained from calibration
    custom_threshold = None

    # Recent anomaly decisions
    score_history = []

    # Last prediction
    last_score = 0.0

    last_label = 0

    last_heatmap = None

    # Frame counter
    frame_count = 0

    # FPS
    fps = 0.0

    fps_counter = 0

    fps_start = time.perf_counter()

    # ========================================================
    # CONTROLS
    # ========================================================

    print("=" * 55)
    print("CONTROLS")
    print("=" * 55)
    print()
    print("D  -> Toggle detection ON/OFF")
    print("C  -> Calibrate using a GOOD object")
    print("S  -> Save current frame")
    print("Q  -> Quit")
    print()

    print(
        "Detection initially:",
        "ON" if detection_enabled else "OFF",
    )

    print()

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Failed to read camera frame."
            )

            break

        frame_count += 1

        # ====================================================
        # DETECTION
        # ====================================================

        if detection_enabled:

            # Only run inference every N frames
            if (
                frame_count % args.skip
                == 0
            ):

                # --------------------------------------------
                # Extract ROI
                # --------------------------------------------

                roi = get_roi(
                    frame
                )

                # --------------------------------------------
                # Preprocess
                # --------------------------------------------

                tensor = preprocess(
                    roi,
                    resize,
                    device,
                )

                # --------------------------------------------
                # Model inference
                # --------------------------------------------

                (
                    score,
                    anomaly_map,
                ) = predict(
                    model,
                    tensor,
                )

                last_score = score

                # --------------------------------------------
                # Generate heatmap
                # --------------------------------------------

                last_heatmap = make_heatmap(
                    anomaly_map,
                    roi.shape,
                )

                # --------------------------------------------
                # Custom calibrated threshold
                # --------------------------------------------

                if (
                    custom_threshold
                    is not None
                ):

                    frame_is_defect = (
                        score
                        >= custom_threshold
                    )

                    score_history.append(
                        frame_is_defect
                    )

                    # Keep only most recent
                    # HISTORY_SIZE results
                    if (
                        len(score_history)
                        > HISTORY_SIZE
                    ):

                        score_history.pop(
                            0
                        )

                    defect_votes = sum(
                        score_history
                    )

                    # Require multiple anomalous
                    # frames before declaring defect
                    if (
                        defect_votes
                        >= DEFECT_REQUIRED
                    ):

                        last_label = 1

                    else:

                        last_label = 0

                else:

                    # We don't trust the default
                    # MVTec threshold for webcam
                    last_label = 0

        # ====================================================
        # DISPLAY
        # ====================================================

        display = frame.copy()

        # ----------------------------------------------------
        # ROI BOX
        # ----------------------------------------------------

        if detection_enabled:

            roi_color = (
                (0, 255, 0)
                if last_label == 0
                else (0, 0, 255)
            )

        else:

            roi_color = (
                0,
                200,
                255,
            )

        cv2.rectangle(
            display,
            (ROI_X, ROI_Y),
            (
                ROI_X + ROI_W,
                ROI_Y + ROI_H,
            ),
            roi_color,
            2,
        )

        # ====================================================
        # HEATMAP
        # ====================================================

        if (
            detection_enabled
            and last_heatmap is not None
        ):

            # Only blend heatmap INSIDE ROI
            roi_display = display[
                ROI_Y : ROI_Y + ROI_H,
                ROI_X : ROI_X + ROI_W,
            ]

            blended_roi = cv2.addWeighted(
                roi_display,
                0.70,
                last_heatmap,
                0.30,
                0,
            )

            display[
                ROI_Y : ROI_Y + ROI_H,
                ROI_X : ROI_X + ROI_W,
            ] = blended_roi

        # ====================================================
        # STATUS PANEL
        # ====================================================

        cv2.rectangle(
            display,
            (10, 10),
            (465, 165),
            (20, 20, 20),
            -1,
        )

        # ----------------------------------------------------
        # DETECTION OFF
        # ----------------------------------------------------

        if not detection_enabled:

            cv2.putText(
                display,
                "DETECTION: OFF",
                (25, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 200, 255),
                2,
            )

            cv2.putText(
                display,
                "Press D to enable",
                (25, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                "Press C to calibrate",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # ----------------------------------------------------
        # DETECTION ON
        # ----------------------------------------------------

        else:

            # -----------------------------------------------
            # Status
            # -----------------------------------------------

            if (
                custom_threshold
                is None
            ):

                status = "NOT CALIBRATED"

                status_color = (
                    0,
                    200,
                    255,
                )

            elif last_label == 1:

                status = "DEFECT"

                status_color = (
                    0,
                    0,
                    255,
                )

            else:

                status = "NORMAL"

                status_color = (
                    0,
                    255,
                    0,
                )

            cv2.putText(
                display,
                f"STATUS: {status}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

            # -----------------------------------------------
            # Score
            # -----------------------------------------------

            cv2.putText(
                display,
                f"SCORE: {last_score:.3f}",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # -----------------------------------------------
            # Threshold
            # -----------------------------------------------

            if (
                custom_threshold
                is not None
            ):

                cv2.putText(
                    display,
                    f"THRESHOLD: {custom_threshold:.3f}",
                    (25, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

                # -------------------------------------------
                # Temporal votes
                # -------------------------------------------

                votes = sum(
                    score_history
                )

                cv2.putText(
                    display,
                    (
                        f"DEFECT VOTES: "
                        f"{votes}/{DEFECT_REQUIRED}"
                    ),
                    (25, 137),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

            else:

                cv2.putText(
                    display,
                    "Press C with GOOD object",
                    (25, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    1,
                )

        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        elapsed = (
            time.perf_counter()
            - fps_start
        )

        if elapsed >= 1.0:

            fps = (
                fps_counter
                / elapsed
            )

            fps_counter = 0

            fps_start = (
                time.perf_counter()
            )

        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (500, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "Industrial Defect Detection",
            display,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # ====================================================
        # KEYBOARD CONTROLS
        # ====================================================

        # ----------------------------------------------------
        # D -> TOGGLE DETECTION
        # ----------------------------------------------------

        if key == ord("d"):

            detection_enabled = (
                not detection_enabled
            )

            if detection_enabled:

                print(
                    "Detection: ON"
                )

            else:

                print(
                    "Detection: OFF"
                )

                last_score = 0.0

                last_label = 0

                last_heatmap = None

                score_history.clear()

        # ----------------------------------------------------
        # C -> CALIBRATION
        # ----------------------------------------------------

        elif key == ord("c"):

            try:

                # Calibration requires detection
                detection_enabled = True

                custom_threshold = (
                    calibrate_threshold(
                        model,
                        resize,
                        device,
                        camera,
                    )
                )

                # Clear old temporal history
                score_history.clear()

                # Clear old heatmap
                last_heatmap = None

                print(
                    "Detection calibration saved."
                )

            except Exception as e:

                print()
                print(
                    f"Calibration failed: {e}"
                )
                print()

        # ----------------------------------------------------
        # S -> SAVE FRAME
        # ----------------------------------------------------

        elif key == ord("s"):

            output_dir = Path(
                "captured"
            )

            output_dir.mkdir(
                exist_ok=True
            )

            timestamp = int(
                time.time()
            )

            filename = (
                output_dir
                / f"realtime_{timestamp}.jpg"
            )

            cv2.imwrite(
                str(filename),
                display,
            )

            print(
                f"Saved: {filename}"
            )

        # ----------------------------------------------------
        # Q -> QUIT
        # ----------------------------------------------------

        elif key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()
