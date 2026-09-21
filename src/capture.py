import cv2
from pathlib import Path


OUTPUT_DIR = Path("captured")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press C to capture.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        cv2.imshow("Industrial Inspection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            output_path = OUTPUT_DIR / "inspection.jpg"
            cv2.imwrite(str(output_path), frame)

            print(f"Captured: {output_path}")

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
