import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera. Check macOS camera permissions."
        )

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        cv2.imshow("Industrial Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
