from pathlib import Path

from ultralytics import YOLO


VIDEO_PATH = Path("videos/people_walking.mp4")


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH.resolve()}"
        )

    # Load the pretrained YOLO model.
    model = YOLO("yolo26n.pt")

    # Track only people.
    # COCO class 0 represents a person.
    model.track(
        source=str(VIDEO_PATH),
        classes=[0],
        conf=0.5,
        tracker="bytetrack.yaml",
        show=True,
        save=True,
        project="outputs",
        name="simple_tracking",
    )


if __name__ == "__main__":
    main()