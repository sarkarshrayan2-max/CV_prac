from pathlib import Path
from typing import Set

import cv2
from ultralytics import YOLO


VIDEO_PATH = Path("videos/people_walking.mp4")
OUTPUT_PATH = Path("outputs/tracking_details.mp4")


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH.resolve()}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo26n.pt")

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the output video.")

    unique_track_ids: Set[int] = set()
    frame_number = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        results = model.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=0.35,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        result = results[0]
        annotated_frame = result.plot()

        active_people = 0

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            active_people = len(track_ids)
            unique_track_ids.update(track_ids)

            for box, track_id, confidence in zip(
                boxes,
                track_ids,
                confidences,
            ):
                x1, y1, x2, y2 = map(int, box)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1,
                )

                print(
                    {
                        "frame": frame_number,
                        "track_id": track_id,
                        "confidence": round(confidence, 3),
                        "box": [x1, y1, x2, y2],
                        "center": [center_x, center_y],
                    }
                )

        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (360, 115),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Active people: {active_people}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Unique track IDs: {len(unique_track_ids)}",
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        writer.write(annotated_frame)
        cv2.imshow("Tracking Details", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nTracking completed.")
    print(f"Unique track IDs: {len(unique_track_ids)}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()