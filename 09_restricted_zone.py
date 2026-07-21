from pathlib import Path
from typing import Set

import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_PATH = Path("videos/people_walking.mp4")
OUTPUT_PATH = Path("outputs/restricted_zone.mp4")


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
        raise RuntimeError("Could not create output video.")

    # Create a rectangular polygon using relative positions.
    restricted_zone = np.array(
        [
            [int(width * 0.30), int(height * 0.30)],
            [int(width * 0.75), int(height * 0.30)],
            [int(width * 0.85), int(height * 0.85)],
            [int(width * 0.20), int(height * 0.85)],
        ],
        dtype=np.int32,
    )

    # IDs that have already generated an entry alert.
    alerted_ids: Set[int] = set()

    while True:
        success, frame = capture.read()

        if not success:
            break

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

        # Draw the restricted zone.
        cv2.polylines(
            annotated_frame,
            [restricted_zone],
            isClosed=True,
            color=(0, 0, 255),
            thickness=3,
        )

        people_inside = 0

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Bottom-centre point.
                point_x = int((x1 + x2) / 2)
                point_y = y2

                polygon_result = cv2.pointPolygonTest(
                    restricted_zone,
                    (float(point_x), float(point_y)),
                    False,
                )

                person_inside = polygon_result >= 0

                if person_inside:
                    people_inside += 1

                    cv2.circle(
                        annotated_frame,
                        (point_x, point_y),
                        7,
                        (0, 0, 255),
                        -1,
                    )

                    cv2.putText(
                        annotated_frame,
                        "RESTRICTED ZONE",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )

                    if track_id not in alerted_ids:
                        alerted_ids.add(track_id)

                        print(
                            f"ALERT: Person ID {track_id} "
                            "entered the restricted zone."
                        )

                else:
                    cv2.circle(
                        annotated_frame,
                        (point_x, point_y),
                        6,
                        (0, 255, 0),
                        -1,
                    )

        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (400, 80),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated_frame,
            f"People inside restricted zone: {people_inside}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

        writer.write(annotated_frame)
        cv2.imshow("Restricted Zone Monitoring", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()