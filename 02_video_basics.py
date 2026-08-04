import cv2

video_path = "videos/people_walking.mp4"
capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    raise RuntimeError(f"Could not open {video_path}")

fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Frames:", frame_count)

frame_number = 0

while True:
    success, frame = capture.read()

    if not success:
        break

    frame_number += 1

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Video Practice", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()