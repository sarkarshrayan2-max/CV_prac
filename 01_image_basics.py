import cv2
import matplotlib.pyplot as plt

image_path = "images/zidane.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read {image_path}")

print("Image shape:", image.shape)
print("Height:", image.shape[0])
print("Width:", image.shape[1])
print("Channels:", image.shape[2])

resized = cv2.resize(image, (640, 480))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


cv2.rectangle(
    resized,
    (100, 100),
    (350, 350),
    (0, 255, 0),
    3,
)

cv2.putText(
    resized,
    "Practice Object",
    (100, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)

cv2.imwrite("outputs/resized.jpg", resized)
cv2.imwrite("outputs/grayscale.jpg", gray)

cv2.imshow("Original", image)
cv2.imshow("Processed", resized)
cv2.imshow("Grayscale", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
processed_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(processed_rgb)
plt.title("Processed Image")
plt.axis("off")
plt.show()