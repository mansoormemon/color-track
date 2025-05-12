import cv2 as cv
import numpy as np

# HSV Color Range for Red Color Detection

# The HSV color space is often used for color-based object detection because it separates color information (Hue)
# from brightness (Value) and saturation (Saturation), making it easier to define color ranges.
# Here we are detecting the color red, which can be tricky because red spans the beginning and end of the Hue (H) spectrum.

# The red color in the HSV space is represented with two main hue ranges:
# - One near the beginning of the Hue spectrum (0 to 10 degrees on the Hue scale).
# - Another near the end of the Hue spectrum (170 to 180 degrees on the Hue scale).

# The HSV color space is represented as:
# Hue (H) = 0-180 (OpenCV ranges Hue to [0, 180] instead of [0, 360]).
# Saturation (S) = 0-255 (0 means no color, 255 means full color intensity).
# Value (V) = 0-255 (0 means black, 255 means brightest).

# Lower and Upper Red Color Range for the First Range (0 to 10 degrees in Hue)
# - Lower Red 1 (0, 64, 32): The lower boundary for red (Hue=0, S=64, V=32).
#     - Hue=0 is at the start of the red spectrum.
#     - Saturation of 64 ensures the color is not desaturated (i.e., no pastel or washed-out red).
#     - Value of 32 means it avoids very dark reds (closer to black).
# - Upper Red 1 (10, 255, 223): The upper boundary for red (Hue=10, S=255, V=223).
#     - Hue=10 still falls in the red range but a bit towards orange.
#     - Saturation of 255 ensures full color intensity (vivid red).
#     - Value of 223 ensures we exclude too light or washed-out reds (close to white).

LOWER_RED_1 = np.array([0, 64, 16])
UPPER_RED_1 = np.array([10, 255, 255])

# Lower and Upper Red Color Range for the Second Range (170 to 180 degrees in Hue)
# - Lower Red 2 (170, 64, 32): The lower boundary for red in the second range (Hue=170, S=64, V=32).
#     - Hue=170 is near the end of the red spectrum (near the 180-degree mark).
#     - Saturation of 64 ensures moderate color intensity (not too washed out).
#     - Value of 32 ensures dark reds are excluded (avoiding blackness).
# - Upper Red 2 (180, 255, 223): The upper boundary for red in the second range (Hue=180, S=255, V=223).
#     - Hue=180 marks the very end of the red spectrum (a more pure red).
#     - Saturation of 255 ensures full color intensity (vivid red).
#     - Value of 223 ensures we exclude overly bright or light reds (close to white).

LOWER_RED_2 = np.array([170, 64, 16])
UPPER_RED_2 = np.array([180, 255, 255])

MIN_AREA = 324

cap = cv.VideoCapture("samples/9.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask_1 = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask_2 = cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    red_mask = cv.bitwise_or(mask_1, mask_2)

    kernel = np.ones((13, 13), np.uint8)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    filtered_countours = [
        contour for contour in contours if cv.contourArea(contour) > MIN_AREA
    ]

    for contour in filtered_countours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

    cv.imshow("Red Mask", red_mask)
    cv.imshow("Stream", frame)

    if cv.waitKey(16) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
