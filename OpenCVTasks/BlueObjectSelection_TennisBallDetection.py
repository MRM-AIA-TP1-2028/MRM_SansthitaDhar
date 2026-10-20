_"""Select and track a blue object while detecting a tennis ball from a webcam.

Click inside a blue object to select it. Press r to reset selection; q or Esc quits.
Install dependencies with: python -m pip install opencv-python numpy
"""

import math
import cv2
import numpy as np

# HSV ranges are starting values; tune these for the camera and lighting.
BLUE_LOWER = np.array([90, 80, 50], dtype=np.uint8)
BLUE_UPPER = np.array([135, 255, 255], dtype=np.uint8)
BALL_LOWER = np.array([20, 80, 80], dtype=np.uint8)
BALL_UPPER = np.array([45, 255, 255], dtype=np.uint8)

MIN_BLUE_AREA = 300
MIN_BALL_AREA = 150
MAX_TRACK_DISTANCE = 150
TRAIL_LENGTH = 100
WINDOW = "Blue Object and Tennis Ball"

selected_center = None
previous_center = None
tracking_points = []
current_blue_contours = []


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_contours(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def center_of(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )


def mouse_callback(event, x, y, flags, param):
    """Select a blue contour by clicking anywhere inside it."""
    global selected_center, previous_center, tracking_points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for contour in current_blue_contours:
        if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
            center = center_of(contour)
            if center is not None:
                selected_center = center
                previous_center = center
                tracking_points = [center]
            return


def find_tennis_ball(mask):
    """Return the largest sufficiently round tennis-ball color contour."""
    candidates = []
    for contour in find_contours(mask):
        area = cv2.contourArea(contour)
        if area < MIN_BALL_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.55:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius >= 5:
            candidates.append((area, contour, (int(x), int(y)), int(radius)))

    return max(candidates, key=lambda item: item[0]) if candidates else None


def main():
    global current_blue_contours, selected_center, previous_center, tracking_points

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Try changing camera index 0.")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue_mask = clean_mask(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))
            ball_mask = clean_mask(cv2.inRange(hsv, BALL_LOWER, BALL_UPPER))

            current_blue_contours = [
                contour for contour in find_contours(blue_mask)
                if cv2.contourArea(contour) >= MIN_BLUE_AREA
            ]

            # Draw unselected blue objects so the user can click one.
            if selected_center is None:
                for contour in current_blue_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 150, 0), 1)
                cv2.putText(frame, "Click a blue object to select it", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            else:
                # Follow the nearest blue contour center; works for any contour shape.
                candidates = []
                for contour in current_blue_contours:
                    center = center_of(contour)
                    if center is not None:
                        distance = math.hypot(
                            center[0] - previous_center[0],
                            center[1] - previous_center[1],
                        )
                        candidates.append((distance, contour, center))

                if candidates:
                    distance, contour, center = min(candidates, key=lambda item: item[0])
                    if distance <= MAX_TRACK_DISTANCE:
                        previous_center = center
                        selected_center = center
                        tracking_points.append(center)
                        tracking_points = tracking_points[-TRAIL_LENGTH:]

                        x, y, w, h = cv2.boundingRect(contour)
                        area = cv2.contourArea(contour)
                        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.circle(frame, center, 4, (255, 255, 255), -1)
                        cv2.putText(frame, f"Blue object area: {area:.0f} px^2",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 100, 0), 2)
                        cv2.putText(frame, f"Bounding box: x={x}, y={y}, w={w}, h={h}",
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 100, 0), 2)

                        # Center trajectory line.
                        for i in range(1, len(tracking_points)):
                            cv2.line(frame, tracking_points[i - 1], tracking_points[i],
                                     (255, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Selected blue object not found", (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Selected blue object not found", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Tennis ball is detected separately using its own color mask.
            ball = find_tennis_ball(ball_mask)
            ball_center = None
            if ball is not None:
                ball_area, ball_contour, ball_center, radius = ball
                cv2.drawContours(frame, [ball_contour], -1, (0, 220, 255), 2)
                cv2.circle(frame, ball_center, radius, (0, 220, 255), 2)
                cv2.circle(frame, ball_center, 4, (0, 0, 255), -1)
                cv2.putText(frame, f"Tennis ball center: {ball_center}", (10, frame.shape[0] - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                cv2.putText(frame, f"Radius: {radius}px  Area: {ball_area:.0f} px^2",
                            (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 220, 255), 2)

            # Relative position of the blue object with respect to the ball.
            if selected_center is not None and ball_center is not None:
                cv2.line(frame, selected_center, ball_center, (0, 255, 0), 2)
                dx = selected_center[0] - ball_center[0]
                dy = selected_center[1] - ball_center[1]
                cv2.putText(frame, f"Blue relative to ball: dx={dx}px, dy={dy}px",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 2)

            # Show only the color camera feed, with annotations.
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                selected_center = None
                previous_center = None
                tracking_points = []
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
