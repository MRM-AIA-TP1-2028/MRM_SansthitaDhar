#!/usr/bin/env python3
"""Live blue-object tracking and tennis-ball detection with ArUco scale estimation.

Run:
    python blue_object_tennis_aruco.py --camera 0 --marker-size-mm 50

Place a printed ArUco marker of the configured size in the same flat plane as
the objects. Click a blue object to select it. Press q or Esc to quit and r to
clear tracks and selection.
"""

import argparse
from collections import deque
import math
import re

import cv2
import numpy as np


BLUE_LOWER = np.array([95, 75, 45], dtype=np.uint8)
BLUE_UPPER = np.array([140, 255, 255], dtype=np.uint8)
BALL_LOWER = np.array([20, 65, 65], dtype=np.uint8)
BALL_UPPER = np.array([42, 255, 255], dtype=np.uint8)

MIN_BLUE_AREA_PX = 350
MIN_BALL_AREA_PX = 250
MAX_MATCH_DISTANCE_PX = 100
TRAIL_LENGTH = 40
MAX_MISSED_FRAMES = 12


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def aruco_detectors(dictionary_ids):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build has no aruco module. "
            "Install opencv-contrib-python."
        )

    aruco = cv2.aruco
    detectors = []

    for dictionary_name, dictionary_id in dictionary_ids:
        dictionary = aruco.getPredefinedDictionary(dictionary_id)

        if hasattr(aruco, "DetectorParameters"):
            parameters = aruco.DetectorParameters()
        else:
            parameters = aruco.DetectorParameters_create()

        parameters.minMarkerPerimeterRate = 0.01
        parameters.maxMarkerPerimeterRate = 4.0

        if (
            hasattr(parameters, "cornerRefinementMethod")
            and hasattr(aruco, "CORNER_REFINE_SUBPIX")
        ):
            parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        if hasattr(aruco, "ArucoDetector"):
            detector = aruco.ArucoDetector(dictionary, parameters)
            detect = detector.detectMarkers
        else:
            detect = lambda gray, d=dictionary, p=parameters: aruco.detectMarkers(
                gray, d, parameters=p
            )

        detectors.append((dictionary_name, detect))

    return detectors


def get_plane_transform(markers, marker_size_mm, marker_id=None):
    """Return an image-pixel to marker-plane-mm homography and marker ID."""
    candidates = []

    for dictionary_name, corners, ids in markers:
        if ids is None:
            continue

        for index, value in enumerate(ids.flatten()):
            if marker_id is None or int(value) == marker_id:
                marker_corners = corners[index][0].astype(np.float32)
                area = cv2.contourArea(marker_corners)
                candidates.append(
                    (area, dictionary_name, marker_corners, int(value))
                )

    if not candidates:
        return None, None

    _, dictionary_name, image_points, chosen_id = max(
        candidates,
        key=lambda candidate: candidate[0],
    )

    mm_points = np.array(
        [
            [0, 0],
            [marker_size_mm, 0],
            [marker_size_mm, marker_size_mm],
            [0, marker_size_mm],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(image_points, mm_points)
    return homography, (dictionary_name, chosen_id)


def map_point(homography, point):
    if homography is None:
        return None

    source = np.array(
        [[[float(point[0]), float(point[1])]]],
        dtype=np.float32,
    )
    result = cv2.perspectiveTransform(source, homography)[0, 0]
    return float(result[0]), float(result[1])


def area_cm2(contour, homography):
    if homography is None:
        return None

    mapped_contour = cv2.perspectiveTransform(
        contour.astype(np.float32),
        homography,
    )
    return abs(float(cv2.contourArea(mapped_contour))) / 100.0


def detect_ball(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_ball = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BALL_AREA_PX:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.55:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        score = area * circularity

        if score > best_score:
            best_score = score
            best_ball = (
                contour,
                (int(x), int(y)),
                int(radius),
            )

    return best_ball


def match_tracks(objects, tracks, missed, next_id):
    """Match detections to nearby tracks and retain tracks briefly when missed."""
    available_tracks = set(tracks)

    for obj in objects:
        center = obj["center"]
        matched_id = None
        closest_distance = MAX_MATCH_DISTANCE_PX

        for track_id in available_tracks:
            previous_center = tracks[track_id][-1]
            distance = math.hypot(
                center[0] - previous_center[0],
                center[1] - previous_center[1],
            )

            if distance < closest_distance:
                matched_id = track_id
                closest_distance = distance

        if matched_id is None:
            matched_id = next_id
            next_id += 1
            tracks[matched_id] = deque(maxlen=TRAIL_LENGTH)
            missed[matched_id] = 0
        else:
            available_tracks.remove(matched_id)
            missed[matched_id] = 0

        tracks[matched_id].append(center)
        obj["track_id"] = matched_id

    for track_id in available_tracks:
        missed[track_id] += 1

        if missed[track_id] > MAX_MISSED_FRAMES:
            tracks.pop(track_id, None)
            missed.pop(track_id, None)

    return next_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        default=50.0,
        help="Printed marker side length in millimeters (default: 50)",
    )
    parser.add_argument(
        "--dictionary",
        default="AUTO",
        help="Dictionary name or AUTO to scan the supported families",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=None,
        help="Use only this marker ID; default uses the largest detected marker",
    )
    args = parser.parse_args()

    if args.marker_size_mm <= 0:
        parser.error("--marker-size-mm must be positive")

    family_pattern = re.compile(r"^DICT_[4567]X[4567]_\d+$")
    dictionary_names = sorted(
        name
        for name in dir(cv2.aruco)
        if family_pattern.match(name) or name == "DICT_ARUCO_ORIGINAL"
    )

    if args.dictionary.upper() == "AUTO":
        dictionaries = [
            (name, getattr(cv2.aruco, name))
            for name in dictionary_names
        ]
    elif args.dictionary in dictionary_names:
        dictionaries = [
            (args.dictionary, getattr(cv2.aruco, args.dictionary))
        ]
    else:
        parser.error(
            f"Unknown dictionary {args.dictionary!r}. "
            f"Available dictionaries: {', '.join(dictionary_names)}"
        )

    detectors = aruco_detectors(dictionaries)
    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. "
            "Check the camera index and permissions."
        )

    tracks = {}
    missed = {}
    next_id = 1
    selected_id = None
    clickable_objects = []
    window_name = "Blue Object and Tennis Ball Tracker"

    def on_mouse(event, x, y, flags, userdata):
        nonlocal selected_id

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for obj in reversed(clickable_objects):
            bx, by, bw, bh = obj["bbox"]

            if bx <= x <= bx + bw and by <= y <= by + bh:
                track_id = obj["track_id"]
                selected_id = None if selected_id == track_id else track_id
                return

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read a frame from the camera.")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue_mask = clean_mask(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))
            ball_mask = clean_mask(cv2.inRange(hsv, BALL_LOWER, BALL_UPPER))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_markers = []
            visible_marker_labels = []

            for dictionary_name, detect in detectors:
                marker_corners, marker_ids, _ = detect(gray)
                detected_markers.append(
                    (dictionary_name, marker_corners, marker_ids)
                )

                if marker_ids is not None:
                    cv2.aruco.drawDetectedMarkers(
                        frame,
                        marker_corners,
                        marker_ids,
                    )
                    visible_marker_labels.extend(
                        f"{dictionary_name}:{int(marker_id)}"
                        for marker_id in marker_ids.flatten()
                    )

            homography, found_marker = get_plane_transform(
                detected_markers,
                args.marker_size_mm,
                args.marker_id,
            )

            blue_contours, _ = cv2.findContours(
                blue_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            objects = []

            for contour in blue_contours:
                pixel_area = cv2.contourArea(contour)
                if pixel_area < MIN_BLUE_AREA_PX:
                    continue

                x, y, width, height = cv2.boundingRect(contour)
                objects.append(
                    {
                        "contour": contour,
                        "bbox": (x, y, width, height),
                        "center": (x + width // 2, y + height // 2),
                    }
                )

            next_id = match_tracks(objects, tracks, missed, next_id)
            clickable_objects[:] = objects

            ball = detect_ball(ball_mask)
            ball_center = None

            if ball is not None:
                ball_contour, ball_center, ball_radius = ball
                ball_area = area_cm2(ball_contour, homography)

                cv2.drawContours(
                    frame,
                    [ball_contour],
                    -1,
                    (0, 220, 255),
                    2,
                )
                cv2.circle(
                    frame,
                    ball_center,
                    ball_radius,
                    (0, 220, 255),
                    2,
                )
                cv2.circle(frame, ball_center, 4, (0, 0, 255), -1)

                area_label = (
                    f"{ball_area:.2f} cm^2"
                    if ball_area is not None
                    else "scale unavailable"
                )
                cv2.putText(
                    frame,
                    f"Ball center {ball_center} | "
                    f"radius {ball_radius}px | {area_label}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (0, 220, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Tennis ball: not detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 255),
                    2,
                )

            for obj in objects:
                contour = obj["contour"]
                x, y, width, height = obj["bbox"]
                center = obj["center"]
                track_id = obj["track_id"]
                selected = track_id == selected_id
                color = (255, 0, 255) if selected else (255, 120, 0)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    color,
                    2,
                )
                cv2.drawContours(frame, [contour], -1, color, 1)
                cv2.circle(frame, center, 4, color, -1)

                trail = list(tracks[track_id])
                for index in range(1, len(trail)):
                    cv2.line(
                        frame,
                        trail[index - 1],
                        trail[index],
                        color,
                        2,
                    )

                object_area = area_cm2(contour, homography)
                area_text = (
                    f"{object_area:.2f} cm^2"
                    if object_area is not None
                    else "scale unavailable"
                )
                selected_text = "SELECTED " if selected else ""

                cv2.putText(
                    frame,
                    f"{selected_text}Blue #{track_id}: {area_text}",
                    (x, max(48, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"bbox=({x},{y},{width},{height}) center={center}",
                    (x, min(frame.shape[0] - 8, y + height + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    color,
                    1,
                )

                if ball_center is not None:
                    cv2.line(frame, center, ball_center, (0, 255, 0), 1)
                    blue_mm = map_point(homography, center)
                    ball_mm = map_point(homography, ball_center)

                    if blue_mm is None:
                        relation = (
                            f"from ball dx={center[0] - ball_center[0]}px "
                            f"dy={center[1] - ball_center[1]}px"
                        )
                    else:
                        dx_cm = (blue_mm[0] - ball_mm[0]) / 10.0
                        dy_cm = (blue_mm[1] - ball_mm[1]) / 10.0
                        relation = (
                            f"from ball dx={dx_cm:.1f}cm "
                            f"dy={dy_cm:.1f}cm"
                        )

                    cv2.putText(
                        frame,
                        relation,
                        (x, min(frame.shape[0] - 8, y + height + 36)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )

            if homography is None:
                if visible_marker_labels and args.marker_id is not None:
                    visible = ", ".join(dict.fromkeys(visible_marker_labels))
                    status = (
                        f"Markers seen ({visible}); "
                        f"none has ID {args.marker_id}"
                    )
                elif visible_marker_labels:
                    visible = ", ".join(dict.fromkeys(visible_marker_labels))
                    status = f"Markers seen ({visible}); scale unavailable"
                else:
                    status = (
                        "No ArUco marker detected: check light, focus, "
                        "size, border, dictionary"
                    )
            else:
                found_dictionary, found_marker_id = found_marker
                status = (
                    f"Scale active: {found_dictionary} ID {found_marker_id}, "
                    f"side {args.marker_size_mm:g} mm"
                )

            status_color = (
                (0, 220, 0) if homography is not None else (0, 0, 255)
            )
            cv2.putText(
                frame,
                status,
                (10, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("r"):
                tracks.clear()
                missed.clear()
                selected_id = None
                next_id = 1

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
