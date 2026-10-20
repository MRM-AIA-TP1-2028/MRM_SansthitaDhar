import cv2
import numpy as np

CAMERA_MATRIX = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1],
], dtype=np.float64)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)
MARKER_SIZE = 0.025  # Marker side length in meters

FAMILIES = [
    ("4x4", cv2.aruco.DICT_4X4_50),
    ("5x5", cv2.aruco.DICT_5X5_50),
    ("6x6", cv2.aruco.DICT_6X6_50),
]


def make_detector(dictionary_id):
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params)

    params = cv2.aruco.DetectorParameters_create()
    return dictionary, params


detectors = [
    (name, make_detector(dictionary_id))
    for name, dictionary_id in FAMILIES
]

half = MARKER_SIZE / 2
object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float32)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam. Try camera index 1.")

try:
    while True:
        ok, frame = camera.read()

        if not ok:
            print("Could not read a frame from the webcam.")
            break

        for family, detector in detectors:
            if isinstance(detector, tuple):
                dictionary, params = detector
                corners, ids, _ = cv2.aruco.detectMarkers(
                    frame, dictionary, parameters=params
                )
            else:
                corners, ids, _ = detector.detectMarkers(frame)

            if ids is None:
                continue

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_corners, marker_id in zip(corners, ids.flatten()):
                image_points = marker_corners.reshape(4, 2).astype(np.float32)

                found, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if not found:
                    continue

                cv2.drawFrameAxes(
                    frame,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
                    rvec,
                    tvec,
                    MARKER_SIZE * 0.5,
                )

                x, y, z = tvec.flatten()
                corner = tuple(marker_corners.reshape(4, 2)[0].astype(int))
                label = f"{family} id={marker_id} depth={z:.2f} m"

                cv2.putText(
                    frame,
                    label,
                    corner,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 0),
                    2,
                )

        cv2.imshow("ArUco pose estimation - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
