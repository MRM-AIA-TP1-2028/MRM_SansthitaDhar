# BlueObjectSelection_TennisBallDetection_MRM
## 1. Problem Statement

The objective is to detect a tennis ball and blue objects from a live camera stream using Python and OpenCV.

The blue object and tennis ball are treated as separate objects. The blue object can have any shape, whereas the tennis ball is expected to be approximately circular.

The program displays the blue object's area, bounding box, center trajectory, and position relative to the tennis ball. For the tennis ball, it displays the center, radius, and area.

## 2. Approach

### A. Live Camera Input
OpenCV's VideoCapture function accesses the webcam. Each frame is processed independently to detect the objects in real time.

### B. Color Segmentation
Each frame is converted from BGR to HSV color space. HSV is useful for color-based detection because hue represents color separately from brightness.

Two independent masks are created:
- Blue mask: identifies pixels within the selected blue HSV range.
- Tennis ball mask: identifies pixels within the yellow-green HSV range.

### C. Noise Removal
Morphological opening removes small unwanted regions, while morphological closing fills small gaps in detected regions.

### D. Blue Object Detection and Tracking
Contours are extracted from the blue mask. The user selects a blue object by clicking on it.

The program calculates the contour's centroid using image moments and its bounding box using `cv2.boundingRect()`. The contour area is calculated using `cv2.contourArea()`.

The object's center is tracked between frames by selecting the detected contour closest to its previous center. A line displays its trajectory.

### E. Tennis Ball Detection
Contours are extracted from the tennis ball mask. Candidate contours are filtered using minimum area and circularity.

The center and radius are estimated using `cv2.minEnclosingCircle()`. The ball's area is calculated from its contour.

The ball is detected independently of the blue object.

### F. Relative Position
When both objects are detected, a line connects their centers.

The relative displacement is calculated as:

- dx = blue center x - ball center x
- dy = blue center y - ball center y

The center-to-center distance is calculated using the Euclidean distance formula.

## 3. Method Used

The solution uses:
- Python
- OpenCV
- NumPy
- HSV color thresholding
- Contour detection
- Image moments
- Bounding rectangles
- Circularity filtering
- Minimum enclosing circles
- Nearest-centroid tracking

## 4. Installation and Execution

Install the required packages:

```bash
python3 -m pip install opencv-python numpy
```

Run the program:

```bash
python3 main.py
```

Click inside a blue object to select it.

Controls:
- Left mouse click: select a blue object.
- `r`: reset the selection.
- `q` or `Esc`: quit the program.

## 5. Output

### Blue Object
- Contour area in pixels squared
- Bounding box coordinates and dimensions
- Center coordinates
- Center trajectory
- Relative displacement from the tennis ball
- Distance between the two centers

### Tennis Ball
- Center coordinates
- Estimated radius in pixels
- Contour area in pixels squared

Separate masks are used to ensure that the two detection processes remain independent.

## 6. Other Possible Methods

### A. Background Subtraction
Background subtraction can isolate moving objects from a relatively stationary background. However, it may fail when the background changes or when objects remain stationary.

### B. Hough Circle Transform
`cv2.HoughCircles()` can detect circular shapes. It is suitable for tennis ball candidates but cannot reliably detect blue objects of arbitrary shape.

### C. Template Matching
Template matching compares image regions against a predefined object template. It can work when the object's appearance and scale remain consistent, but is less robust to rotation and changes in size.

### D. Object Detection Models
A trained model such as YOLO can detect tennis balls and blue objects when suitable labeled training data is available. This can be more robust under changing backgrounds, but requires additional setup and computational resources.

### E. CSRT Tracker or Kalman Filter
A CSRT tracker can track a selected object based on its visual appearance. A Kalman filter can estimate an object's position and predict its next location. These methods can improve tracking when combined with detection.

## 7. Limitations

- Detection depends on suitable lighting and HSV thresholds.
- Blue backgrounds or similarly colored objects may cause false detections.
- The tennis ball's color can overlap with other yellow-green objects.
- Nearest-centroid tracking can switch objects when they cross paths.
- The estimated radius and contour area may vary with camera angle and occlusion.

## 8. Conclusion

The program combines HSV color segmentation, contour-based shape-independent detection, circularity filtering, and centroid tracking to detect a blue object and a tennis ball simultaneously in a live camera stream.
