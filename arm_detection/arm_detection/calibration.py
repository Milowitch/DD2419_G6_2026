# import cv2
# import numpy as np
# import glob

# # Checkerboard settings
# CHECKERBOARD = (8, 6)  # inner corners (adjust to your grid!)
# square_size = 2.8125  # cm

# # Prepare object points (real world)
# objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# objp *= square_size

# objpoints = []  # 3D points
# imgpoints = []  # 2D points

# images = glob.glob("home/dd2419_ws/src/arm_detection/calibration/*.jpg")

# for fname in images:
#     img = cv2.imread(fname)
#     if img is None:
#         print("Failed to load:", fname)
#         continue

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

#     if ret:
#         objpoints.append(objp)
#         imgpoints.append(corners)

#         if img_shape is None:
#             img_shape = gray.shape[::-1]

#         cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
#         cv2.imshow("img", img)
#         cv2.waitKey(200)

# cv2.destroyAllWindows()

# # Calibration
# ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1], None, None
# )

# print("Camera matrix K:\n", K)
# print("Distortion coefficients:\n", distCoeffs)

import cv2
import numpy as np
import glob

# Checkerboard settings
CHECKERBOARD = (8, 4)  # inner corners (adjust to your grid!)
square_size = 2.8125  # cm

# Prepare object points (real world)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = glob.glob("/home/robot/dd2419_ws/src/arm_detection/calibration/grid.png")
print("Found images:", images)

img_shape = None  # define before loop

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print("Failed to load:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow("img", img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# Make sure we found at least one chessboard
if len(objpoints) == 0 or img_shape is None:
    raise ValueError("No chessboard corners detected. Check your images and CHECKERBOARD settings.")

# Calibration
ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print("Camera matrix K:\n", K)
print("Distortion coefficients:\n", distCoeffs)