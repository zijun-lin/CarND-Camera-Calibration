import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
nx = 8
ny = 6
objp = np.zeros((nx*ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('calibration_wide/GO*.jpg')
print(images)

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, add object points, image points
    if ret is True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imshow('img', img)
        print(fname)
        # cv2.waitKey(500)
        plt.show()

cv2.destroyAllWindows()

points_pickle = {'objpoints': objpoints, 'imgpoints': imgpoints}
pickle.dump(points_pickle, open('points_pickle.p', 'wb'))
print('Save point parameters')


# Test undistortion on an image
image = cv2.imread('calibration_wide/test_image.jpg')
image_size = (image.shape[1], image.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

dst = cv2.undistort(image, mtx, dist, None, mtx)
cv2.imwrite('calibration_wide/test_undist.jpg', dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
points_pickle = {'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
pickle.dump(points_pickle, open('mtx_dist_pickle.p', 'wb'))
print('Save mtx dist parameters')

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()