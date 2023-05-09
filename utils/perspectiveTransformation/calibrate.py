from transform import four_point_transform
import numpy as np
import argparse
import cv2

"""
python calibrate.py --image image.jpg
"""

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])


pts = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0
AR = (740, 1280)
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])

# function to select four points on a image to capture desired region
def draw_circle(event, x, y, flags, param):
    global image
    global pointIndex
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        pts[pointIndex] = (x, y)
        # print(pointIndex)
        if pointIndex == 3:
            cv2.line(image, pts[0], pts[1], (0, 255, 0), thickness=2)
            cv2.line(image, pts[0], pts[2], (0, 255, 0), thickness=2)
            cv2.line(image, pts[1], pts[3], (0, 255, 0), thickness=2)
            cv2.line(image, pts[2], pts[3], (0, 255, 0), thickness=2)
        pointIndex = pointIndex + 1


def show_window():
    while True:
        # print(pts,pointIndex-1)
        cv2.imshow("img", image)

        if pointIndex == 4:

            break

        if cv2.waitKey(20) & 0xFF == 27:
            break


cv2.namedWindow("img")
cv2.setMouseCallback("img", draw_circle)
show_window()



destination_size = (600, 400)
W, H = destination_size
transformed_points = np.float32([[0,0],[W,0],[0,H],[W,H]])
pts = np.asarray(pts, dtype=np.float32)
print(pts)
print(transformed_points)
M_cv = cv2.getPerspectiveTransform(pts[:4], transformed_points)
print(M_cv)
warped = cv2.warpPerspective(image, M_cv, destination_size)

# warped_2, temp, transformation_matrix = four_point_transform(
#     image, np.array(pts[:4], dtype="float32"), pts[4]
# )
# print(transformation_matrix)
np.save("transformation_matrix.npy", M_cv)

# cv2.circle(warped, tuple(map(int, tuple(temp[0][0]))), 1, (0, 255, 0), -1)
# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped) #Uncommment if you want to see the transformed image
# cv2.imshow("Warped_2", warped_2)
cv2.waitKey(0)
