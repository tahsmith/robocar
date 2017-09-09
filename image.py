import cv2
import numpy as np


rho = 1
theta = np.pi/180
threshold = 3
min_line_length = 20
max_line_gap = 1
kernel_size = 5

low_threshold = 50
high_threshold = 150

image_size = (256, 352)
crop = ((100, 0), (200, 352))


def apply_thresholds(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    return (
        (hsv[:, :, 2] < 60)
    )


def pipeline(img):
    img = img[crop[0][0]:crop[1][0], crop[0][1]:crop[1][1], :]
    return apply_thresholds(img)


def hough_lines(image):
    edges = np.uint8(pipeline(image))
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    if lines is not None:
        return lines.reshape(-1, 4)
    return np.empty((0, 4))


def split_lines(lines):
    left = np.zeros(lines.shape[0], dtype=np.bool)
    right = np.zeros(lines.shape[0], dtype=np.bool)
    for i in range(lines.shape[0]):
        x0, y0, x1, y1 = lines[i, :]
        if x1 < x0:
            x1, x0 = x0, x1
            y1, y0 = y0, y1
        m = (y1 - y0) / (x1 - x0)
        if m > 0 and (x0 > image_size[1] // 2):
            right[i] = True
        if m <=0 and (x1 <= image_size[1] // 2):
            left[i] = True
    return lines[left, :], lines[right, :]


def fit_best_line(lines):
    if lines.shape[0] < 1:
        return None

    lines = lines.reshape(-1, 2)
    return np.polyfit(lines[:, 1], lines[:, 0], 1)


def x_intercept(line):
    return line[0] * crop[1][0] - crop[0][0] + line[1]


def get_lane_lines(image):
    lines = hough_lines(image)
    return [fit_best_line(lines) for lines in split_lines(lines)]