import cv2
import numpy as np


def normalize_histogram(image: np.ndarray):
    pass


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")
    img_filtered = normalize_histogram(img)

    cv2.imshow("Normalized histogram", img_filtered)