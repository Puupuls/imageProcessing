#  Realizēt Sobela un Laplasa operatorus, pārbaudīt tos attēla robežu noteikšanai.
import cv2
import numpy as np
import matplotlib.pyplot as plt


def edge_detect_sobel(image: np.ndarray, plot=True):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobela kodoli
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    img_processed_x = cv2.filter2D(image, -1, sobel_x)
    img_processed_y = cv2.filter2D(image, -1, sobel_y)
    img_processed = img_processed_x + img_processed_y

    # Normalizējam
    img_processed = img_processed - np.min(img_processed)
    img_processed = img_processed / np.max(img_processed)
    img_processed = img_processed * 255

    img_processed = img_processed.astype(np.uint8)
    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img_processed)
        plt.tight_layout()
        plt.show()
    return img_processed


def edge_detect_laplace(image: np.ndarray, plot=True):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Laplasa kodols
    laplace = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

    img_processed = cv2.filter2D(image, -1, laplace)

    # Normalizējam
    img_processed = img_processed - np.min(img_processed)
    img_processed = img_processed / np.max(img_processed)
    img_processed = img_processed * 255

    img_processed = img_processed.astype(np.uint8)
    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img_processed)
        plt.tight_layout()
        plt.show()
    return img_processed


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")
    img_filtered = edge_detect_sobel(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), plot=True)
    cv2.imwrite('demo_result_sobel.jpg', cv2.cvtColor(img_filtered, cv2.COLOR_RGB2BGR))
    img_filtered = edge_detect_laplace(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), plot=True)
    cv2.imwrite('demo_result_laplace.jpg', cv2.cvtColor(img_filtered, cv2.COLOR_RGB2BGR))

