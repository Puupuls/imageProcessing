#  Izstrādāt programmu, kas koriģē krāsas no kvēlspuldzes apgaismojuma uz apmākušās dienas gaismu.
import cv2
import numpy as np
import matplotlib.pyplot as plt


def color_temp(image: np.ndarray, plot=True):
    if not len(image.shape) == 3:
        raise "Image not in expected shape, required num_dims=3"
    img_processed = np.zeros(image.shape, dtype=np.float32)

    temp1 = 3000
    temp2 = 6500
    # https://andi-siess.de/rgb-to-color-temperature/
    rgb1 = np.array([255, 180, 107])
    rgb2 = np.array([255, 249, 253])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                img_processed[i, j, k] = image[i, j, k] * rgb2[k] / rgb1[k]

    img_processed = np.clip(img_processed, 0, 255)
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
    img_filtered = color_temp(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), plot=True)
    cv2.imwrite('demo_result.jpg', cv2.cvtColor(img_filtered, cv2.COLOR_RGB2BGR))

