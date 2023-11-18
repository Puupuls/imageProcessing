# 5. Izstrādāt programmu, kas doto attēlu palielina 4 reizes, lietojot bilineāro (vai bikubisko) interpolāciju.
import cv2
import numpy as np
import matplotlib.pyplot as plt


def upscale(image: np.ndarray, plot=True):
    if not len(image.shape) == 3:
        raise "Image not in expected shape, required num_dims=3"

    image_processed = np.zeros((image.shape[0]*2, image.shape[1]*2, image.shape[2]), dtype=np.uint16)
    image = image.astype(np.uint16)
    for i in range(image.shape[2]):
        for j in range(image.shape[0]*2-1):
            for k in range(image.shape[1]*2-1):
                if j % 2 == 0 and k % 2 == 0:
                    image_processed[j, k, i] = image[j//2, k//2, i]
                elif j % 2 == 1 and k % 2 == 0:
                    image_processed[j, k, i] = (image[j//2, k//2, i] + image[j//2+1, k//2, i]) // 2
                elif j % 2 == 0 and k % 2 == 1:
                    image_processed[j, k, i] = (image[j//2, k//2, i] + image[j//2, k//2+1, i]) // 2
                else:
                    image_processed[j, k, i] = (image[j//2, k//2, i] + image[j//2+1, k//2, i] + image[j//2, k//2+1, i] + image[j//2+1, k//2+1, i])//4

    image_processed = image_processed.astype(np.uint8)
    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(image_processed)
        plt.tight_layout()
        plt.show()
    return image_processed


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")
    img_filtered = upscale(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), plot=True)
    cv2.imwrite('demo_result.png', cv2.cvtColor(img_filtered, cv2.COLOR_RGB2BGR))

