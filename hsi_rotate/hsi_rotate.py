# 5. Izstrādāt programmu, kas doto attēlu palielina 4 reizes, lietojot bilineāro (vai bikubisko) interpolāciju.
import cv2
import numpy as np


def hsi_rotate(image: np.ndarray, degrees=10, plot=True):
    if not len(image.shape) == 3:
        raise "Image not in expected shape, required num_dims=3"

    image_processed = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            image_processed[y, x, :] = image[y, x, :]
            image_processed[y, x, 0] += degrees / 360 * 255
            image_processed[y, x, 0] %= 255

    return image_processed.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Izmantoju HLS, princips tieši tas pats kas HSI, bet HSI nedaudz atšķiras gaišuma/intensitātes kanāls. HSI nav implementēts nevienā standarta bibliotēkā, tāpēc izmantoju HLS
    img_filtered = hsi_rotate(img_hls, degrees=180)
    img_filtered_bgr = cv2.cvtColor(img_filtered, cv2.COLOR_HLS2BGR)
    cv2.imwrite('demo_result.jpg', img_filtered_bgr)

