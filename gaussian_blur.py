# Realizēt izpludināšanu ar Gausa filtru
# Lieliem attēliem (kā pievienotais demo.jpg attēls izpilde ir ilga jo python cikli nav effektīvi un iterēt
# Pāri 24 megapixeļu attēlam ir ilgi.
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_blur(image: np.ndarray, kernel_size=3, sigma=2, plot=True):
    if not kernel_size % 2 == 1:
        raise "Kernel size must be an odd number"
    if not len(image.shape) == 3:
        raise "Image not in expected shape, required num_dims=3"
    img_processed = np.zeros_like(image)

    kernel = np.zeros((kernel_size, kernel_size))
    c = kernel_size//2
    for i in range(kernel_size):
        for j in range(kernel_size):
            posx = i - c
            posy = j - c
            kernel[j, i] = np.exp(-(posx ** 2 + posy ** 2) / (2 * sigma ** 2))
    # Normalizējam lai attēla kopējā spožuma vērtība nemainītos
    kernel = kernel / np.sum(kernel)
    # Pārveidojam kodolu par trīsdimensionālu, lai varētu to pielietot trīsdimensionālam attēlam
    kernel = kernel.reshape((kernel_size, kernel_size, 1))

    # Izveidojam "atspīdumu" aiz attēla robežām lai attēla malas nebūtu tumšākas
    image_padded = np.pad(image, ((c,), (c,), (0,)), 'symmetric')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            img_processed[y, x] = np.sum(image_padded[y:y+kernel_size, x:x+kernel_size] * kernel, axis=(0, 1))

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
    img_s = cv2.resize(img, (600, 400))  # Samazinam attēlu, lai paātrinātu procesu
    img_filtered = gaussian_blur(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB), kernel_size=101, plot=True)
    img_filtered = cv2.resize(img_filtered, (img.shape[1], img.shape[0]))  # Atgriežam oriģinālo izmēru
    cv2.imwrite('demo_result.jpg', cv2.cvtColor(img_filtered, cv2.COLOR_RGB2BGR))

