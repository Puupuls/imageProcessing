# Realizēt attēlu saspiešanu ar paredzēšanas metodi.
import warnings

import cv2
import numpy as np

from entropy import calculate_entropy
warnings.simplefilter('ignore', category=RuntimeWarning)


def compress_image(image: np.ndarray):
    compressed = np.empty_like(image, dtype=np.int16)
    for channel in range(image.shape[2]):
        for i in range(image.shape[0]):
            # Paturam rindas pirmo pixeli kāds tas bija
            compressed[i, 0, channel] = image[i, 0, channel]
            for j in range(1, image.shape[1]):
                compressed[i, j, channel] = image[i, j, channel] - image[i, j - 1, channel]

                # Ietaupam vietu atmetot mazas starpības
                # if np.abs(compressed[i, j, channel]) < 5:
                #     compressed[i, j, channel] = 0
    compressed = np.mod(compressed, 256).astype(np.uint8)
    return compressed

def decompress_image(compressed: np.ndarray):
    decompressed = np.zeros_like(compressed, dtype=np.int16)
    for channel in range(compressed.shape[2]):
        for i in range(compressed.shape[0]):
            # Pirmais pikselis nav jāatspiež
            decompressed[i, 0, channel] = compressed[i, 0, channel]
            for j in range(1, compressed.shape[1]):
                decompressed[i, j, channel] = decompressed[i, j - 1, channel] + compressed[i, j, channel]

    decompressed = np.mod(decompressed, 256).astype(np.uint8)
    return decompressed


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")
    print("Original image entropy: ", calculate_entropy(img))
    compressed = compress_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print("Compressed image entropy: ", calculate_entropy(compressed))
    decompressed = decompress_image(compressed)
    print("Decompressed image entropy: ", calculate_entropy(decompressed))
    cv2.imwrite("compressed.jpg", cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR))
    cv2.imwrite("decompressed.jpg", cv2.cvtColor(decompressed, cv2.COLOR_RGB2BGR))



