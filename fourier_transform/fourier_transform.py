# 3. Realizēt filtrēšanu ar Furjē transformācijas palīdzību
# Izmantojam gausian blur implementāciju kā pamatu
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fourier_transform(image: np.ndarray, kernel_size=3, sigma=2, plot=True):
    if not kernel_size % 2 == 1:
        raise "Kernel size must be an odd number"
    if not len(image.shape) == 3:
        raise "Image not in expected shape, required num_dims=3"

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

    # ================================================================================================================
    # Furjē transformācijas kods


    def fourier2d(image):
        n, m = image.shape[:2]
        result = np.zeros((n, m, image.shape[2]), dtype=complex)

        # Pārnesam iekšējos iteratorus uz numpy masīviem lai vaŗetu veikt matricu operācijas un paātrināt procesu
        l = np.arange(m)
        k = np.arange(n)

        # Izveidojam 1, -1 matricu lai centrētu furjē transformāciju
        c = np.ones((n, m, 1))
        c[1::2, :] = -1
        c[:, 1::2] *= -1

        for u in range(n):
            print(f"Calculating Fourier transform for row {u}/{n}")
            for v in range(m):

                # Izveidojam 2d matricu ar k un l vērtībām lai aizstātu iekšējos divus ciklus ar matricu operāciju
                uk = u * k / n
                vl = v * l / m
                p = np.zeros((n, m, 1))
                p += vl.reshape((m, 1))
                p += uk.reshape((n, 1, 1))

                # Izrēķinam eksponentu matricu
                e = np.exp(-2j * np.pi * p)

                # Izrēķinam furjē transformāciju pikselim
                s = image * e# * c
                s = np.sum(s, axis=(0, 1))
                s /= n * m

                # Saglabājam rezultātu
                result[u, v] = s
        return result

    def inverse_fourier2d(image):
        n, m = image.shape[:2]
        result = np.zeros((n, m, image.shape[2]), dtype=complex)

        # Pārnesam iekšējos iteratorus uz numpy masīviem lai vaŗetu veikt matricu operācijas un paātrināt procesu
        u = np.arange(n)
        v = np.arange(m)

        for k in range(n):
            print(f"Calculating inverse Fourier transform for row {k}/{n}")
            for l in range(m):
                # Izveidojam 2d matricu ar k un l vērtībām lai aizstātu iekšējos divus ciklus ar matricu operāciju
                uk = u * k / n
                vl = v * l / m
                p = np.zeros((n, m, 1))
                p += vl.reshape((m, 1))
                p += uk.reshape((n, 1, 1))

                # Izrēķinam eksponentu matricu
                e = np.exp(2j * np.pi * p)

                # Izrēķinam furjē transformāciju pikselim
                s = image * e
                s = np.sum(s, axis=(0, 1))
                # s /= n * m

                # Salabojam zīmi reālajai daļai, jo tā var būt negatīva
                s = abs(s.real) + s.imag * 1j

                # Saglabājam rezultātu
                result[k, l] = s
        return result


    # Iegūstam attēla furjē transformāciju
    print("Calculating Fourier transform for image...")
    image_ft = fourier2d(image)

    kernel_large = np.zeros(image.shape)
    # Ieliekam centrā mūsu filtra kodoli
    diffx = (image.shape[0] - kernel_size) // 2
    diffy = (image.shape[1] - kernel_size) // 2
    kernel_large[diffx:diffx + kernel_size, diffy:diffy + kernel_size] = kernel
    # Izrādas kodolu vajag pārvietot uz stūriem ar np.fft.ifftshift funkciju
    # Mana implementācija šai funkcijai:
    def ifftshift(image):
        shiftx = image.shape[0] // 2
        shifty = image.shape[1] // 2
        return np.roll(image, (shiftx, shifty), axis=(0, 1))
    kernel_large = ifftshift(kernel_large)
    # Iegūstam kodola furjē transformāciju
    print("Calculating Fourier transform for kernel...")
    kernel_ft = fourier2d(kernel_large)

    image_ft_filtered = image_ft * kernel_ft
    # Iegūstam filtrēto attēlu
    print("Calculating inverse Fourier transform for filtered image...")
    image_processed = inverse_fourier2d(image_ft_filtered).real
    # Nonormalizejam attēlu, lai tas būtu 0-255
    image_processed = (image_processed - np.min(image_processed)) / (np.max(image_processed) - np.min(image_processed))
    image_processed = (image_processed * 255).astype(np.uint8)
    # ================================================================================================================

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
    img_filtered = fourier_transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), kernel_size=25, plot=True)
    cv2.imwrite('demo_result.jpg', cv2.cvtColor(img_filtered, cv2.COLOR_RGB2BGR))

