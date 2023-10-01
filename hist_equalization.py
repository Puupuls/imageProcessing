# Realizēt histogrammas vienmērīgošanas algoritmu
# Implementācija paredzēta krāsainu attēlu apstrādei,
# implementēta gan krāsu izlīdināšana RGB formātā, gan vērtību izlīdzināšana HSV krāsu formātam
# Katrs no tiem dod cita veida rezultātu, jo izlīdzinot RGB tiek izmainīts krāsu balanss attēlā

import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalize_histogram_rgb(image: np.ndarray, plot=False):
    img_processed = np.zeros_like(image)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Izveidojam krāsu histogrammas
    hist_red = np.array([np.count_nonzero(red == i) for i in range(256)])
    hist_green = np.array([np.count_nonzero(green == i) for i in range(256)])
    hist_blue = np.array([np.count_nonzero(blue == i) for i in range(256)])

    # Izveidojam kumlatīvās histogrammas
    cum_hist_red = np.array([np.sum(hist_red[:i+1]) for i in range(256)])
    cum_hist_green = np.array([np.sum(hist_green[:i+1]) for i in range(256)])
    cum_hist_blue = np.array([np.sum(hist_blue[:i+1]) for i in range(256)])

    # Normalizējam kumlatīvo histogrammu

    cum_hist_red_norm = cum_hist_red / cum_hist_red[-1]
    cum_hist_green_norm = cum_hist_green / cum_hist_green[-1]
    cum_hist_blue_norm = cum_hist_blue / cum_hist_blue[-1]

    red = np.round(cum_hist_red_norm[red] * 255)
    green = np.round(cum_hist_green_norm[green] * 255)
    blue = np.round(cum_hist_blue_norm[blue] * 255)


    # Izveidojam krāsu histogrammas
    hist_red_done = np.array([np.count_nonzero(red == i) for i in range(256)])
    hist_green_done = np.array([np.count_nonzero(green == i) for i in range(256)])
    hist_blue_done = np.array([np.count_nonzero(blue == i) for i in range(256)])

    # Izveidojam kumlatīvās histogrammas
    cum_hist_red_done = np.array([np.sum(hist_red_done[:i+1]) for i in range(256)])
    cum_hist_green_done = np.array([np.sum(hist_green_done[:i+1]) for i in range(256)])
    cum_hist_blue_done = np.array([np.sum(hist_blue_done[:i+1]) for i in range(256)])

    # Saliekam atpakaļ attēlu
    img_processed[:, :, 0] = red
    img_processed[:, :, 1] = green
    img_processed[:, :, 2] = blue

    if plot:
        plt.subplot(4, 2, 1)
        plt.plot(hist_red, color='r')
        plt.plot(hist_green, color='g')
        plt.plot(hist_blue, color='b')

        plt.subplot(4, 2, 3)
        plt.plot(cum_hist_red, color='r')
        plt.plot(cum_hist_green, color='g')
        plt.plot(cum_hist_blue, color='b')

        plt.subplot(4, 2, 2)
        plt.plot(hist_red_done, color='r')
        plt.plot(hist_green_done, color='g')
        plt.plot(hist_blue_done, color='b')

        plt.subplot(4, 2, 4)
        plt.plot(cum_hist_red_done, color='r')
        plt.plot(cum_hist_green_done, color='g')
        plt.plot(cum_hist_blue_done, color='b')

        plt.subplot(2, 2, 3)
        plt.imshow(image)
        plt.subplot(2, 2, 4)
        plt.imshow(img_processed)

        plt.tight_layout()
        plt.show()

    return img_processed


def equalize_histogram_hsv(image: np.ndarray, plot=False):
    value = image[:, :, 2]

    # Izveidojam spilgtuma histogrammu
    hist_value = np.array([np.count_nonzero(value == i) for i in range(256)])

    # Izveidojam kumlatīvo histogrammu
    cum_hist_value = np.array([np.sum(hist_value[:i+1]) for i in range(256)])

    # Normalizējam kumlatīvo histogrammu
    cum_hist_value_norm = cum_hist_value / cum_hist_value[-1]
    value = np.round(cum_hist_value_norm[value] * 255)

    # Izveidojam spilgtuma histogrammu
    hist_value_done = np.array([np.count_nonzero(value == i) for i in range(256)])

    # Izveidojam kumlatīvo histogrammu
    cum_hist_value_done = np.array([np.sum(hist_value_done[:i+1]) for i in range(256)])

    # Saliekam atpakaļ attēlu
    img_processed = np.zeros_like(image)
    img_processed[:, :, 0] = image[:, :, 0]
    img_processed[:, :, 1] = image[:, :, 1]
    img_processed[:, :, 2] = value

    if plot:
        plt.subplot(4, 2, 1)
        plt.plot(hist_value, color='#999')

        plt.subplot(4, 2, 3)
        plt.plot(cum_hist_value, color='#999')

        plt.subplot(4, 2, 2)
        plt.plot(hist_value_done, color='#999')

        plt.subplot(4, 2, 4)
        plt.plot(cum_hist_value_done, color='#999')

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(img_processed, cv2.COLOR_HSV2RGB))

        plt.tight_layout()
        plt.show()

    return img_processed


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")

    img_filtered = equalize_histogram_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), True)
    cv2.imwrite('demo_result.jpg', cv2.cvtColor(img_filtered, cv2.COLOR_RGB2BGR))

    img_filtered = equalize_histogram_hsv(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), True)
    cv2.imwrite('demo_result.jpg', cv2.cvtColor(img_filtered, cv2.COLOR_HSV2BGR))

