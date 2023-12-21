# Author: ChatGPT
# OpenAI. (2023). ChatGPT [Large language model]. https://chat.openai.com
import numpy as np


def calculate_entropy(image: np.ndarray):
    """
    Calculate the entropy of an image.

    Args:
    image (numpy.ndarray): The image for which to calculate the entropy.

    Returns:
    float: The entropy of the image.
    """
    entropy = 0
    for c in range(image.shape[2]):
        # Flatten the image array and calculate the histogram
        histogram, _ = np.histogram(image[:, :, c].flatten(), bins=256, range=(0, 256), density=True)
        # Remove zeros to avoid issues with log
        histogram_nonzero = histogram[histogram > 0]
        # Calculate the entropy
        entropy += -np.sum(histogram_nonzero * np.log2(histogram_nonzero))
    return entropy