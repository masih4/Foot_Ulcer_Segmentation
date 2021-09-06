import cv2
import numpy as np

def TTA(test_image):
    augmented_images = np.zeros((8, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
    augmented_images[0, :, :, :] = test_image
    augmented_images[1, :, :, :] = cv2.rotate(test_image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images[2, :, :, :] = cv2.rotate(test_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    augmented_images[3, :, :, :] = cv2.rotate(test_image, cv2.ROTATE_180)
    augmented_images[4, :, :, :] = cv2.flip(test_image, 1)
    augmented_images[5, :, :, :] = cv2.flip(augmented_images[1, :, :, :], 1)
    augmented_images[6, :, :, :] = cv2.flip(augmented_images[3, :, :, :], 1)
    augmented_images[7, :, :, :] = cv2.flip(augmented_images[2, :, :, :], 1)
    return augmented_images

def TTA_reverse(predictions):
    reverse = np.zeros((predictions.shape[0], predictions.shape[1], predictions.shape[2]))

    reverse[0, :, :] = predictions[0 ,:, :]
    reverse[1, :, :] = cv2.rotate(predictions[1 ,:, :], cv2.ROTATE_90_COUNTERCLOCKWISE)
    reverse[2, :, :] = cv2.rotate(predictions[2 ,:, :], cv2.ROTATE_90_CLOCKWISE)
    reverse[3, :, :] = cv2.rotate(predictions[3 ,:, :], cv2.ROTATE_180)
    reverse[4, :, :] = cv2.flip(predictions[4, :, :], 1)
    reverse[5, :, :] = cv2.flip(predictions[5, :, :], 1)
    reverse[5, :, :] = cv2.rotate(reverse[5, :, :], cv2.ROTATE_90_COUNTERCLOCKWISE)
    reverse[6, :, :] = cv2.flip(predictions[6, :, :], 1)
    reverse[6, :, :] = cv2.rotate(reverse[6, :, :], cv2.cv2.ROTATE_180)
    reverse[7, :, :] = cv2.flip(predictions[7, :, :], 1)
    reverse[7, :, :] = cv2.rotate(reverse[7, :, :], cv2.ROTATE_90_CLOCKWISE)
    return reverse