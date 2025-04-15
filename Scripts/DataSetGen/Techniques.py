import cv2
import numpy as np

DELIMITER = '<EOF>'

def text_to_binary(text):
    """Convert text to binary string."""
    return ''.join(format(ord(char), '08b') for char in text)

def fit_message_to_image(image, message):
    """Ensure the message (including delimiter) fits within the image capacity."""
    binary_message = text_to_binary(message)
    max_bits = image.size  # Total available values (H x W x C)

    if len(binary_message) > max_bits:
        print(f"[WARNING] Message too long ({len(binary_message)} bits). Truncating to fit.")
        max_chars = (max_bits // 8) - len(DELIMITER)  # Adjust for delimiter
        message = message[:max_chars].rstrip() + DELIMITER
        binary_message = text_to_binary(message)

    return binary_message

def embed_lsb(image, message):
    """Embed a message into an image using LSB steganography."""
    binary_message = fit_message_to_image(image, message)
    data_index = 0
    message_length = len(binary_message)

    h, w, c = image.shape
    flat_image = image.flatten()

    for i in range(len(flat_image)):
        if data_index < message_length:
            flat_image[i] = (flat_image[i] & 0b11111110) | int(binary_message[data_index])
            data_index += 1
        else:
            break

    return flat_image.reshape((h, w, c)), message_length

def embed_dct(image, message):
    """Embed a message into an image using DCT steganography."""
    h, w, c = image.shape
    message_bits = text_to_binary(message + DELIMITER)

    img_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    block_size = 8
    bit_index = 0
    message_length = len(message_bits)

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if bit_index >= message_length:
                break

            y_block = img_ycbcr[i:i+block_size, j:j+block_size, 0]
            dct_block = cv2.dct(y_block)

            coef_x, coef_y = 3, 4
            coef_value = round(dct_block[coef_x, coef_y])
            coef_bin = format(int(coef_value), '08b')
            new_coef_bin = coef_bin[:-1] + message_bits[bit_index]
            dct_block[coef_x, coef_y] = int(new_coef_bin, 2)

            img_ycbcr[i:i+block_size, j:j+block_size, 0] = cv2.idct(dct_block)
            bit_index += 1

        if bit_index >= message_length:
            break

    img_ycbcr = np.clip(img_ycbcr, 0, 255).astype(np.uint8)
    stego_image = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2BGR)

    return stego_image, message_length