import cv2
import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from torch.nn import functional as F
from skimage.filters import threshold_multiotsu


def apply_brightness_contrast(input_img: np.array, brightness: int = 0, contrast: int = 0) -> np.array:
    """Apply to the image specified values of brightness and contrast

    @param input_img: Input image for brightness/contrast change
    @param brightness: The value of the brightness which should be added to the image
    @param contrast: The value of the brightness which should be added to the image
    @returns: Image with applied brightness and contrast change

    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        alpha_c = 131*(contrast + 127)/(127*(131-contrast))
        gamma_c = 127*(1-alpha_c)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def augment_image(image: np.array) -> np.array:
    """Augment the image. The resize and shift augmentations are applied to the different elements located on the image.
    Different elements are detected using Otsu thresholding. Separate elements are shifted and scaled after the detection.
    Finally, all elements are placed on the original image and combined.

    @param input_img: Input image for augmentation
    @returns: Image with applied brightness and contrast change

    """
    blurred_image = cv2.GaussianBlur(image, (5, 5), 5)
    green_channel_image = blurred_image[:, :, 1]
    if len(np.unique(green_channel_image)) < 5:
        return image

    _, original_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    thresholds = threshold_multiotsu(green_channel_image)

    regions = np.digitize(green_channel_image, bins=thresholds)
    regions = regions/np.max(regions)

    values, counts = np.unique(regions, return_counts=True)
    max_elements_count = np.argmax(counts)
    background_values = values[max_elements_count]

    background_mask = regions.copy()
    background_mask[background_mask == background_values] = 255
    background_mask[background_mask != 255] = 0

    foreground_mask = regions.copy()
    foreground_mask[foreground_mask != background_values] = 255
    foreground_mask[foreground_mask == background_values] = 0

    kernel = np.ones((5, 5))
    background_mask = cv2.erode(background_mask, kernel, iterations=1)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)

    background_mask = np.uint8(background_mask)
    foreground_mask = np.uint8(foreground_mask)
    original_mask = np.uint8(original_mask[:, :, 0])

    blured_input_image = cv2.medianBlur(image, 31)

    background = cv2.bitwise_and(image, image, mask=background_mask)
    _, blured_mask = cv2.threshold(background, 1, 255, cv2.THRESH_BINARY_INV)
    blured_mask = np.uint8(blured_mask[:, :, 0])

    blured_input_image = cv2.bitwise_and(blured_input_image, blured_input_image, mask=original_mask)
    blured_input_image = cv2.bitwise_and(blured_input_image, blured_input_image, mask=blured_mask)

    final_background = background+blured_input_image
    foreground_image = cv2.bitwise_and(image, image, mask=foreground_mask)

    contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    foreground_augmented = np.zeros((foreground_mask.shape[0], foreground_mask.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):

        if hierarchy[0, i, 3] != -1:
            continue

        x, y, w, h = cv2.boundingRect(contours[i])
        sub_image = foreground_image[y:y+h, x:x+w, :].copy()

        scale_factor = 1-(np.random.rand(1)-0.5)/4
        new_size = [int(w*scale_factor), int(h*scale_factor)]
        resized_sub_img = cv2.resize(sub_image, new_size)

        shift_x = int((w-new_size[0])/2) + np.random.randint(-5, 5)
        shift_y = int((h-new_size[1])/2) + np.random.randint(-5, 5)

        x = x + shift_x
        y = y + shift_y

        if x + new_size[0] > foreground_image.shape[1]:
            new_size[0] = foreground_image.shape[1] - x
        if y + new_size[1] > foreground_image.shape[0]:
            new_size[1] = foreground_image.shape[0] - y

        if x < 0:
            new_size[0] += x
            x = 0
        if y < 0:
            new_size[1] += y
            y = 0

        if new_size[0] < 0:
            new_size[0] = 0

        if new_size[1] < 0:
            new_size[1] = 0

        _, sub_img_mask = cv2.threshold(resized_sub_img, 1, 255, cv2.THRESH_BINARY)
        sub_img_mask = np.uint8(sub_img_mask[:, :, 0])
        resized_sub_img = cv2.bitwise_and(resized_sub_img, resized_sub_img, mask=sub_img_mask)

        foreground_augmented[y:y+new_size[1], x:x+new_size[0], :] = resized_sub_img[:new_size[1], :new_size[0]]

    _, foreground_augmented_mask = cv2.threshold(foreground_augmented, 1, 255, cv2.THRESH_BINARY_INV)
    foreground_augmented_mask = np.uint8(foreground_augmented_mask[:, :, 0])
    final_image = cv2.add(cv2.bitwise_and(final_background, final_background,
                          mask=foreground_augmented_mask), foreground_augmented)

    difference_mask = np.abs(np.float32(final_image) - np.float32(final_background))
    difference_mask = np.uint8(255*(difference_mask/np.max(difference_mask)))
    difference_mask = cv2.cvtColor(difference_mask, cv2.COLOR_BGR2GRAY)

    _, final_mask_foreground = cv2.threshold(difference_mask, 1, 255, cv2.THRESH_BINARY)
    _, final_mask_background = cv2.threshold(difference_mask, 1, 255, cv2.THRESH_BINARY_INV)

    final_mask_foreground = np.uint8(final_mask_foreground)
    final_mask_background = np.uint8(final_mask_background)

    final_background_masked = cv2.bitwise_and(final_background, final_background, mask=final_mask_background)

    final_image_blurred = cv2.medianBlur(np.uint8(final_image), 3)
    final_foreground_masked = cv2.bitwise_and(final_image_blurred, final_image_blurred, mask=final_mask_foreground)

    return final_background_masked + final_foreground_masked


def align_images(im1: np.array, im2: np.array) -> np.array:

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(3, 3, dtype=np.float32)

    number_of_iterations = 500
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    try:
        warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)[1]
    except:
        return im2

    im2_aligned = cv2.warpPerspective(
        im2, warp_matrix, (im1.shape[1],
                           im1.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im2_aligned


def crop_smaller_patch(image: np.array, image_size: int = 8) -> np.array:
    """Crop the square patch from the center of the image

    @param input_img: Input image for cropping
    @param image_size: The size of the output image
    @returns: Cropped image patch

    """

    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    x_center = int(width/2)
    y_center = int(height/2)

    crop_half_size = int(image_size / 2)

    height_min = max(y_center-crop_half_size, 0)
    height_max = min(y_center+crop_half_size, image.shape[0])
    width_min = max(x_center-crop_half_size, 0)
    width_max = min(x_center+crop_half_size, image.shape[1])

    if len(image.shape) == 3:
        crop_image = image[height_min:height_max, width_min:width_max, :].copy()
    else:
        crop_image = image[height_min:height_max, width_min:width_max].copy()

    return crop_image


def save_embeddings(data: List, embedding_coreset: np.array, embedding_test: np.array) -> None:
    image_id = ''.join(data[-1][0].split('/')[-1].split('.'))
    np.save(f"data/{image_id}_coreset.npy", embedding_coreset)
    np.save(f"data/{image_id}_test.npy", embedding_test)


def visualize_scores(data: List, score_patches: np.array) -> None:
    plt.subplot(331), plt.imshow(score_patches[:, 0].reshape((16, 16)))
    plt.subplot(332), plt.imshow(score_patches[:, 1].reshape((16, 16)))
    plt.subplot(333), plt.imshow(score_patches[:, 2].reshape((16, 16)))
    plt.subplot(334), plt.imshow(score_patches[:, 3].reshape((16, 16)))
    plt.subplot(335), plt.imshow(score_patches[:, 4].reshape((16, 16)))
    plt.subplot(336), plt.imshow(score_patches[:, 5].reshape((16, 16)))
    plt.subplot(337), plt.imshow(score_patches[:, 6].reshape((16, 16)))
    plt.subplot(338), plt.imshow(score_patches[:, 7].reshape((16, 16)))
    plt.subplot(339), plt.imshow(np.uint8(data[2].permute(0, 2, 3, 1).cpu().numpy()[0]*255))
    plt.show()


def embedding_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Concatenate two feature embeddings into a single one

    @param x: First embedding
    @param y: Second embedding
    @returns: Result of the concatenation

    """
    B1, C1, H1, W1 = x.size()
    B2, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B1, C1, -1, H2, W2)
    z = torch.zeros(B1, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B1, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


def reshape_embedding(embedding: np.array) -> List[np.array]:
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


def crop_resize(image: np.array, crop_size: int, resize_size: List[int] = [128, 128]) -> np.array:
    image_cropped = crop_smaller_patch(image, crop_size)
    return cv2.resize(image_cropped, resize_size)


def preprocess_image(query: np.array,
                     reference: List[np.array],
                     small_img_size: int = 32,
                     new_size: List[int] = [128, 128]) -> Tuple[np.array, List[np.array]]:

    references_shifted = []
    for r in reference:
        reference_resized = crop_resize(r, small_img_size, new_size)
        references_shifted.append(reference_resized)

    query_resized = crop_resize(query.copy(), small_img_size, new_size)

    return query_resized, references_shifted


def gaussian_kernel(length: int = 5, sigma: float = 1.) -> np.array:
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return np.outer(gauss, gauss)


def prepare_features(features: List[torch.Tensor]) -> np.array:
    embeddings = []
    for feature in features:
        m = torch.nn.AvgPool2d(3, 1, 1)
        embeddings.append(m(feature))

    embedding = embeddings[0]
    for i in np.arange(1, len(embeddings)):
        embedding = embedding_concat(embedding, embeddings[i])

    return reshape_embedding(np.array(embedding))


def euclidian_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1-p2)


def mask_image_border(image: np.array, mask_size: int = 2) -> np.array:
    image[:mask_size, :] = 0
    image[-mask_size:, :] = 0
    image[:, :mask_size] = 0
    image[:, -mask_size:] = 0
    return image


def pixelize_binary_image(image: np.array, w: int, h: int) -> np.array:
    """Pixelize the input binary image w.r.t. specified width and height

    @param image: Input image
    @param w: Width of the pixelized image
    @param h: Height of the pixelized image
    @returns: Result of the pixelation

    """

    height, width = image.shape
    height_step = height/h
    width_step = width/w
    pixelized_image = np.ones((h, w), dtype=np.uint8)

    for i in np.arange(0, w):
        for j in np.arange(0, h):
            image_block = image[int(i*height_step):int((i+1)*height_step), int(j*width_step):int((j+1)*width_step)]
            if (image_block == 0).any():
                pixelized_image[i, j] = 0

    return cv2.resize(pixelized_image, (width, height), interpolation=cv2.INTER_NEAREST)


def compare_circles(query_circles: List[List],
                    reference_circles: List[List],
                    distance_threshold: int = 3) -> Tuple[List, List]:
    """Compare the sizes of the detected circles on query and reference image patches

    @param query_circles: List of circles detected on the query image patch
    @param reference_circles: List of circles detected on the reference image patch
    @param distance_threshold: The maximum distance between similar circles
    @returns: Lists of circles on query and reference images which are similar

    """
    query_circles = query_circles[0, :]
    reference_circles = reference_circles[0, :]
    query_indexes = []
    reference_indexes = []

    for query_idx, query_circle in enumerate(query_circles):
        for reference_idx, reference_circle in enumerate(reference_circles):
            if np.abs(query_circle[2]-reference_circle[2]) < distance_threshold:
                distance_between_circles = euclidian_distance(
                    query_circle[0],
                    query_circle[1],
                    reference_circle[0],
                    reference_circle[1])
                largest_radius = max(query_circle[2], reference_circle[2])
                if distance_between_circles < largest_radius/2:
                    query_indexes.append(query_idx)
                    reference_indexes.append(reference_idx)

    query_indexes = np.unique(np.array(query_indexes))
    reference_indexes = np.unique(np.array(reference_indexes))

    query_circles = [query_circles[idx] for idx in query_indexes]
    reference_circles = [reference_circles[idx] for idx in reference_indexes]

    return query_circles, reference_circles


def circles_to_mask(circles: List[List[float]],
                    image_gray: np.array,
                    circle_mask_threshold: int,
                    mask_shape: Tuple[int, int]) -> np.array:
    """Converts the list of detected circles into the image. Each detected circle is drawn white on the black image.
    It creates the binary image

    @param circles: List of circles
    @param image_gray: Input grayscale image of thee patch
    @param circle_mask_threshold: The maximum color value that should be considered as a hole
    @param mask_shape: The shape of the output mask
    @returns: Lists of circles on query and reference images which are similar

    """
    mask = np.zeros(mask_shape, np.uint8)
    for circle in circles:
        empty_mask = np.zeros(mask_shape, dtype=np.uint8)
        circle = np.array(circle, dtype=np.uint8)
        center = (circle[0], circle[1])
        radius = circle[2]

        temporary_mask = cv2.circle(empty_mask, center, radius, (255, 255, 255), -1)
        temporary_mask = cv2.threshold(temporary_mask, 10, 1, cv2.THRESH_BINARY)[1]

        masked_circles = image_gray * temporary_mask
        if np.median(masked_circles[masked_circles != 0]) < circle_mask_threshold:
            mask += temporary_mask

    return mask
