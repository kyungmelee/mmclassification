from typing import List, Tuple


def overlaps_with_any(
        a: Tuple[int, int, int, int], bboxes: List[Tuple[int, int, int, int]]
) -> bool:
    for b in bboxes:
        if overlaps(a, b):
            return True
    return False


def overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    return w > 0 and h > 0


def contains(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """a contains b"""
    return (
            a[0] <= b[0]
            and a[1] <= b[1]
            and a[0] + a[2] >= b[0] + b[2]
            and a[1] + a[3] >= b[1] + b[3]
    )


def scale_candidate(
        bbox: Tuple[int, int, int, int], img_shape: int, scale_size: int
) -> Tuple[int, int]:
    """
    Move x and y coordinates, so that the size of the candidate is scaled to a selected scale_size.

    :param bbox: (List) x, y, width, height of proposed candidate
    :param img_shape: (int) shape of the input image, the maximum height and width
    :param scale_size: (int) new (target) size of the candidate region
    :return: (int, int) updated x and y coordinates
    """
    x, y, w, h = bbox

    x_new = x - round((scale_size - w) / 2)
    y_new = y - round((scale_size - h) / 2)

    x_new = max(0, x_new)
    y_new = max(0, y_new)

    diff_right = img_shape - (x_new + scale_size)
    diff_bottom = img_shape - (y_new + scale_size)

    if diff_right < 0:
        x_new += diff_right

    if diff_bottom < 0:
        y_new += diff_bottom

    return x_new, y_new


def update_candidate_bbox(
        x: int, y: int, w: int, h: int, img_shape_size: int
) -> Tuple[int, int, int]:
    if w <= 32 and h <= 32:
        cropsize = 32
    elif (w > 32 or h > 32) and (w <= 64 and h <= 64):
        cropsize = 64
    elif (w > 64 or h > 64) and (w <= 128 and h <= 128):
        cropsize = 128
    else:
        cropsize = max(h, w)

    new_x, new_y = scale_candidate((x, y, w, h), img_shape_size, cropsize)

    return new_x, new_y, cropsize


def scale_square(
        bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int], patch_size: int = 128
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox

    # this is required since we use bounding box output as a box of contour
    w -= 1
    h -= 1

    img_h, img_w = img_shape[:2]

    center_x = x + w / 2
    center_y = y + h / 2

    # max out of bounding box w and h and the patch size
    output_patch_size = max(patch_size, w, h)

    new_x = center_x - output_patch_size / 2
    new_y = center_y - output_patch_size / 2

    if new_x + output_patch_size > img_w:
        new_x = img_w - output_patch_size
    elif new_x < 0:
        new_x = 0

    if new_y + output_patch_size > img_h:
        new_y = img_h - output_patch_size
    elif new_y < 0:
        new_y = 0

    return int(new_x), int(new_y), output_patch_size, output_patch_size


def calculate_box_size(bbox: Tuple[int, int, int, int], max_size: int) -> int:
    pixels_number = bbox[2] * bbox[3]
    return min(int(pixels_number / 10) * 10, max_size)


if __name__ == "__main__":
    bbox = (2423, 3070, 7, 2)
    img_shape = (3072, 600)
    patch_size = 100
    print(scale_square(bbox, img_shape, patch_size))
