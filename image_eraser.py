import cv2
import numpy as np


def erase(image, mask, window=(9,9)):

    # We need our mask to contain 0s and 1s.
    mask = mask / 255.
    mask = mask.round().astype(np.uint8)

    image_height, image_width= image.shape[:2]
    confidence = (1 - mask).astype(np.float64)

    finished = float("inf")
    while finished != 0:
        # We need to convert the image to an 8-bit representation to get the LAB output
        # in the correct ranges. Explained here:
        # https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html#color_convert_rgb_lab
        lab_image = cv2.cvtColor(image.astype(np.float32) / 256., cv2.COLOR_BGR2LAB)
        
        # 1. Find the fill front using countours.
        fill_front = compute_fill_front(mask)

        # 2. Find priority.
        priority, confidence = find_priority(image, fill_front, mask, confidence, window)

        # 3. Find the max priority.
        max_priority = np.where(priority == priority.max())
        max_priority = max_priority[0][0], max_priority[1][0]

        # Find the mask and image regions from the bounding window.
        window_half = (window[0]//2-1, window[1]//2-1)
        x_bounds, y_bounds = get_region_boundary(image_height, image_width, max_priority, window_half)
        x1, x2 = x_bounds[0], x_bounds[1]+1
        y1, y2 = y_bounds[0], y_bounds[1]+1
        max_priority_indices = [[x1, x2], [y1, y2]]
        target_image = image[x1:x2, y1:y2]
        target_image_lab = lab_image[x1:x2, y1:y2]
        target_mask = mask[x1:x2, y1:y2]
        target_mask = np.stack(target_mask[:,:,np.newaxis]).repeat(3, axis=2)
        source_mask = 1 - target_mask

        # 4. Find the window x, y locations with max priority.
        max_region = get_region_max_priority(lab_image, mask, target_image_lab, source_mask, max_priority_indices)

        # 5. Update the confidence.
        # This step is important as it 'decays' the confidence value when
        # new pixels are inserted in our image. In the next iteration we
        # recompute the confidence.
        # C(p) = C(ˆp) ∀p ∈ Ψˆp ∩ Ω
        front_points = np.argwhere(target_mask == 1)
        confidence[front_points] = confidence[max_priority]

        # 5. Propagate information back to the original mask and image.
        image, mask = update_mask_and_image(image, mask, max_region, target_image,
                                            target_mask, source_mask, max_priority_indices)

        finished = mask.sum()
        print(f"Remaining pixels to paint: {finished}")

    return image.astype(np.uint8)


def compute_fill_front(mask):
    """
    Fill front are the boundary locations
    around our mask image. This function finds
    contours in the given mask and draws a
    boundary around them.
    """
    if mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    fill_front = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    fill_front = cv2.drawContours(fill_front, contours, -1, (255, 255, 255), thickness=1) / 255.
    return fill_front.astype('uint8')


def get_region_boundary(height, width, point_xy, window):
    """
    Return the x, y coordinates around the given point_xy indices
    """
    x1 = max(0, point_xy[0] - window[0])
    x2 = min(point_xy[0] + window[0], height-1)
    y1 = max(0, point_xy[1] - window[1])
    y2 = min(point_xy[1] + window[1], width-1)
    return (x1, x2), (y1, y2)


def find_priority(image, fill_front, mask, confidence, window, threshold=0.001):
    # 2.1. Find the confidence using:
    # C(p) = ∑q∈Ψp∩(I−Ω) C(q) / |Ψp|
    for front in np.argwhere(fill_front == 1):
        x_bounds, y_bounds = get_region_boundary(image.shape[0], image.shape[1], front, window)
        x1, x2 = x_bounds[0], x_bounds[1]+1
        y1, y2 = y_bounds[0], y_bounds[1]+1
        psi = confidence[x1: x2, y1: y2]
        confidence[front[0], front[1]] = np.sum(psi) / ((x2 - x1) * (y2 - y1))

    # 2.2. Run Sobel edge detector to find the normal along the x- and y-axis.
    data = cv2.Sobel(src=mask.astype(float), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    # Normalize the vector.
    data /= np.linalg.norm(data)
    # This helps nudge the algorithm if it gets stuck.
    data +=  threshold

    # 2.3. Find the priority.
    priority = fill_front * confidence * data
    return priority, confidence


def get_region_max_priority(lab_image, mask, target_image_lab, source_mask, max_indices):
    """
    This function returns the x, y pairs which have the least Euclidean distance between
    the given target image and the source image (this is taken by iterating over the
    entire image).

    Note: This function expects the image to be in CIE LAB color scheme.
    """
    image_height, image_width = lab_image.shape[:2]
    r_height = max_indices[0][1] - max_indices[0][0]
    r_width = max_indices[1][1] - max_indices[1][0]

    min_source_examplar, patch_difference = [], float('inf')
    for x in range(image_height - r_height + 1):
        for y in range(image_width - r_width + 1):
            s_x1, s_x2 = x, x + r_height
            s_y1, s_y2 = y, y + r_width

            # Only process unexplored mask regions, otherwise we get back the original image.
            if mask[s_x1: s_x2, s_y1: s_y2].sum() != 0:
                continue

            target_image = target_image_lab * source_mask
            source_image = lab_image[s_x1: s_x2, s_y1: s_y2] * source_mask

            # Compute the Euclidean distance.
            difference = np.linalg.norm(target_image - source_image)
            if difference < patch_difference:
                patch_difference = difference
                min_source_examplar = [(s_x1, s_x2-1), (s_y1, s_y2-1)]

    return min_source_examplar


def update_mask_and_image(image, mask, best_match, target_image, target_mask, source_mask, max_indices):
    """
    This function simply copies data back to our image and mask based on the found source and max
    priority region.
    """
    sx, sy = best_match
    source_image = image[sx[0]: sx[1]+1, sy[0]:sy[1]+1]

    # Update the image.
    image[max_indices[0][0]: max_indices[0][1],
          max_indices[1][0]: max_indices[1][1]] = source_image * target_mask + target_image * source_mask

    # Update the mask.
    mask[max_indices[0][0]: max_indices[0][1],
          max_indices[1][0]: max_indices[1][1]] = 0 # Fill with black.

    return image, mask
