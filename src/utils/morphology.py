import numpy as np
from numpy.typing import NDArray


def dilation(img: NDArray[np.uint8], kernal: NDArray[np.uint8], iterations: int = 1) -> NDArray[np.uint8]:
    row, col = kernal.shape
    half_r, half_c = row // 2, col // 2

    row_im, col_im = img.shape
    img_copy = np.array(img, dtype=np.uint8)
    result = np.array(img, dtype=np.uint8)

    for i in range(iterations):
        for r in range(row):
            for c in range(col):
                if kernal[r, c]:
                    dr, dc = r - half_r, c - half_c

                    k_rows = max(dr, 0), min(row_im + dr, row_im)
                    k_cols = max(dc, 0), min(col_im + dc, col_im)

                    im_rows = max(-dr, 0), min(row_im - dr, row_im)
                    im_cols = max(-dc, 0), min(col_im - dc, col_im)

                    result[im_rows[0]: im_rows[1], im_cols[0]: im_cols[1]] = np.maximum(
                        img_copy[k_rows[0]: k_rows[1], k_cols[0]: k_cols[1]],
                        result[im_rows[0]: im_rows[1], im_cols[0]: im_cols[1]]
                    )
        img_copy = np.array(result, dtype=np.uint8)
    return result


def erosion(img: NDArray[np.uint8], kernal: NDArray[np.uint8], iterations: int = 1) -> NDArray[np.uint8]:
    row, col = kernal.shape
    half_r, half_c = row // 2, col // 2

    row_im, col_im = img.shape
    img_copy = np.array(img, dtype=np.uint8)
    result = np.array(img, dtype=np.uint8)

    for i in range(iterations):
        for r in range(row):
            for c in range(col):
                if kernal[r, c]:
                    dr, dc = r - half_r, c - half_c

                    k_rows = max(dr, 0), min(row_im + dr, row_im)
                    k_cols = max(dc, 0), min(col_im + dc, col_im)

                    im_rows = max(-dr, 0), min(row_im - dr, row_im)
                    im_cols = max(-dc, 0), min(col_im - dc, col_im)

                    result[im_rows[0]: im_rows[1], im_cols[0]: im_cols[1]] = np.minimum(
                        img_copy[k_rows[0]: k_rows[1], k_cols[0]: k_cols[1]],
                        result[im_rows[0]: im_rows[1], im_cols[0]: im_cols[1]]
                    )
        img_copy = np.array(result, dtype=np.uint8)
    return result
