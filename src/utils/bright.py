import numpy as np
from numpy.typing import NDArray
from utils.color import cvt_RGB2YCbCr, cvt_YCbCr2RGB


def gamma(img: NDArray[np.float64], alpha: float = 1.) -> NDArray[np.float64]:
    img_new = 255. * (img / 255.) ** alpha
    return img_new


def gamma_color(img: NDArray[np.float64], alpha: float = 1.) -> NDArray[np.float64]:
    img_ycc = cvt_RGB2YCbCr(img)

    img_ycc[:, :, 0] = gamma(img_ycc[:, :, 0], alpha)

    img_new = cvt_YCbCr2RGB(img_ycc)

    return img_new
