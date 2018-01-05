import numpy as np
import cv2

def imcv2_recolor(im, a = .1):
    # different scale of gaussian filter
    if np.random.binomial(1, .05):
        ksize = np.random.choice([3,5,7])
        im = cv2.GaussianBlur(im, (ksize,ksize), 0)

    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2. - 1.

    # random amplify each channel
    im = im * (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform() * 2 - 1
    im = cv2.pow(im/mx, 1. + up * .5)

    # additive random noise
    sigma = np.random.rand()*0.04
    im += np.random.randn(*im.shape)*sigma

    return np.array(im * 255., np.uint8)

def imcv2_affine_trans(im):
    # Scale and translate
    h, w, c = im.shape
    scalex = np.random.uniform() / 10. + 1.
    scaley = np.random.uniform() / 10. + 1.
    max_offx = (scalex-1.) * w
    max_offy = (scaley-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0,0), fx = scalex, fy = scaley)
    im = im[offy : (offy + h), offx : (offx + w)]
    #flip = np.random.binomial(1, .5)
    flip = 0 # left eye, right eye
    if flip: im = cv2.flip(im, 1)
    return im, [w, h, c], [[scalex,scaley], [offx, offy], flip]
