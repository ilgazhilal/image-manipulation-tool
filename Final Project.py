#!/usr/bin/env python
# coding: utf-8

import numpy as np
import skimage as sk
import skimage.io as skio
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, misc
import math
from functools import reduce
from PIL import Image, ImageMath, ImageChops, ImageOps, ImageEnhance
from PIL.ImageMath import imagemath_convert as _convert
from PIL.ImageMath import imagemath_float as _float
from skimage import io
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
import os


def my_clip(a, m_min=0, m_max=255):
    return min(max(a, m_min), m_max)

#change image contrast according to given quantitiy
def my_contr(img, quantity=1):
    assert quantity >= 0
    return img.point(lambda x: round(x * quantity + -127.5 * quantity + 127.5))

#apply gray filter according to given quantity 
def my_gray(img, quantity=1):
    assert quantity >= 0
    g = 1 - min(quantity, 1)
    matrix = [
        .2126 + .7874 * g, .7152 - .7152 * g, .0722 - .0722 * g, 0,
        .2126 - .2126 * g, .7152 + .2848 * g, .0722 - .0722 * g, 0,
        .2126 - .2126 * g, .7152 - .7152 * g, .0722 + .9278 * g, 0,
    ]
    grayscaled = convImg(img, 'RGB').convert('RGB', matrix)
    return convImg(grayscaled, img.mode)

#change brightness according to given quantity
def my_bright(img, quantity=1):
    assert quantity >= 0
    return img.point(lambda x: round(x * quantity))


def apply_lt(img, lt):
    if len(lt) != 256:
        raise ValueError('A size of LUT must be 256: {}'.format(len(lt)))
    return img.point(lt * len(img.getbands()))

#change saturation according to given quantity
def sat(img, quantity=1):
    assert quantity >= 0
    matrix = [
        .213 + .787 * quantity, .715 - .715 * quantity, .072 - .072 * quantity, 0,
        .213 - .213 * quantity, .715 + .285 * quantity, .072 - .072 * quantity, 0,
        .213 - .213 * quantity, .715 - .715 * quantity, .072 + .928 * quantity, 0,
    ]
    saturated = convImg(img, 'RGB').convert('RGB', matrix)
    return convImg(saturated, img.mode)

#convert imgto given mode
def convImg(img, mode):
    return img if img.mode == mode else img.convert(mode)

#apply alpha blend to given to img and _screen
def screen(img1, img2):
    return alpha_blend(img1, img2, _screen)

#Superimposes two inverted images on top of each other.
def _screen(img1, img2):
    return ImageChops.screen(img1, img2)


def alpha_blend(img1, img2, blending):
    #split images to color channel
    img1, a1 = a_split(img1)
    img2, a2 = a_split(img2)
    #Superimposes two inverted images on top of each other
    im_blended = blending(img1, img2)
    #if img1 is rgb and img2 is rgb
    if a1 is not None and a2 is not None:
        im_blended_alpha = ImageChops.multiply(a1, a2)
        im1_alpha = subt(a1, im_blended_alpha)
        im2_alpha = subt(a2, im_blended_alpha)
        c1 = ImageChops.multiply(convert_A_to_RGB(im2_alpha), img2)
        c2 = ImageChops.multiply(convert_A_to_RGB(im_blended_alpha), im_blended)
        c3 = ImageChops.multiply(convert_A_to_RGB(im1_alpha), img1)
        im_blended = triple_add(c1, c2, c3)
    elif a1 is not None:
        a1_rgb = convert_A_to_RGB(a1)
        a1_invert_rgb = convert_A_to_RGB(inv(a1))
        im_blended = add(
            ImageChops.multiply(a1_rgb, im_blended),
            ImageChops.multiply(a1_invert_rgb, img2))
    elif a2 is not None:
        a2_rgb = convert_A_to_RGB(a2)
        a2_invert_rgb = convert_A_to_RGB(inv(a2))
        im_blended = add(
            ImageChops.multiply(a2_rgb, im_blended),
            ImageChops.multiply(a2_invert_rgb, img1))

    return im_blended

def convert_A_to_RGB(im):
    if im.mode == 'L':
        # NOTE: `merge` is slower than `convert` when using Vanilla Pillow
        im = im.convert('RGB')
        return im
    else:
        raise ValueError('Unsupported mode: ' + im.mode)

#add 3 img 
def triple_add(img1, img2, img3):
    img1 = np.asarray(img1, dtype=np.int16)
    img2 = np.asarray(img2)
    img3 = np.asarray(img3)
    img = img1 + img2 + img3
    img = img.my_clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

#split img to rgb channels
def a_split(img):
    if img.mode == 'RGBA':
        #method is used to split the image into individual bands. This method returns a tuple of individual image bands from an image.
        a = img.split()[3]
        img = img.convert('RGB')
        return img, a
    elif img.mode == 'RGB':
        return img, None
    else:
        raise ValueError('Unsupported mode: ' + img.mode)

#add 2 images
def add(img1, img2):
    # NOTE: When using Vanilla Pillow, `ImageChops.add` is slower than numpy
    img1 = np.asarray(img1, dtype=np.int16)  # avoid overflow
    img2 = np.asarray(img2)
    img= img1 + img2
    img = img.clip(0, 255).astype(np.uint8)

    return Image.fromarray(img)
#return invert an image
def inv(img):
    return Image.fromarray(255 - np.asarray(img))

#substract two images 
def subt(img1, img2):
    img1 = np.asarray(img1, dtype=np.int16)  # avoid underflow
    img2 = np.asarray(img2)
    img = img1 - img2
    img = img.clip(0, 255).astype(np.uint8)

    return Image.fromarray(img)

def fill(size, color):
    assert len(size) == 2
    assert len(color) in [3, 4]

    if len(color) == 4:
        color[3] = int(round(color[3] * 255))  # alpha

    unq = list(set(color))
    map_c = {c: Image.new('L', size, c) for c in unq}

    if len(color) == 3:
        r, g, b = color
        return Image.merge('RGB', (map_c[r], map_c[g], map_c[b]))
    else:
        r, g, b, a = color
        return Image.merge('RGBA', (map_c[r], map_c[g], map_c[b], map_c[a]))

def _drkn(img1, img2):
    return ImageChops.darker(img1, img2)


def drkn(im1, im2):
    return alpha_blend(im1, im2, _drkn)

def _prp_lg_mask(size, start, end, horizontal=True):
    """Returns prepared linear gradient mask."""
    assert end >= 1

    msk = inv(Image.linear_gradient('L'))
    w, h = msk.size
    box = (0, round(h * start), w, round(h / end))
    rs_mask = msk.resize(size, box=box)

    if horizontal:
        return rs_mask.rotate(90)
    else:
        return rs_mask


def lg_mask(size, start=0, end=1, is_horizontal=True):
    assert len(size) == 2
    if end >= 1:
        return _prp_lg_mask(
                size, start, end, is_horizontal)

    w, h = size
    start *= 255
    end *= 255

    if is_horizontal:
        row = np.linspace(start, end, num=w, dtype=np.uint8)
        mask = np.tile(row, (h, 1))
    else:
        row = np.linspace(start, end, num=h, dtype=np.uint8)
        mask = np.tile(row, (w, 1)).T

    return Image.fromarray(mask)


def lg(size, start, end, is_horizontal=True):
    assert len(size) == 2
    assert len(start) == 3
    assert len(end) == 3

    im_start = fill(size, start)
    im_end = fill(size, end)
    mask = lg_mask(size, is_horizontal=is_horizontal)

    return Image.composite(im_start, im_end, mask)

def rot_hue(img, degree=0):
    cos_h = math.cos(math.radians(degree))
    sin_h = math.sin(math.radians(degree))

    matrix = [
        .213 + cos_h * .787 - sin_h * .213,
        .715 - cos_h * .715 - sin_h * .715,
        .072 - cos_h * .072 + sin_h * .928,
        0,
        .213 - cos_h * .213 + sin_h * .143,
        .715 + cos_h * .285 + sin_h * .140,
        .072 - cos_h * .072 - sin_h * .283,
        0,
        .213 - cos_h * .213 - sin_h * .787,
        .715 - cos_h * .715 + sin_h * .715,
        .072 + cos_h * .928 + sin_h * .072,
        0,
    ]
    rotated = convImg(img, 'RGB').convert('RGB', matrix)
    return convImg(rotated, img.mode)

def _lgtn(img1, img2):
    return ImageChops.lighter(img1, img2)


def lgtn(img1, img2):
    return alpha_blend(img1, img2, _lgtn)

def sepia(img, quantity=1):
    assert quantity >= 0

    quantity = 1 - min(quantity, 1)
    matrix = [
        .393 + .607 * quantity, .769 - .769 * quantity, .189 - .189 * quantity, 0,
        .349 - .349 * quantity, .686 + .314 * quantity, .168 - .168 * quantity, 0,
        .272 - .272 * quantity, .534 - .534 * quantity, .131 + .869 * quantity, 0,
    ]

    sepia_t = convImg(img, 'RGB').convert('RGB', matrix)
    return convImg(sepia_t, img.mode)

def overlay(img1, img2):
    return hrdl(img2, img1)

LUT_2x = [my_clip(2 * i) for i in range(256)]
LUT_2x_1 = [my_clip(2 * i - 255) for i in range(256)]

def _hrdl(img1, img2):
    im2_multiply = apply_lt(img2, LUT_2x)
    multiply = np.asarray(ImageChops.multiply(img1, im2_multiply))

    im2_screen = apply_lt(img2, LUT_2x_1)
    screen = np.asarray(ImageChops.screen(img1, im2_screen))

    cm = np.where(np.asarray(img2) < 128, multiply, screen)
    return Image.fromarray(cm)


def hrdl(im1, im2):
    return alpha_blend(im1, im2, _hrdl)


def rg_mask(size, length=0, scale=1, center=(.5, .5)):
    if length >= 1:
        return Image.new('L', size, 255)

    if scale <= 0:
        return Image.new('L', size, 0)

    w, h = size
    cx, cy = center

    rw_left = w * cx
    rw_right = w * (1 - cx)
    rh_top = h * cy
    rh_bottom = h * (1 - cy)

    x = np.linspace(-rw_left, rw_right, w)
    y = np.linspace(-rh_top, rh_bottom, h)[:, None]

    # farhest corner radius is r
    r = math.sqrt(max(rw_left, rw_right) * 2 + max(rh_top, rh_bottom) * 2)
    # for divide by 0 error
    base = max(scale - length, 0.001)

    # calculate distance from center
    msk = np.sqrt(x * 2 + y * 2) / r
    msk = (msk - length) / base
    msk = 1 - msk
    msk *= 255
    msk = msk.clip(0, 255)

    return Image.fromarray(np.uint8(msk.round()))





"""
def rg_mask(size, length=0, scale=1, center=(.5, .5)):
    if length >= 1:
        return Image.new('L', size, 255)

    if scale <= 0:
        return Image.new('L', size, 0)

    w, h = size
    cx, cy = center

    # use faster method if possible
    if length == 0 and scale >= 1 and w == h and center == (.5, .5):
        return _prepared_radial_gradient_mask(size, scale)

    rw_left = w * cx
    rw_right = w * (1 - cx)
    rh_top = h * cy
    rh_bottom = h * (1 - cy)

    x = np.linspace(-rw_left, rw_right, w)
    y = np.linspace(-rh_top, rh_bottom, h)[:, None]

    # r is a radius to the farthest-corner
    r = math.sqrt(max(rw_left, rw_right) ** 2 + max(rh_top, rh_bottom) ** 2)
    base = max(scale - length, 0.001)  # avoid a division by zero

    msk = np.sqrt(x ** 2 + y ** 2) / r  # distance from center
    msk = (msk - length) / base  # adjust ending shape
    msk = 1 - msk  # inv: distance to center
    msk *= 255
    msk = msk.clip(0, 255)

    return Image.fromarray(np.uint8(msk.round()))

"""
def rg(size, colors, positions=None, **kwargs):
    assert len(size) == 2
    assert len(colors) >= 2
    for color in colors:
        assert len(color) == 3

    if positions is None:
        positions = np.linspace(0, 1, len(colors))
    else:
        assert len(positions) >= 2
        assert len(colors) == len(positions)

    colors = [fill(size, color) for color in colors]

    def compose(x, y):
        kwargs_ = kwargs.copy()
        kwargs_['length'] = x[1]
        kwargs_['scale'] = y[1]
        mask = rg_mask(size, **kwargs_)
        return (Image.composite(x[0], y[0], mask), y[1])

    return reduce(compose, zip(colors, positions))[0]

def _d_cb(cb):
    """Returns D(Cb) - Cb"""
    cb = float(cb) / 255

    if cb <= .25:
        d = ((16 * cb - 12) * cb + 4) * cb
    else:
        d = math.sqrt(cb)

    return round((d - cb) * 255)


LUT_1_2_x_cs = [my_clip(255 - 2 * i) for i in range(256)]
LUT_cb_x_1_cb = [round(my_clip(i * (1 - i / 255))) for i in range(256)]
LUT_2_x_cs_1 = [my_clip(2 * i - 255) for i in range(256)]
LUT_d_cb = [_d_cb(i) for i in range(256)]


def _sfl(img1, img2):
    _1_2_x_cs = apply_lt(img2, LUT_1_2_x_cs)
    cb_x_1_cb = apply_lt(img1, LUT_cb_x_1_cb)
    c1 = subt(img1, ImageChops.multiply(_1_2_x_cs, cb_x_1_cb))

    _2_x_cs_1 = apply_lt(img2, LUT_2_x_cs_1)
    d_cb = apply_lt(img1, LUT_d_cb)
    c2 = add(img1, ImageChops.multiply(_2_x_cs_1, d_cb))

    cm = np.where(np.asarray(img2) <= 128, np.asarray(c1), np.asarray(c2))
    return Image.fromarray(cm)


def sfl(img1, img2):
    return alpha_blend(img1, img2, _sfl)

def _1977(img):
    cb = convImg(img, 'RGB')

    cs = fill(cb.size, [243, 106, 188, .3])
    cr = screen(cb, cs)

    cr = my_contr(cr, 1.1)
    cr = my_bright(cr, 1.1)
    cr = sat(cr, 1.3)

    return cr


def _color_dodge_image_math(cb, cs_inv):
    """Returns ImageMath operands for color dodge blend mode"""
    cb = _float(cb)
    cs_inv = _float(cs_inv)

    cm = ((cb != 0) * (cs_inv == 0) + (cb / cs_inv)) * 255
    return _convert(cm, 'L')


def _color_dodge(im1, im2):
    return Image.merge('RGB', [
        ImageMath.eval(
            'f(cb, cs_inv)', f=_color_dodge_image_math, cb=cb, cs_inv=cs_inv)
        for cb, cs_inv in zip(im1.split(), inv(im2).split())
    ])


def color_dodge(im1, im2):
    return alpha_blend(im1, im2, _color_dodge)


def aden(im):
    cb = convImg(im, 'RGB')

    cs = fill(cb.size, [66, 10, 14])
    cs = drkn(cb, cs)

    alpha_mask = lg_mask(cb.size, start=.8)
    cr = Image.composite(cs, cb, alpha_mask)

    cr = rot_hue(cr, -20)
    cr = my_contr(cr, .9)
    cr = sat(cr, .85)
    cr = my_bright(cr, 1.2)

    return cr

def brannan(im):

    cb = convImg(im, 'RGB')
    cs = fill(cb.size, [161, 44, 199, .31])
    cr =lgtn(cb, cs)

    cr = sepia(cr, .5)
    cr = my_contr(cr, 1.4)

    return cr

def brooklyn(im):
    cb = convImg(im, 'RGB')

    cs1 = fill(cb.size, [168, 223, 193, .4])
    cm1 = overlay(cb, cs1)

    cs2 = fill(cb.size, [196, 183, 200])
    cm2 = overlay(cb, cs2)

    gradient_mask = rg_mask(cb.size, length=.7)
    cr = Image.composite(cm1, cm2, gradient_mask)

    cr = my_contr(cr, .9)
    cr = my_bright(cr, 1.1)

    return cr

def clarendon(im):
    cb = convImg(im, 'RGB')

    cs = fill(cb.size, [127, 187, 227, .2])
    cr = overlay(cb, cs)

    cr = my_contr(cr, 1.2)
    cr = sat(cr, 1.35)

    return cr

def earlybird(im):
    cb = convImg(im, 'RGB')

    cs = rg(
            cb.size,
            [(208, 186, 142), (54, 3, 9), (29, 2, 16)],
            [.2, .85, 1])
    cr = overlay(cb, cs)

    cr = my_contr(cr, .9)
    cr = sepia(cr, .2)

    return cr


def hudson(im):
    cb = convImg(im, 'RGB')

    cs = rg(
            cb.size,
            [(166, 177, 255), (52, 33, 52)],
            [.5, 1])
    cs = multiply(cb, cs)
    cr = Image.blend(cb, cs, .5)  # opacity

    cr = my_bright(cr, 1.2)
    cr = my_contr(cr, .9)
    cr = saturate(cr, 1.1)

    return cr

def inkwell(im):
    #pil imageyi rgb ye dönüştürüyor
    cb = convImg(im, 'RGB')
    #sepia fonksiyonu verilen resim ve katsayıya göre içindeki filtreyi uyguluyor
    cr = sepia(cb, .3)
    #imagenin contrast değerini değiştiriyor
    cr = my_contr(cr, 1.1)
    #parlaklığını değiştiriyor
    cr = my_bright(cr, 1.1)
    #my_gray içindeki filtreyi uyguluyor
    cr = my_gray(cr)
    #sonucu döndürüyor
    return cr


def lark(im):
    #pill image to rgb
    cb = convImg(im, 'RGB')
    #image size
    cs1 = fill(cb.size, [34, 37, 63])
    
    cm1 = color_dodge(cb, cs1)

    cs2 = fill(cb.size, [242, 242, 242, .8])
    cr = drkn(cm1, cs2)

    cr = my_contr(cr, .9)

    return cr



#HİLAL-FUNCTIONS

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)

def colorfuledges(img): 
    img=convert_from_image_to_cv2(img)
    image = cv2.bilateralFilter(img, 50, 10, 10) 
    blackboard_kernel = np.array([[1,-1,0], [-1,4,-1], [-1,0,-1]])
    result = cv2.filter2D(image, -1, blackboard_kernel)
    result=convert_from_cv2_to_image(result)
    return result

def pencilsketch(img):
    img=convert_from_image_to_cv2(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bit_image = cv2.bitwise_not(gray_image)
    smoothed = cv2.GaussianBlur(bit_image, (21, 21),sigmaX=0, sigmaY=0)
    final= cv2.divide(gray_image, 255 - smoothed, scale=256)
    final=convert_from_cv2_to_image(final)
    return final

def cartoon(img):
    img=convert_from_image_to_cv2(img)
    bilateral=50
    d=9
    sigmacolor=9
    sigmaspace=7
    rgb=img
    rgb = cv2.pyrDown(rgb)
    rgb = cv2.pyrDown(rgb)
    for _ in range(bilateral):  
        rgb = cv2.bilateralFilter(rgb, d, sigmacolor, sigmaspace)  
    rgb = cv2.pyrUp(rgb)
    rgb = cv2.pyrUp(rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    blurred = cv2.medianBlur(gray, 3)     
    edges = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 9, 2)     
    a,b,c = rgb.shape  
    edges = cv2.resize(edges,(b,a))  
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    resault=cv2.bitwise_and(rgb, edges) 
    resault=convert_from_cv2_to_image(resault)
    return resault
def sketchplusdetail(img):
    img=convert_from_image_to_cv2(img)
    detEn=cv2.detailEnhance(img, sigma_s=60, sigma_r=0.2)
    black_image, color_image  = cv2.pencilSketch(detEn, sigma_s=120, sigma_r=0.07, shade_factor=0.03)
    black_image=convert_from_cv2_to_image(black_image)
    return black_image  
def sketchplusdetailcolor(img):
    img=convert_from_image_to_cv2(img)
    detEn=cv2.detailEnhance(img, sigma_s=60, sigma_r=0.2)
    black_image, color_image  = cv2.pencilSketch(detEn, sigma_s=120, sigma_r=0.07, shade_factor=0.03)
    color_image=convert_from_cv2_to_image(color_image)
    return color_image
def waterdetail(img):
    img=convert_from_image_to_cv2(img)
    detEn=cv2.detailEnhance(img, sigma_s=60, sigma_r=0.2)
    watercolor = cv2.stylization(detEn, sigma_s=120, sigma_r=0.8)
    watercolor=convert_from_cv2_to_image(watercolor)
    return watercolor
def bilateralcartoon(img):
    img=convert_from_image_to_cv2(img)
    blur = cv2.bilateralFilter(img,9,75,75)
    cartoon=cv2.stylization(blur, sigma_s=100, sigma_r=0.15)
    cartoon=convert_from_cv2_to_image(cartoon)
    return cartoon   
def edgepresevepluswater(img):
    img=convert_from_image_to_cv2(img)
    epf = cv2.edgePreservingFilter(img, flags=2, sigma_s=120, sigma_r=0.5)
    watercolor = cv2.stylization(epf, sigma_s=120, sigma_r=0.8)
    watercolor=convert_from_cv2_to_image(watercolor)
    return watercolor
def vintageFilter(img): #focusing center
    img=convert_from_image_to_cv2(img)
    rows,cols = img.shape[:2]
    empty = np.copy(img)
    empty[:,:,:] = 0
    col_gauss = cv2.getGaussianKernel(cols,300)
    row_gauss = cv2.getGaussianKernel(rows,300)
    transpose = row_gauss*col_gauss.T
    val = transpose/transpose.max()
    empty[:,:,0] = img[:,:,0]*val
    empty[:,:,1] = img[:,:,1]*val
    empty[:,:,2] = img[:,:,2]*val
    empty=convert_from_cv2_to_image(empty)
    return empty

def emboss(img):
    img=convert_from_image_to_cv2(img) 
    emboss_kernel=np.array(([[-2,-1,0],[-1,1,1],[0,1,2]]),np.float32)
    embossfilter=cv2.filter2D(src=img,kernel=emboss_kernel,ddepth=-1)
    img=convert_from_cv2_to_image(embossfilter)
    return img

def outline(img):
    img=convert_from_image_to_cv2(img)
    outline_kernel=np.array(([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),np.float32)
    outlinefilter=cv2.filter2D(src=img,kernel=outline_kernel,ddepth=-1)
    outlinefilter=convert_from_cv2_to_image(outlinefilter)
    return outlinefilter

def sharpening(img):
    img=convert_from_image_to_cv2(img)
    sharpen_kernel=np.array(([[0,-1,0],[-1,5,-1],[0,-1,0]]),np.float32)
    sharpenfilter=cv2.filter2D(src=img,kernel=sharpen_kernel,ddepth=-1)
    sharpenfilter=convert_from_cv2_to_image(sharpenfilter)
    return sharpenfilter

def custom1(img):
    img=convert_from_image_to_cv2(img)
    custom1_kernel=np.array(([[0,-1,0],[-1,4,-1],[0,-1,0]]),np.float32)
    custom1filter=cv2.filter2D(src=img,kernel=custom1_kernel,ddepth=-1)
    sharpen_kernel=np.array(([[0,-1,0],[-1,5,-1],[0,-1,0]]),np.float32)
    sharpenfilter=cv2.filter2D(src=custom1filter,kernel=sharpen_kernel,ddepth=-1)
    sharpenfilter=convert_from_cv2_to_image(sharpenfilter)
    return sharpenfilter

def custom2(img):
    img=convert_from_image_to_cv2(img)
    emboss_kernel=np.array(([[-2,-1,0],[-1,1,1],[0,1,2]]),np.float32)
    embossfilter=cv2.filter2D(src=img,kernel=emboss_kernel,ddepth=-1)
    oilpaint = cv2.xphoto.oilPainting(embossfilter, 10, 3)
    oilpaint=convert_from_cv2_to_image(oilpaint)
    return oilpaint

def custom3(img):
    img=convert_from_image_to_cv2(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpen_kernel=np.array(([[0,-1,0],[-1,5,-1],[0,-1,0]]),np.float32)
    sharpenfilter=cv2.filter2D(src=gray,kernel=sharpen_kernel,ddepth=-1)
    gaussianblur_kernel=np.array(([[1,2,1],[2,4,2],[1,2,1]]),np.float32)
    gaussianfilter=cv2.filter2D(src=sharpenfilter,kernel=gaussianblur_kernel,ddepth=-1)
    final=sharpenfilter-gaussianfilter
    final=convert_from_cv2_to_image(final)
    return final

def custom4(img):
    img=convert_from_image_to_cv2(img)
    sharpen_kernel=np.array(([[0,-1,0],[-1,5,-1],[0,-1,0]]),np.float32)
    sharpenfilter=cv2.filter2D(src=img,kernel=sharpen_kernel,ddepth=-1)
    gaussianblur_kernel=np.array(([[1,2,1],[2,4,2],[1,2,1]]),np.float32)
    gaussianfilter=cv2.filter2D(src=sharpenfilter,kernel=gaussianblur_kernel,ddepth=-1)
    final=sharpenfilter-gaussianfilter
    final=convert_from_cv2_to_image(final)
    return final

def color_saturation(img, a = 1, b = 1):
    img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = hsv[:,:,0]
    saturation = hsv[:,:,1]
    pantone=179
    white=255
    black=0
    color = np.clip(color * a ,black,pantone)
    saturation = np.clip(saturation * b,black,white)
    hsv[:,:,0] = color
    hsv[:,:,1] = saturation
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

def brightness_setting(img, a = 1.0, b = 0):
    img = img.copy()
    contrast = img * (a)
    bright = contrast + (b)
    bright = np.clip(bright,0,255)
    bright = bright.astype(np.uint8)
    return bright

def color_setting(img,huel=0,satl=0,val=0,hue=0,sat=0,v=0,red=0,green=0,blue=0):
    img = img.copy()
    row, col = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_val = np.array([huel,satl,hue])
    upper_val = np.array([hue,sat,v])
    color = cv2.inRange(hsv, lower_val, upper_val) 
    img[color>0]=(blue,green,red)
    return img

def color_weighting(img,red,green,blue,alpha):
    img = img.copy()
    color = img.copy()
    color[:,:,0] = blue
    color[:,:,1] = green
    color[:,:,2] = red
    result = cv2.addWeighted(img,1-alpha,color,alpha,0)
    return result

def toaster(img, color = 1, saturation = 0.9, contrast = 1.4, brightness = -20):
    img=convert_from_image_to_cv2(img)
    img = color_setting(img,v=128,red=51)
    img = color_setting(img,huel=150,satl=255,val=50,hue=170,sat=255,v=128,red=51)
    img = color_saturation(img, color, saturation)
    img = brightness_setting(img, contrast, brightness)
    img = color_weighting(img,255,99,66,0.1)
    img = color_weighting(img,250,250,0,0.3)
    img=convert_from_cv2_to_image(img)
    return img

def kelvin(img, color = 1.2, saturation = 1.5, contrast = 1.3, brightness = 10):
    img=convert_from_image_to_cv2(img)
    img = color_saturation(img, color, saturation)
    img = brightness_setting(img, contrast, brightness)
    img = color_weighting(img,240,240,0,0.1)
    img=convert_from_cv2_to_image(img)
    return img

def gingham(img, color = 1.1, saturation = 0.9, contrast = 1.1, brightness = -20):
    img=convert_from_image_to_cv2(img)
    img = color_saturation(img, color, saturation)
    img = brightness_setting(img, contrast, brightness)
    img=convert_from_cv2_to_image(img)
    return img

def amaro(img, color = 1.1, saturation = 1.5, contrast = 0.9, brightness = 10):
    img=convert_from_image_to_cv2(img)
    img = color_saturation(img, color, saturation)
    img = brightness_setting(img, contrast, brightness)
    img=convert_from_cv2_to_image(img)
    return img
def mirror(image):
    imageMirrored = ImageOps.flip(image)
    return imageMirrored


# BASIC OPERATIONS

def rotate(image, angle, direction="left"):
    if direction == "right":
        imageRotated = image.rotate(-1 * angle)
    else:
        imageRotated = image.rotate(angle)
    return imageRotated

def flip(image, axis):
    if (axis == "horizontal"):
        imageFlipped = image.transpose(Image.FLIP_TOP_BOTTOM)
    elif (axis == "vertical"):
        imageFlipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        imageFlipped = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    return imageFlipped

def mirror(image):
    imageMirrored = ImageOps.flip(image)
    return imageMirrored

def my_crop(image,topLeftX, topLeftY, bottomRightX, bottomRightY):
    width, height = image.size

    if ((bottomRightX < topLeftX) or (bottomRightY < topLeftY)):
        messagebox.showinfo(title='Warning!',message="bottomRightX degeri topLeftX degerinden büyük olmalıdır.")
        messagebox.showinfo(title='Warning!',message="bottomRightY degeri topLeftY degerinden büyük olmalıdır")

    #Editing for topLeftX
    if (topLeftX < 0):
        topLeftX = 0
    elif (topLeftX > width):
        topLeftX = width
    else:
        topLeftX = topLeftX

    # Editing for topLeftY
    if (topLeftY < 0):
        topLeftY = 0
    elif (topLeftY > height):
        topLeftY = height
    else:
        topLeftY = topLeftY

    #Editing for bottomRightX
    if (bottomRightX < 0):
        bottomRightX = 0
    elif (bottomRightX > width):
        bottomRightX = width
    else:
        bottomRightX = bottomRightX

    # Editing for bottomRightY
    if (bottomRightY < 0):
        bottomRightY = 0
    elif (bottomRightY > height):
        bottomRightY = height
    else:
        bottomRightY = bottomRightY

    area = (topLeftX, topLeftY, bottomRightX, bottomRightY)
    imageCropped = image.crop(area)
    return imageCropped

def enhanceBrightness(image, factor):
    window = tk.Tk()
    window.title("Came with Alert window")
    window.withdraw()

    if ((factor < 0) or (factor > 100)):
        messagebox.showwarning("Uyarı", "Please enter a value between 0-100")
        window.destroy()
        return image
    else:
        factorEndPoints = 5.00 - 0.0
        interval = factorEndPoints / 100
        finalFactor = factor * interval + 0.0
        enhancer = ImageEnhance.Brightness(image)
        imageBrighted = enhancer.enhance(finalFactor)
        window.destroy()
        return imageBrighted


def enhanceContrast(image, factor):
    window = tk.Tk()
    window.title("Came with Alert window")
    window.withdraw()

    if ((factor < 0) or (factor > 100)):
        messagebox.showwarning("Uyarı", "Please enter a value between 0-100")
        window.destroy()
        return image
    else:
        factorEndPoints = 4.00 - (-2.00)
        interval = factorEndPoints / 100
        finalFactor = factor * interval + (-2.00)
        enhancer = ImageEnhance.Contrast(image)
        imageConstrasted = enhancer.enhance(finalFactor)
        window.destroy()
        return imageConstrasted

def invert(image):
    pxl = image.load()

    for i in range(image.size[0]):
        for j in range(image.size[1]):
            r, g, b = pxl[i, j][0], pxl[i, j][1], pxl[i, j][2]
            r, g, b = abs(r - 255), abs(g - 255), abs(b - 255)
            pxl[i, j] = (r, g, b)
    return image

def histogramEqualization(image):
    bgr = np.array(image)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
  
    img_yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    final=Image.fromarray(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))
    return final

def clahe(image):
    bgr = np.array(image)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    final=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return final

#image_rotate = rotate(image, 405, "right")
#image_rotate.show()

#image_flip = flip(image, "vertical")
#image_flip.show()

#image_mirror = mirror(image)
#image_mirror.show()

#image_crop = crop(image, 100, 100, 300, 300)
#image_crop.show()

#------------------------------------------------------
#####   def enhanceBrightness(image, factor):
#####   Factor 0-100 arasında çalışır.
#####   0 karanlık görüntüdür. 20 Normal görüntü.
#####   100 ise tanımlanan en parlak görüntüdür.

##### For more darker
#image_enhanceBrightness = enhanceBrightness(image, 15)
#image_enhanceBrightness.show()

##### For more brighter
#image_enhanceBrightness = enhanceBrightness(image, 40)
#image_enhanceBrightness.show()
#------------------------------------------------------


#------------------------------------------------------
#####   def enhanceContrast(image, factor):
#####   Factor 0-100 arasında çalışır.
#####   0 karanlık negatif uç kontrast. 50 Normal görüntü.
#####   100 ise tanımlanan en yüksek kontsat değeridir.

##### For more contrast
#image_enhanceContrast = enhanceContrast(image, 70)
#image_enhanceContrast.show()

##### For less contrast
#image_enhanceContrast = enhanceContrast(image, 30)
#image_enhanceContrast.show()
#------------------------------------------------------


#image_invert = invert(image)
#image_invert.show()


#imagePath = "birImageYoluVer.jpg"
#image_histogramEqualization = histogramEqualization(imagePath)
#cv2.imshow("hist equalization", image_histogramEqualization)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#imagePath = "birImageYoluVer.jpg"
#image_clahe = clahe(imagePath)
#cv2.imshow("CLAHE", image_clahe)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


form=tk.Tk()
form.geometry("600x500+600+300")
form.minsize(400,300)
form.resizable(False, False)
#form.maxsize(1000,600)
#form.resizable(False,False)
form.state('normal')
#form.state('zoomed')
#form.state('iconic')
#form.wm_attributes('-alpha',0.5)
form.title("Image Processing Final Project")

img=None
img_path=None
record_count=1
#img=Image.open("city.jpg")

def fileName():
    #buraya data type kontrolü ekle jpg, png vs...
    path = filedialog.askopenfilename(title='Open an image', filetypes=[("All İmage Types", "*.jpg"),
                                                                         ("All İmage Types", "*.jpeg"),
                                                                         ("All İmage Types", "*.png"),
                                                                         ("JPG Image", "*.jpg"),
                                                                         ("JPEG Image", "*.jpeg"),
                                                                         ("PNG Image", "*.png")])

    return path

def openImg():
    global img, img_path, record_count
    img_path = fileName()
    if img_path!='':
        record_count = 1
        img=Image.open(img_path)
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="Please choose a image")

def reset():
    global img_path, img
    if img_path!='':
        img=Image.open(img_path)
        showImg(img)
        tbContrast.set(50.0)
        tbBrightness.set(20.0)
    else:
        messagebox.showinfo(title='Warning!',message="Please choose a image")


def save():
    global img, img_path, record_count

    if img_path!='':
        file, ext = os.path.splitext(img_path)
        newPath = file + "_" + str(record_count) + ext
        img.save(newPath)
        record_count += 1
        justFileName = os.path.basename(newPath)
        messagebox.showinfo(title='Saved!',message="Image saved as '"+ justFileName +"'"+"\nPath: "+newPath)
    else:
        messagebox.showinfo(title='Warning!',message="Please choose a image!")

def showImg(img):
    imgTS=img.resize((400,250), Image.ANTIALIAS)
    imgTS=ImageTk.PhotoImage(imgTS)
    panel =tk.Label(form, image=imgTS)
    panel.image=imgTS
    panel.place(x=120,y=120)

def printImgName():
    if img != None:
        print(img.size)  


"""def apply_1977():
    if img!=None:
        imgT=_1977(img)
        showImg(imgT)
    else:
        messagebox.showinfo(title='Warning!',message="please open a image")

def applyAden():
    if img!=None:
        imgT=aden(img)
        showImg(imgT)
    else:
        messagebox.showinfo(title='Warning!',message="please open a image")

def applyBrannan():
    if img!=None:
        imgT=brannan(img)
        showImg(imgT)
    else:
        messagebox.showinfo(title='Warning!',message="please open a image")


def applyBrooklyn():
    if img!=None:
        imgT=brooklyn(img)
        showImg(imgT)
    else:
        messagebox.showinfo(title='Warning!',message="please open a image")"""

def applySelections():
    if img!=None:
        if cmb1.get() == "_1977":
            imgT=_1977(img)
            showImg(imgT)
        elif cmb1.get() == "aden":
            imgT=aden(img)
            showImg(imgT)
        elif cmb1.get() == "brannan":
            imgT=brannan(img)
            showImg(imgT)
        elif cmb1.get() == "brooklyn":
            imgT=brooklyn(img)
            showImg(imgT)
        elif cmb1.get() == "clarendon":
            imgT=clarendon(img)
            showImg(imgT)
        elif cmb1.get() == "earlybird":
            imgT=earlybird(img)
            showImg(imgT)
        elif cmb1.get() == "gingham":
            imgT=gingham(img)
            showImg(imgT)
        elif cmb1.get() == "inkwell":
            imgT=inkwell(img)
            showImg(imgT)
        elif cmb1.get() == "kelvin":
            imgT=kelvin(img)
            showImg(imgT)
        elif cmb1.get() == "lark":
            imgT=lark(img)
            showImg(imgT)
        elif cmb1.get() == "casper":
            imgT=edgepresevepluswater(img)
            showImg(imgT)
        elif cmb1.get() == "spongebob":
            imgT=waterdetail(img)
            showImg(imgT)
        elif cmb1.get() == "emboss":
            imgT=emboss(img)
            showImg(imgT)
        elif cmb1.get() == "outline":
            imgT=outline(img)
            showImg(imgT)
        elif cmb1.get() == "sharpening":
            imgT=sharpening(img)
            showImg(imgT)
        elif cmb1.get() == "beginnerLuck":
            imgT=custom1(img)
            showImg(imgT)
        elif cmb1.get() == "shmoo":
            imgT=custom2(img)
            showImg(imgT)
        elif cmb1.get() == "corpsebride":
            imgT=custom3(img)
            showImg(imgT)
        elif cmb1.get() == "pencilsketch":
            imgT=pencilsketch(img)
            showImg(imgT)
        elif cmb1.get() == "cartoon":
            imgT=cartoon(img)
            showImg(imgT)
        elif cmb1.get() == "toaster":
            imgT=toaster(img)
            showImg(imgT)
        elif cmb1.get() == "vintageFilter":
            imgT=vintageFilter(img)
            showImg(imgT)
        elif cmb1.get() == "carbile":
            imgT=bilateralcartoon(img)
            showImg(imgT)
        elif cmb1.get() == "amaro":
            imgT=amaro(img)
            showImg(imgT)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")


def rotate_left():
    global img
    if img!=None: 
        img=rotate(img, 90)
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")

def rotate_right():
    global img
    if img!=None:
        img=rotate(img, 90, "right")
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="please load a image")

def hide_button(widget): 
    # This will remove the widget from toplevel 
    widget.place_forget() 

def hide_all():
    #rotate
    hide_button(button_rl)
    hide_button(button_rr)
    #flip
    hide_button(button_fh)
    hide_button(button_fv)
    hide_button(button_fo)
    #mirror
    hide_button(button_mr)
    #invert
    hide_button(button_inv)
    #contrast
    hide_button(tbContrast)
    hide_button(lblAlpha)
    #brightness
    hide_button(tbBrightness)
    hide_button(lblBrightness)
    #histogram eq
    hide_button(button_he)
    #clahe
    hide_button(button_cl)
    #crop
    hide_button(lbl_crop1)
    hide_button(input1)
    hide_button(lbl_crop2)
    hide_button(input2)
    hide_button(lbl_crop3)
    hide_button(input3)
    hide_button(lbl_crop4)
    hide_button(input4)
    hide_button(lbl_crop5)
    hide_button(button_crop)

# Method to make Button(widget) visible 
def show_button(widget,X,Y): 
    # This will recover the widget from toplevel 
    widget.place(x=X,y=Y)

def rotate_btn():
    hide_all()
    show_button(button_rl,180,400)
    show_button(button_rr,320,400)
    
def flip_horizantal():
    global img
    if img!=None:
        img=flip(img,"horizontal")
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")
def flip_vertical():
    global img
    if img!=None:
        img=flip(img,"vertical")
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")
def flip_other():
    global img
    if img!=None:
        img=flip(img,"bos")
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")
        
def flip_btn():
    hide_all()
    show_button(button_fh,120,400)
    show_button(button_fv,260,400)
    show_button(button_fo,400,400)
    
def mirror_btn():
    hide_all()
    show_button(button_mr,250,400)
    
def apply_mirror():
    global img
    if img!=None:
        img=mirror(img)
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")

def invert_btn():
    hide_all()
    show_button(button_inv,250,400)
    
def aplly_invert():
    global img
    if img!=None:
        img=invert(img)
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")

def contrast_btn():
    hide_all()
    global img
    if img!=None:
        show_button(tbContrast,180,400)
        show_button(lblAlpha,280,400)
    else:
         messagebox.showinfo(title='Warning!',message="Please load a image")
    
def tbContrast_ValueChangedEvent(alpha):
    global img
    if img!=None:
        lblAlpha.config(text='Default Value : ' + '50')
        f_alpha = float(alpha)
        imgT=enhanceContrast(img,f_alpha)
        showImg(imgT)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")
    pass

def brightness_btn():
    hide_all()
    global img
    if img!=None:
        show_button(tbBrightness,180,400)
        show_button(lblBrightness,280,400)
    else:
         messagebox.showinfo(title='Warning!',message="Please load a image")
    
def tbBrightness_ValueChangedEvent(b):
    global img
    if img!=None:
        lblBrightness.config(text='Default Value : ' + '20')
        f_b = float(b)
        imgT=enhanceBrightness(img,f_b)
        showImg(imgT)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")
    pass

def histogram_btn():
    hide_all()
    show_button(button_he,250,400)
    
def aplly_histogram():
    global img, img_path
    if (img!=None and img_path!=None):
        img=histogramEqualization(img)
        showImg(img)
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")

def clahe_btn():
    hide_all()
    show_button(button_cl,250,400)
    
def aplly_clahe():
    global img, img_path
    if (img!=None and img_path!=None):
        img=clahe(img)
        showImg(img)
       
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")
        
def crop_btn():
    hide_all()
    global img
    if img!=None:
        show_button(lbl_crop1,120,420)
        show_button(input1,220,420)
        show_button(lbl_crop2,220,390)
        show_button(input2,280,420)
        show_button(lbl_crop3,280,390)
        show_button(input3,340,420)
        show_button(lbl_crop4,340,390)
        show_button(input4,400,420)
        show_button(lbl_crop5,400,390)
        show_button(button_crop,460,420)
        
    else:
         messagebox.showinfo(title='Warning!',message="Please load a image")

def aplly_crop():
    global img
    if (img!=None):
        e1_val=e1.get()
        e2_val=e2.get()
        e3_val=e3.get()
        e4_val=e4.get()
        img=my_crop(img,e1_val,e2_val,e3_val,e4_val)
        showImg(img)
       
    else:
        messagebox.showinfo(title='Warning!',message="Please load a image")
        
#Rotate
button_rl=tk.Button(form,text="Left",fg="white",bg="black",width=15,height=1,command=lambda: rotate_left())
button_rr=tk.Button(form,text="Right",fg="white",bg="black",width=15,height=1,command=lambda: rotate_right())
#Flip
button_fh=tk.Button(form,text="Horizantal",fg="white",bg="black",width=15,height=1,command=lambda: flip_horizantal())
button_fv=tk.Button(form,text="Vertical",fg="white",bg="black",width=15,height=1,command=lambda: flip_vertical())
button_fo=tk.Button(form,text="Both",fg="white",bg="black",width=15,height=1,command=lambda: flip_other())
#mirror
button_mr=tk.Button(form,text="Mirror",fg="white",bg="black",width=20,height=1,command=lambda: apply_mirror())
#invert
button_inv=tk.Button(form,text="Invert",fg="white",bg="black",width=20,height=1,command=lambda: aplly_invert())
#contrast
tbContrast = tk.Scale(form, from_=0.0, to=100.0, bg='#303030', fg='White',length=300, highlightthickness=0,
                      resolution=0.1, orient='horizontal', command=tbContrast_ValueChangedEvent)
tbContrast.set(50.0)
#
lblAlpha = tk.Label(form, bg='#353535', fg='White',text='Alpha Value : 1.0')
#brightness
tbBrightness = tk.Scale(form, from_=0.0, to=100.0, bg='#303030', fg='White',length=300, highlightthickness=0,
                      resolution=0.1, orient='horizontal', command=tbBrightness_ValueChangedEvent)
tbBrightness.set(20.0)
#
lblBrightness = tk.Label(form, bg='#353535', fg='White',text='Alpha Value : 1.0')
#histogram eq
button_he=tk.Button(form,text="Histogram Equalization",fg="white",bg="black",width=20,height=1,command=lambda: aplly_histogram())
#clahe
button_cl=tk.Button(form,text="Clahe",fg="white",bg="black",width=20,height=1,command=lambda: aplly_clahe())
#crop
lbl_crop1 = tk.Label(form, bg='#353535', fg='White',text='enter crop pixels')
e1 = tk.IntVar()
e2 = tk.IntVar()
e3 = tk.IntVar()
e4 = tk.IntVar()
e1.set(100)
e2.set(100)
e3.set(400)
e4.set(400)
input1=tk.Entry(form,width=8,textvariable = e1)
lbl_crop2 = tk.Label(form, bg='#353535', fg='White',text='topLX')
input2=tk.Entry(form,width=8,textvariable = e2)
lbl_crop3 = tk.Label(form, bg='#353535', fg='White',text='topLY')
input3=tk.Entry(form,width=8,textvariable = e3)
lbl_crop4 = tk.Label(form, bg='#353535', fg='White',text='botRX')
input4=tk.Entry(form,width=8,textvariable = e4)
lbl_crop5 = tk.Label(form, bg='#353535', fg='White',text='botRY')
button_crop=tk.Button(form,text="Crop",fg="white",bg="black",width=12,height=1,command=lambda: aplly_crop())

label2=tk.Label(form,text="Open an Image:",font="Times 16 italic")
label2.place(x=120,y=15)
button1=tk.Button(form,text="Open",fg="white",bg="black",width=18,height=1,command=openImg)
button1.place(x=280,y=16)

label3=tk.Label(form,text="Choose a filter: ",font="Times 16 italic")
label3.place(x=134,y=70)

button_reset=tk.Button(form,text="Reset",fg="white",bg="black",width=12,height=1,command=reset)
button_reset.place(x=480,y=70)

button_save=tk.Button(form,text="Save",fg="white",bg="black",width=12,height=1,command=save)
button_save.place(x=480,y=16)


cmb1Arr=['_1977', 'aden', 'brannan','brooklyn', 'clarendon', 'earlybird','gingham', 'inkwell', 'kelvin', 'lark','amaro','casper', 'spongebob', 'emboss', 'outline', 'sharpening','beginnerLuck', 'shmoo', 'corpsebride','pencilsketch', 'cartoon','toaster', 'vintageFilter', 'carbile']
ctrl1=tk.StringVar()
ctrl1.set('_1977')
cmb1=Combobox(form,values=cmb1Arr, state="readonly",height=10,width=16,textvariable=ctrl1)
cmb1.place(x=280,y=75)
button3=tk.Button(form,text="Filter Apply",fg="white",bg="black",height=2,width=15,command=applySelections)
button3.pack(side = BOTTOM, fill=X)

#left frame

Button(form, text='Rotate',command=lambda: rotate_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)
Button(form, text='Flip',command=lambda: flip_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)
Button(form, text='Mirror',command=lambda: mirror_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)
Button(form, text='Crop',command=lambda: crop_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)
Button(form, text='Contrast',command=lambda: contrast_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)
Button(form, text='Brightness',command=lambda: brightness_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)
Button(form, text='Invert',command=lambda: invert_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)
Button(form, text='HistogramN',command=lambda: histogram_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)
Button(form, text='Clahe',command=lambda: clahe_btn()).pack(side=TOP, anchor=W, fill=Y, expand=YES)

form.mainloop()



