import cv2
from matplotlib import pyplot as plt
import random
from sklearn.cluster import KMeans
import numpy as np
from functools import reduce
from operator import add

import pytesseract
from PIL import Image


def show(img):
    if isinstance(img, list):
        for i in img:
            show_image(i)
    else:
        show_image(img)


def show_image(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img, cmap='brg')
    plt.show()


def crop(img):
    (x, y, w, h) = cv2.boundingRect(img)
    t = img[y:y + h, x:x + w]
    return t


def crop_erode(img):
    c_erode = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    c_erode = cv2.dilate(c_erode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
    (x, y, w, h) = cv2.boundingRect(c_erode)
    d=5
    (x, y, w, h) = (max(x-d,0), max(y-d,0), w+d, h+d)
    t = img[y:y + h, x:x + w]
    return (x,t)


def load_random_img(n=None):
    fname = n or round(random.random()*100)
    img = cv2.imread("img/4/%d.jpg" % fname)
    return fname, img


def k_means_cluster_centres(image):
    clt = KMeans(n_clusters=12)
    clt.fit(image)
    return clt.cluster_centers_


def get_cluster(img, cluster):
    d = 20
    res = cv2.inRange(img, cluster-d,cluster+d)
    return res


def get_hist(img):
    arr = np.transpose(img)
    hist = np.zeros(arr.shape[0])
    m = np.max(arr)
    m = max([m, 1])
    for i in range(arr.shape[0]):
        hist[i] = sum(arr[i]/m)
    return hist


class IntervalBuilder:
    def __init__(self):
        self.beg = None
        self.end = None
        self.ints = []

    def cont(self, val):
        if self.beg is None:
            self.beg = val
        else:
            self.end = val

    def brk(self):
        if self.end is not None:
            self.ints.append((self.beg, self.end))
        self.beg = None
        self.end = None


def fund_cont_subsec(arr):
    intervals = IntervalBuilder()
    for i,a in enumerate(arr):
        if a != 0:
            intervals.cont(i)
        else:
            intervals.brk()
    intervals.brk()
    return intervals.ints


def clusterize(img):
    h,w = img.shape[0],img.shape[1]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    im = img.reshape((img.shape[0] * img.shape[1], 3))

    clt = KMeans(n_clusters=8)

    labels = clt.fit_predict(im)
    for x in range(8):
        cluster_cent = clt.cluster_centers_.astype("uint8")
        cluster_cent = np.zeros(cluster_cent.shape)
        cluster_cent[x] = clt.cluster_centers_.astype("uint8")[x]
        quant = cluster_cent[labels]
        quant = quant.reshape((h, w, 3)).astype("uint8")
        yield quant


def iterate_2d_arr(img):
    for row in img:
        for p in row:
            yield p


def simple_bin(img):
    res = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if not np.all(img[x,y] == 0):
                res[x,y] = 1

    return res


def non_zero_pixels(img):
    s = 0
    for p in iterate_2d_arr(img):
        if (not np.all(p == 0)):
            s += 1
    return s


def recognize(img):
    pilimg = Image.fromarray(img)
    res = pytesseract.image_to_string(pilimg,
                                      config='-psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
    return res


def filter_comp_count(img):
    c2 = cv2.connectedComponentsWithStats(morph(img))[0]
    return 1 < c2 <= 10


def morph(img):
    c_erode = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    c_erode = cv2.morphologyEx(c_erode, cv2.MORPH_DILATE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 6)))
    return c_erode


def try_collect_letters(img):
    c_m = morph(img)
    return fund_cont_subsec(get_hist(c_m))


def non_zero_seq(img):
    int_len = lambda t: t[1] - t[0]
    s = try_collect_letters(img)
    # s = filter(lambda t: int_len(t) >= 10, s)
    # s = list(s)
    # if len(s) == 0:
    #     return False
    # if len(s) > 10:
    #     return False
    if any(map(lambda t: int_len(t) < 10, s)):
        return False
    if any(map(lambda t: int_len(t) > 50, s)):
        return False
    return True


def get_filtered_clusters(img):
    clusters = map(simple_bin, clusterize(img))
    clusters = filter(filter_comp_count, clusters)
    clusters = filter(non_zero_seq, clusters)
    clusters = list(clusters)
    return clusters


def break_captcha(img):
    res = []
    for c in get_filtered_clusters(img):
        inter = try_collect_letters(c)
        for (b, e) in inter:
            letter = recognize(c[:, b:e])
            res.append((b, letter))
    r = reduce(add, map(lambda x: x[1], sorted(res, key=lambda x: x[0])), '')
    return r
