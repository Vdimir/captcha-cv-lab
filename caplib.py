import cv2
from matplotlib import pyplot as plt
import random
from sklearn.cluster import KMeans
import numpy as np


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

    for i in range(arr.shape[0]):
        hist[i] = sum(arr[i])
    return hist


def find_longest_subseq(arr, elem):
    max_len = 0
    cur_len = 0
    changes = 0
    for x in arr:
        if x <= elem:
            cur_len += 1
        else:
            max_len = max(max_len, cur_len)
            if (cur_len > 0):
                changes += 1
            cur_len = 0
    return (max_len, changes)

#     inr_erode = cv2.erode(inr, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
#     (x,y,w,h) = tuple(map(add, cv2.boundingRect(inr_erode), (-4,-4,4,4)))
#     res = inr[y:y+h,x:x+w]
# #     res = inr
#     if (w == 0 or h == 0):
#         return None

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

def simple_bin(img):
    for row in img:
        for p in row:
            if (np.all(p == [0,0,0])):
                p[:] = 0
            else:
                p[:] = 255
    return img

# import pytesseract
#
# from PIL import Image
# def recognize(img):
#     pilimg = Image.fromarray(img)
#     res = pytesseract.image_to_string(pilimg,
#                                       config='-psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
#     return res

# dirmane = "/home/vdimir/userdata2/cvis/captcha/out/"+str(fname)
# if not os.path.exists(dirmane):
#     os.makedirs(dirmane)
# cv2.imwrite(dirmane+"/"+ str(x)+".jpg", quant)
