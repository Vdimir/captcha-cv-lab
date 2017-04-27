from tkinter import *
from PIL import ImageTk, Image
import os
from os.path import isfile, join


class ImgIter:
    def __init__(self):
        self.dirname = "/mnt/userdata2/cvis/captcha/_out3/"
        listdir = os.listdir(self.dirname)
        self.fnames = [f for f in listdir if isfile(join(self.dirname, f))]
        self.i = 0
        self.n = len(self.fnames)

    def accept(self, good):
        if good:
            subdir = '/goods/'
        else:
            subdir = '/bads/'
        os.rename(self.get_fname(), self.dirname+subdir+self.fnames[self.i])

    def get_fname(self):
        return self.dirname+self.fnames[self.i]

    def get_name(self):
        return self.fnames[self.i].replace(".jpg","")

    def get_prog(self):
        return str(self.i) + '/' + str(self.n)

    def next(self):
        self.i += 1
        return self.i == self.n


it = ImgIter()

root = Tk()

img = ImageTk.PhotoImage(Image.open(it.get_fname()))
panel = Label(root, image=img)
panel.pack(side="bottom", fill="both", expand="yes")

w = Label(root, text=it.get_prog())
w.pack()

w2 = Label(root, text=it.get_name())
w2.pack()


def next_img(good):
    it.accept(good)
    if it.next():
        pass
    img2 = ImageTk.PhotoImage(Image.open(it.get_fname()))
    panel.configure(image=img2)
    panel.image = img2
    w.config(text=str(it.get_prog()))
    w2.config(text=str(it.get_name()))


def callback1(e):
    next_img(False)

def callback2(e):
    next_img(True)


root.bind('<space>', callback1)
root.bind('<Return>', callback2)
root.mainloop()
