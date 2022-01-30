from tkinter import *
from PIL import Image, ImageGrab
import os

import torchvision.models as models
from torchvision import transforms
import pickle as pk

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Paint(object):

    DEFAULT_PEN_SIZE = 4.0
    DEFAULT_COLOR = 'black'
    WIDTH = 400
    HEIGHT = 400

    def __init__(self):
        self.root = Tk()

        self.evaluate_button = Button(self.root, text='evaluate', command=self.evaluate_picture)
        self.evaluate_button.grid(row=0, column=3)
        
        self.eraseall_button = Button(self.root, text='erase all', command=self.erase_all)
        self.eraseall_button.grid(row=0, column=4)
        
        self.c = Canvas(self.root, bg='white', width=self.WIDTH, height=self.HEIGHT)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def evaluate_picture(self):
        winx = self.root.winfo_x()
        winy = self.root.winfo_y()
        self.c.update()
        ImageGrab.grab((winx+10,winy+60,winx+self.WIDTH+10,winy+self.HEIGHT+60)).save('temp.png')

        results = make_prediction_proba(Image.open('temp.png'))
        
        sns.barplot(data=results, x='proba', y='class')
        plt.xlim(0,1)
        plt.show()
        
    def erase_all(self):
        self.c.delete('all')
    
    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        self.line_width = self.DEFAULT_PEN_SIZE
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

def preprocess(img):  
    img = img.resize((224,224)).convert('RGB')
    tensor = transforms.ToTensor()(img)
    return tensor

def make_prediction_proba(img):
    
    dic_class = {k:v for k,v in zip(range(len(os.listdir(test_dir))),os.listdir(test_dir))}
    
    features_extracted = conv_layer.features(preprocess(img).reshape((1, 3, 224, 224)))
    conv_layer_output = features_extracted.detach().numpy().reshape((256 * 6 * 6))
    
    results = pd.DataFrame({
        'class':list(dic_class.values()),
        'proba':clf_trained.predict_proba([conv_layer_output])[0],
    }).sort_values('proba',ascending=False).head()
    
    return results
    
if __name__ == '__main__':
    
    data_dir = 'data/'
    test_dir = os.path.join(data_dir, 'test/')
    
    # Load model
    conv_layer = models.alexnet(pretrained=True)
    with open("data/classifier.pkl", 'rb') as fid:
        clf_trained = pk.load(fid)

    # Start canvas
    Paint()