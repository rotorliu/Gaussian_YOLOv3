# -*- coding: utf-8 -*-

import os
from os import walk, getcwd
from PIL import Image
import json

""" Class label (coco_navinfo) """
classes = ["pl" , "pr" , "pa", "w", "ph", "pn", "pg", "i", "ih", "id", "gan", "tl"]

""" Convert function """
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
    
""" Configure Paths"""   
mypath = "/hdd/data/BDD/coco_navinfo/annotations/test.json"         # json file path
outpath = "/hdd/data/BDD/coco_navinfo/annotations/"   # txt file path
imgpath = "/hdd/data/BDD/coco_navinfo/images/"   # image file path
#if not os.path.isdir(outpath):
    #os.mkdir(outpath, 0755)

wd = getcwd()
list_file = open('%s/test_coco_navinfo_list.txt'%(wd), 'w')

""" Get input text file list """
annos = json.load(open(mypath))
images = annos["images"]
annotations = annos["annotations"]

""" Process """
for img in images:
    img_file = imgpath + img["file_name"]

    """ Open input image files """
    img_size = Image.open(img_file).size

    """ Open output text files """
    txt_outpath = outpath + img["file_name"].replace("jpg","txt")
    txt_outfile = open(txt_outpath, 'w')

    ct = 0
    for ann in [a for a in annotations if a["image_id"] == img["id"]]:
        ct = ct + 1
        cls_id = ann["category_id"] - 1 # yolo start with 0 ,but coco is 1
        xmin = ann["bbox"][0]
        xmax = ann["bbox"][0] + ann["bbox"][2]
        ymin = ann["bbox"][1]
        ymax = ann["bbox"][1] + ann["bbox"][3]

        box = (float(xmin), float(xmax), float(ymin), float(ymax))
        bb = convert(img_size, box)
        txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    txt_outfile.close()

    if(ct != 0):
        list_file.write(img_file + "\n")
            
list_file.close()      