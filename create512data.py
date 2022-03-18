import numpy as np
from PIL import Image
import cv2
import os

root = "/scratch/nka77/DATA/test/"
outdir = "/scratch/nka77/DATA/test512/images/"
os.makedirs(outdir, exist_ok=True)

count = 0
with open("/scratch/nka77/DATA/test512/list/demo.txt", "w") as outlist:
    with open("/scratch/nka77/DATA/test/list/demo.txt") as f:
         im_i = f.readline()
         while im_i != None:
             count += 1
             if count%100==0:
                 print(count)
             
             #if (im_i[:-1] + "_pre_disaster.png").split('/')[-1] not in os.listdir(root+"train/images/") and (im_i[:-1] + "_pre_disaster.png").split('/')[-1] not in os.listdir(root+"tier3/images"):
             #    print(im_i)
             #    im_i = f.readline() 
             #    continue
             '''
             img = np.array(Image.open((root + im_i[:-1].replace("images","masks") + "_post_disaster.png")))
             im_ = im_i.split("/")[-1]

             cv2.imwrite((outdir + im_[:-1] + "_1" + "_post_disaster.png"), img[:512,:512])
             cv2.imwrite((outdir + im_[:-1] + "_2" + "_post_disaster.png"), img[512:,:512])
             cv2.imwrite((outdir + im_[:-1] + "_3" + "_post_disaster.png"), img[:512,512:])
             cv2.imwrite((outdir + im_[:-1] + "_4" + "_post_disaster.png"), img[512:,512:])
             
             cv2.imwrite((img[:512,:512]).save(outdir + im_[:-1] + "_1" + "_post_disaster.png")
             cv2.imwrite((img[:512,:512]).save(outdir + im_[:-1] + "_2" + "_post_disaster.png")
             cv2.imwrite((img[:512,:512]).save(outdir + im_[:-1] + "_3" + "_post_disaster.png")
             cv2.imwrite((img[:512,:512])(outdir + im_[:-1] + "_4" + "_post_disaster.png")
             '''  
             img = Image.open(root + im_i[:-1] + "_pre_disaster.png")
             im_ = im_i.split("/")[-1]
             img = np.array(img)

             
             Image.fromarray(img[:512,:512,:]).save(outdir + im_[:-1] + "_1" + "_pre_disaster.png")
             Image.fromarray(img[512:,:512,:]).save(outdir + im_[:-1] + "_2" + "_pre_disaster.png")
             Image.fromarray(img[:512,512:,:]).save(outdir + im_[:-1] + "_3" + "_pre_disaster.png")
             Image.fromarray(img[512:,512:,:]).save(outdir + im_[:-1] + "_4" + "_pre_disaster.png")

             img = np.array(Image.open(root + im_i[:-1] + "_post_disaster.png"))
             Image.fromarray(img[:512,:512,:]).save(outdir + im_[:-1] + "_1" + "_post_disaster.png")
             Image.fromarray(img[512:,:512,:]).save(outdir + im_[:-1] + "_2" + "_post_disaster.png")
             Image.fromarray(img[:512,512:,:]).save(outdir + im_[:-1] + "_3" + "_post_disaster.png")
             Image.fromarray(img[512:,512:,:]).save(outdir + im_[:-1] + "_4" + "_post_disaster.png")

             outlist.write("images/"+im_[:-1] + "_1\n")
             outlist.write("images/"+im_[:-1] + "_2\n")
             outlist.write("images/"+im_[:-1] + "_3\n")
             outlist.write("images/"+im_[:-1] + "_4\n")
             im_i = f.readline()
    f.close()
outlist.close()      
'''
count = 0
with open("/scratch/nka77/DATA/trainval512/list/train.txt") as inlist:
   with open("/scratch/nka77/DATA/trainval512/list/train1.txt", "w") as outlist:
       im = inlist.readline()
       while(im != None and im != ""):
          count += 1
          if count%100==0:
             print(count)
          outlist.write(im[:-1]+"_1\n")
          outlist.write(im[:-1]+"_2\n")
          outlist.write(im[:-1]+"_3\n")
          outlist.write(im[:-1]+"_4\n")
          im = inlist.readline()
       outlist.close()
   inlist.close()'''
