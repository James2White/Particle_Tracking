    
# ii_Slicer.py takes in a FileName and Savelocation to slice .avi video into Frames. 
# Should pass the desired number of slices per video, filter and sharpening variables.
# Returns a png stack and a pickle containing fps, totFrames, filtering and sharpening variables.

# Also generates the folders and data storage files for further processing.

import glob
import cv2
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image, ImageOps, ImageChops, ImageFilter


# Location = 'Z:\\labdata\\project_optical_sorting\\Data/20220415\\ShazND_720nm_100mW/2022_04_15_13_01_17'
# DN = os.path.basename(os.path.dirname(Location))

Location = 'D:\\Data/20220518/ShazND_800nm_100mW/2022_05_18_17_04_31/'
DN = os.path.basename(os.path.dirname(os.path.dirname(Location)))
Set = 1

subset = True
LF = 0
UF = 1
v_bri = 1
v_con = 2
nb = True
flip = False


def Slicer(Location, Set, DN, subset, LF, UF, nb, flip, v_bri, v_con):
    
    def NormaliseArray(array):
        return (array - np.min(array)) / (np.max(array) - np.min(array))
    
#    edited_frames = glob.glob (Location + '/E*.png')
#    
#    for e in edited_frames:
#        try:
#            os.remove(e)
#        except:
#            pass


    imlist = glob.glob (Location + '/Frames/*.png')
    imlist.sort()
    
    if subset == False:
        LF = 0
        UF = len(imlist)   
    
    #Image List Information
    
    img = Image.open(imlist[0])

    width, height = img.size
    
    backarr = np.zeros((height, width), dtype = np.int64)

    if nb == True:
    
        # Build up average pixel intensities, casting each image as an array of floats
        for im in imlist:
            imarr = np.array(Image.open(im), dtype = np.int64)
            backarr = backarr + imarr/ len(imlist)
        
        # Round values in array and cast as 8-bit integer
        backoutarr = np.array(backarr*0.8, dtype = np.uint8)
        backout = Image.fromarray(backoutarr, 'L')
        backout.save(Location + '/Processing/Set' + str(Set) + '/Plots/0_' + DN +'_Background.png')

    else:
        
        try:
            
            backout = Image.open(Location + '/Processing/Set' + str(Set) + '/Plots/0_' + DN +'_Background.png')
            backoutarr =  np.array(np.array(backout)*0.8, dtype = np.uint8)
        
        except:
            
            backoutarr = backarr
    

        
    counter = 0
   
    for im in imlist[LF:UF]:
        
        print ("\r Slicing Frame ", len(imlist[LF:UF]) - counter)

        counter += 1

        img1 = Image.open(im)
        img1arr = np.array(img1, dtype = np.uint8)
        
        forearr = cv2.subtract(img1arr, backoutarr)
        
        normfore = NormaliseArray(forearr)*255
        fore = Image.fromarray(normfore).convert("L")

        bright = ImageEnhance.Brightness(fore).enhance(v_bri)
 
        if flip == True:
            
            flipIM = fore.transpose(Image.FLIP_LEFT_RIGHT)
            bright = ImageEnhance.Brightness(flipIM).enhance(v_bri)

        
        # blurarr = np.array(blur, dtype = np.int64)
        # blur.show()

        # subavearr = cv2.subtract(blurarr, blurarr.mean())
        # subave = PIL.Image.fromarray(subavearr).convert("L")
    
        #inv = ImageOps.invert(bright)

        contrast = ImageEnhance.Contrast(bright).enhance(v_con)

        #blur = contrast.filter(ImageFilter.GaussianBlur(radius=2))

        plt.imsave(Location + '/Frames_Edited/' + 'E_' + os.path.basename(im), contrast, cmap = 'gray', dpi = 1200)
    
    
    fig, axs = plt.subplots(2, 2,figsize=(10, 8), dpi= 1200)
    fig.suptitle(DN + ' - Image Enhancement of Frame ' + str(UF))

    axs[0, 0].imshow(backout, cmap = 'gray')
    axs[0, 0].title.set_text('Background')
    axs[0, 1].imshow(fore, cmap='gray')
    axs[0, 1].title.set_text('Foreground')
    axs[1, 0].imshow(bright, cmap='gray')
    axs[1, 0].title.set_text('Bright')
    axs[1, 1].imshow(contrast, cmap='gray')
    axs[1, 1].title.set_text('Contrast')
    fig.tight_layout()

    fig.savefig(Location + '/Processing/Set' + str(Set) + '/Plots/0_' + DN +'_Image_Enhancement.png')
    SlicerVar = pd.DataFrame({'subset': subset, 'LF': LF, 'UF': UF, 'v_bri': v_bri, 'v_con': v_con}.items())
    SlicerVar.to_pickle(Location + '/Processing/Set' + str(Set) + '/Variables/Slicer_V.pickle')   

