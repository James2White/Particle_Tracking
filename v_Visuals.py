## This code creates a gif of the tracking traces over the video frames
#
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import glob
import io
import os
from PIL import Image 

Location = 'D:\\Data/20220518/ShazND_800nm_100mW/2022_05_18_17_04_31/'

DN = os.path.basename(os.path.dirname(os.path.dirname(Location)))
DIR = os.path.dirname(os.path.dirname(os.path.dirname(Location)))

Set = 1
#Set = 'D:\\Data/20220518/ShazND_800nm_100mW/2022_05_18_17_04_31/Processing/Set1/'

edited = True
LF = 0
UF = 10
tracking = True


def Visuals(Location, Set, DN, LF, UF, edited, tracking):

    LFV = LF
    UFV = UF
        
    # Location = (r'C:\Users\mq42458072\Desktop\Tracking.py\Data\20200406')
    # Name = 'YG_500nm_CAPI_200mW'
    
    if edited == True:
        Frames = glob.glob (Location + '/Frames_Edited/E*.png')
    else:
        Frames = glob.glob (Location + '/Frames/*.png')
     
    Frames.sort()
    
    # access the trajectory dataframe
    
    tracks = pd.read_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_t2.pickle')
    tracks = tracks.sort_values(by=['particle'])
    
    # Code identifies how many unique particles are in the dataframe and assigns a random colour to each in rgb scaled to 0:1
    colors = []
    Particles = pd.DataFrame(sorted(tracks['particle'].unique()))
    Particles.columns=['particle']
    
    for P in range(len(Particles)):
        randcolor = (np.random.choice(range(256), size=3))
        scale_color = [x / 256 for x in randcolor]
        colors.append(scale_color)
    
    Particles['colors'] = colors
    
    # Merges the assigned colors for each particle with the dataframe
    tracks_color = tracks.reset_index().merge(Particles, how="left").set_index('index')
    tracks_color = tracks_color.dropna()
    tracks_color = tracks_color.sort_values(by=['particle','frame'])
    
    images = []
     
    # For each frame the code opens an image, overlays the particle placement before saving to a stack
    for i in range(LF,UF):
        
        im = Image.open(Frames[i])
        plt.imshow(im, cmap = 'gray')
        
        if tracking == True:
           
            tracks_filt = tracks_color[(tracks_color['frame'] >= i ) & (tracks_color['frame'] <= i+1)] 
            
            for P in range(len(tracks_filt)):
                try:
                    plt.plot(tracks_filt.iat[P, 3], tracks_filt.iat[P, 4], 'o', color = tracks_filt.iat[P, 14], markersize=5)
                except:
                    pass
        
        buf = io.BytesIO()
        plt.title('Particle Trace for Frames ' + str(LF) + ' to ' + str(UF))
        plt.savefig(buf, format='png')
        buf.seek(0)
        im2 = Image.open(buf)
        
        images.append(im2)
    
        print("Frame " + str(i) + " / " + str(UF-1))
        
    if tracking == True:
        images[0].save(Location + '/Processing/Set' + str(Set) + '/Plots/Tracjectory_Trace.gif', save_all=True, append_images=images[1:], duration=10, loop=0)
    else:
        images[0].save(Location + '/Processing/Set' + str(Set) + '/Plots/Animated_Frames.gif', save_all=True, append_images=images[1:], duration=10, loop=0)
   
    
    im.close()
    VisualsVar = pd.DataFrame({'edited': edited, 'tracking': tracking, 'LFV': LFV, 'UFV': UFV}.items())
    VisualsVar.to_pickle(Location + '/Processing/Set' + str(Set) + '/Variables/Visuals_V.pickle')
    