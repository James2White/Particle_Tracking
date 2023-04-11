
# iii_Tracker.py - locates particles in each frame, links trajectories between frames and filters locations/ trajectories
# Similar to the Predictive Tracker Code (Matlab)

#from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3





import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import glob
import pims
from pims import ImageSequence
import PIL
import trackpy as tp
import cv2
import os
import math
import seaborn as sns
sns.set_palette("Set1")

# # %%
# Location = 'Z:\\labdata\\project_optical_sorting\\Data/20220415\\ShazND_740nm_100mW/2022_04_15_14_29_11'
# Location = 'Z:\\labdata\\project_optical_sorting\\Data/20220415\\ShazND_720nm_100mW/2022_04_15_13_01_17'
# DN = os.path.basename(os.path.dirname(Location))


Location = 'D:\\Data/20220517/ShazND_735nm_050mW/2022_05_17_16_41_34/'
DN = os.path.basename(os.path.dirname(os.path.dirname(Location)))
DIR = os.path.dirname(os.path.dirname(os.path.dirname(Location)))

Set = 0

# # Subset Variables
subset = True
LF = 0
UF = 2000

# # Bright Locater Variables
pixCon = 10
d_pSize = 35
d_mMass = 15000
b_pSize = 35
b_mMass = 20000
thresh = 0

# # Linker Variables
pDist = 120
lDist = 200
mem = 0

# # Filter Variables    
filt = 10
lmass = 5000
umass = 80000
lsize = 5
usize = 20
shape = 1
lxdis = 50
lydis = 0
fps = 80

def Locate(Location, Set, DN, subset, LF, UF, pixCon, d_pSize, d_mMass, b_pSize, b_mMass, thresh):
  
    frames = []
    
    FileList = glob.glob (Location + '/Frames_Edited/*.png')  
    FileList.sort()
    
    # frames = pims.ImageSequence(Location + '/Frames_Edited/*.png', asgrey = True)

    
    if subset == False:
        LF = 0
        UF = len(FileList)   
    else:
        pass
    
    for F in FileList[LF:UF]:
        frame = cv2.imread(F, cv2.IMREAD_GRAYSCALE)       
        frames.append(frame)

    # Locates Particles in the First Frame
    fd = tp.locate(frame, d_pSize, minmass= d_mMass, threshold = thresh, invert=True) # Locate Gaussian-like blobs of some approximate size (Psize) in an image (invert for Brightfield).FOI
    fb = tp.locate(frame, b_pSize, minmass= b_mMass, threshold = thresh, invert=False) # Locate Gaussian-like blobs of some approximate size (Psize) in an image (invert for Brightfield).FOI
      
    fig1 = plt.figure(dpi = 1200)
    plt.title(DN + ' - Dark Particles (' + str(UF) + ')\n ' + 'minMass = ' + str(d_mMass) + ', pSize = ' + str(d_pSize)+ ', thresh = ' + str(thresh))
    tp.annotate(fd, frame)

    fig2 = plt.figure(dpi = 1200)
    plt.title(DN + ' - Bright Particles (' + str(UF) + ')\n ' + 'minMass = ' + str(b_mMass) + ', pSize = ' + str(b_pSize)+ ', thresh = ' + str(thresh))
    tp.annotate(fb, frame);

    fd = tp.batch(frames, d_pSize, minmass= d_mMass, threshold = thresh, invert=True, processes = 'auto')
    fb = tp.batch(frames, b_pSize, minmass= b_mMass, threshold = thresh, invert=False, processes = 'auto')

    fd.to_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Dark_Particle_Locations.pickle')
    fb.to_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Bright_Particle_Locations.pickle')
    
    fig1.savefig(Location + '/Processing/Set' + str(Set) + '/Plots/1_' + DN + '_Dark_Particle_Locations.png')
    fig2.savefig(Location + '/Processing/Set' + str(Set) + '/Plots/2_' + DN + '_Bright_Particle_Locations.png')

    LocaterVar = pd.DataFrame({'subset': subset, 'LF': LF, 'UF': UF, 'pixCon': pixCon, 'd_pSize': d_pSize, 'd_mMass': d_mMass, 'b_pSize': b_pSize, 'b_mMass': b_mMass, 'thresh': thresh}.items())
    LocaterVar.to_pickle(Location + '/Processing/Set' + str(Set) + '/Variables/Locater_V.pickle') 


    # if inv == False:
    #     # Locates Particles in the First Frame
        

            
    #     fb = tp.batch(frames, pSize, minmass= mMass, threshold = thresh, invert=inv, processes = 'auto')
    #     fb.to_pickle(Location + '/Datasets/Bright_Particle_Locations.pickle')
    #     fig1.savefig(Location + '/Plots/1_Bright_Particle_Locations.png')

    
## Uses a predictor and adaptive search to link particle location to find trajectories
 
def Link(Location, Set, DN, subset, LF, UF, pDist, lDist, mem):
        
    FileList = glob.glob (Location + '/Frames_Edited/*.png')
    
    if subset == False:
        LF = 0
        UF = len(FileList)   
    else:
        pass
    
    fd = pd.read_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Dark_Particle_Locations.pickle')
    fb = pd.read_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Bright_Particle_Locations.pickle')

    pred = tp.predict.NearestVelocityPredict()
    # pred = tp.predict._RecentVelocityPredict()
    #pred = tp.predict.ChannelPredict(pDist, 'x', minsamples = 2)
    
    td = pred.link_df(fd, lDist, memory = mem, adaptive_stop = 5, adaptive_step = 0)
    tb = pred.link_df(fb, lDist, memory = mem, adaptive_stop = 5, adaptive_step = 0)
    
    tb['particle'] = tb['particle'] + td['particle'].max() + 1

    t = pd.concat([td, tb], ignore_index = True)

    t.to_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Linked_Trajectories.pickle')
    
    t2 = t.iloc[:50001]
    
    fig3 = plt.figure(dpi = 1200)
    plt.title(DN + '\nTrajectory Trace - Frames: ' + str(LF) + ' - ' + str(UF))
    #FileList = glob.glob (Location + '/Frames_Edited/*.png')
    #plt.imshow(cv2.imread(FileList[-1], cv2.IMREAD_GRAYSCALE), cmap = 'gray')
    tp.plot_traj(t2);

    #tb.to_pickle(Location + '/Datasets/Bright_Linked_Trajectories.pickle')
    fig3.savefig(Location + '/Processing/Set' + str(Set) + '/Plots/3_' + DN + '_Linked_Trajectories.png')

    LinkerVar = pd.DataFrame({'subset': subset, 'LF': LF, 'UF': UF, 'pDist': pDist, 'lDist': lDist, 'mem': mem}.items())
    LinkerVar.to_pickle(Location + '/Processing/Set' + str(Set) + '/Variables/Linker_V.pickle')


## Filters the trajectories using track length, brightness, size and eccentricity variables.

def Filter(Location, Set, DN, DIR, filt, lmass, umass, lsize, usize, shape, lxdis, lydis, fps):    

    t = pd.read_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Linked_Trajectories.pickle')
    FileList = glob.glob (Location + '/Frames_Edited/*.png')

    t1 = tp.filter_stubs(t, filt)
    
    # Compare the number of particles in the unfiltered and filtered data.
    
    print('Before filtering:', t['particle'].nunique())
    print('After filtering quickies:', t1['particle'].nunique())
    
    fig4 = plt.figure(dpi = 1200)
    plt.title(DN + ' - Trajectory Trace\n Removing short tracks (>' + str(filt) + ')')
    #plt.imshow(cv2.imread(FileList[-1], cv2.IMREAD_GRAYSCALE), cmap = 'gray')
    tp.plot_traj(t1)

    
    
    fig5 = plt.figure(dpi = 1200)
    ax = fig5.add_subplot(1, 1, 1)
    plt.title(DN + ' - Particle Size / Mass')
    ax.add_patch(Rectangle((lmass, lsize), umass - lmass, usize - lsize,linewidth=1,edgecolor='r',facecolor='none'))
    tp.mass_size(t.groupby('particle').mean());
   
    
    # Filter for Size and Brightness using boundary conditions
    
    t2 = t1[((t1['mass'] > lmass) & (t1['mass'] < umass) & (t1['size'] > lsize)  & (t1['size'] < usize) & (t1['ecc'] < shape))]
    
    print('After filtering for size n brightness:', t2['particle'].nunique())

  
    # Cleans up datasets, calculates displacement and velocity (x,y|u,v)
    
    t2.index.name = None
    t2 = t2.sort_values(by=['particle', 'frame'])
    
    t2 = t2.drop(columns=['signal', 'raw_mass', 'ep'])
    t2['time'] = t2.frame / fps #fps
    t2 = t2[["particle", "frame", "time", "x", "y", "mass", "size", "ecc"]]
      
    fig6 = plt.figure(dpi = 1200)
    plt.title(DN + ' - Trajectory Trace \n Mass = ' + str(lmass) + ' - ' + str(umass) + ', Size = '+ str(lsize) + ' - ' + str(usize))
    #plt.imshow(cv2.imread(FileList[-1], cv2.IMREAD_GRAYSCALE), cmap = 'gray')
    tp.plot_traj(t2);
    
    
    # Calculates Displacement and Velocity
    
    t2['xdis'] = t2.x.diff()
    t2['ydis'] = t2.y.diff()
    
    t2['xdis'] = t2['xdis'].where(t2.particle.diff() == 0, float('nan'))
    t2['ydis'] = t2['ydis'].where(t2.particle.diff() == 0, float('nan'))
    
    t2['u'] = (t2.x.diff()/t2.time.diff()) 
    t2['v'] = (t2.y.diff()/t2.time.diff())  
    
    t2['u'] = t2['u'].where(t2.particle.diff() == 0, float('nan'))
    t2['v'] = t2['v'].where(t2.particle.diff() == 0, float('nan'))

    
    t3 = pd.DataFrame()
    t3['particle'] = t2['particle'].unique()
    t3 = t3.set_index('particle')
    t3.index.name = 'particle'    

    grouped1 = t2.groupby("particle")
    
    for i in t3.index:
        group1 = grouped1.get_group(i).reset_index()
        t3.loc[i,1] = abs(group1.iloc[-1,4] - group1.iloc[0,4])
        t3.loc[i,2] = abs(group1.iloc[-1,5] - group1.iloc[0,5])

    t3 = t3.reset_index()
    t3.columns = ['particle', 'txdis', 'tydis']
    
    t2 = pd.merge(t2, t3, on = 'particle')
    
    t4 = t2[(t2['txdis'] > lxdis)]

    fig7 = plt.figure(dpi = 300)
    plt.title(DN + ' - Total Displacement of each track')
    plt.xlabel('Total X Displacement [pixels]'); plt.ylabel('Total Y Displacement [pixels]')
    plt.plot(t4.txdis, t4.tydis, 'o', alpha = 0.3, markersize=2)
    plt.gca().add_patch(mpl.patches.Rectangle((0, 0), lxdis, lydis, linewidth=1, edgecolor='r',facecolor='none'))
    plt.xlim(0);plt.ylim(0)
    print('After filtering for total displacement:', t4['particle'].nunique())
    
        
    fig8 = plt.figure(dpi = 300)
    plt.title(DN + ' - Trajectory Trace \n Removing Out of Focus Trajectories')
    #plt.imshow(cv2.imread(FileList[0], cv2.IMREAD_GRAYSCALE), cmap = 'gray')
    tp.plot_traj(t4);
    
    # Saves the plots as figures
    fig4.savefig(Location + '/Processing/Set' + str(Set) +'/Plots/4_' + DN + '_Remove_Short_Tracks.png')
    fig5.savefig(Location + '/Processing/Set' + str(Set) +'/Plots/5_' + DN + '_Mass_Size.png')
    fig6.savefig(Location + '/Processing/Set' + str(Set) +'/Plots/6_' + DN + '_Remove_Mass_Size.png')
    fig7.savefig(Location + '/Processing/Set' + str(Set) +'/Plots/7_' + DN + '_Total_Displacement.png')
    fig8.savefig(Location + '/Processing/Set' + str(Set) +'/Plots/8_' + DN + '_Remove_Distance.png')
    
    fig8.savefig(DIR + '/Compiled_Results/Plots/8_' + DN + '_Remove_Distance.png')

    
    # Creates pickles for the different datasets
    
    t1.to_pickle(Location + '/Processing/Set' + str(Set) +'/Datasets/' + DN + '_t1.pickle')
    t2.to_pickle(Location + '/Processing/Set' + str(Set) +'/Datasets/' + DN + '_t2.pickle')
    t4.to_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_t4.pickle')

    FilterVar = pd.DataFrame({'filt': filt, 'lmass': lmass, 'umass': umass, 'lsize': lsize, 'usize': usize, 'shape': shape, 'lxdis': lxdis, 'lydis': lydis}.items())
    FilterVar.to_pickle(Location + '/Processing/Set' + str(Set) + '/Variables/Filter_V.pickle')  
