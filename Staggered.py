# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:45:21 2022

@author: MQ42458072
"""



import glob
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from _Folders import folder_generator
import joypy
#
#nBins = 100
#
#Date = '20200721'
#Path = (r'C:\Users\mq42458072\Desktop\Data')
##Path = (r'Z:\labdata\project_optical_sorting')
##Path = (r'C:/Users/mq42458072/Desktop\Tracking.py\Data')
#
#Experiment = 'Exp2'
#
#FileLocation = (Path + '/' + Date + '/')
#FileList = glob.glob (FileLocation + '/' + '*.avi')
#


## Removes duplicate names from the list
#FileNames = list(set(FileNames))
#FileNames.sort()

# Location = (r'Z:\labdata\project_optical_sorting\Data\20220415')
# FolderList = glob.glob (Location + '/*mW')

FolderList = []

DIR = (r'D:/Data/20220517')
FolderList = glob.glob (DIR + '/*50mW')

DIR2 = (r'D:\\Data/20220518')
FolderList = FolderList + glob.glob (DIR2 + '/*100mW')

Set = 0

stat = 'Mean'
ptile = 5

FileNames = []

for i in FolderList[0]:
    Name = (os.path.basename(i))
    End = Name.find('mW')
    Name = Name[:End+2]
    FileNames.append(Name)





def Staggered(DIR, Set, stat, ptile):
   
    def gauss(x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
   
    folder_generator(DIR, '/Compiled_Results')
    #folder_generator(Location + '/Compiled_Results/', Experiment)
   
    df0 = pd.DataFrame()
    
    df5 = pd.DataFrame()
    df6 = pd.DataFrame()
    df7 = pd.DataFrame()




    
    for f in FolderList:

        Name = (os.path.basename(f))
        End = Name.find('mW')
        Name = Name[:End+2]
        
        uScores = [i for i in range(len(Name)) if Name.startswith('_', i)] 
        pName = Name[:uScores[0]]
        wLength = int(Name[uScores[0]+1:uScores[1]-2])
        power = int(Name[uScores[1]+1:End])
        info = [Name, pName, wLength, power, ptile]
        
        df0 = df0.append([info], sort = False, ignore_index=True)
    
        SubFolder = glob.glob (f + '/*')
        DN = os.path.basename(f)
        DN1 = (DN + '_' + str(ptile))
        
        beamO = pd.read_pickle(SubFolder[0] + '/Processing/Set' + str(Set)  + '/Datasets/' + DN1 + '_DataOUT.pickle')
        beamO['FileName'] = os.path.basename(f)
        beamI = pd.read_pickle(SubFolder[0] + '/Processing/Set' + str(Set)  + '/Datasets/' + DN1 + '_DataIN.pickle')
        beamI['FileName'] = os.path.basename(f)


        
        df5 = df5.append(beamO, sort = False, ignore_index=True)
        df6 = df6.append(beamI, sort = False, ignore_index = True)
        
    df0.columns = ['FileName', 'pName', 'wLength', 'power', 'ptile']
    
    df5 = df5.sort_values(by=['FileName']).reset_index(drop=True)
    df6 = df6.sort_values(by=['FileName']).reset_index(drop=True) 
    
    beamO5 = pd.merge(df5, df0, on = 'FileName')
    beamI6 = pd.merge(df6, df0, on = 'FileName')

    beamOb = []
    beamIb = []
    
    binsO = []
    binsI = []
    dataO = []
    dataI = []
    
        
     
    grouped5 = beamO5.groupby("wLength")
    grouped6 = beamI6.groupby("wLength")
        
    wLengths = beamO5['wLength'].unique().tolist()

    for j in wLengths:
        group5 = grouped5.get_group(j) 
        group6 = grouped6.get_group(j)         
        
        beamOb.append(group5)
        beamIb.append(group6)

        
    nBins = 1000

        
    for k in range(len(wLengths)):

        binsO = np.linspace(beamOb[k].u.min(), beamOb[k].u.max(), nBins)
        binsI = np.linspace(beamIb[k].u.min(), beamIb[k].u.max(), nBins)

        data1, binsO = np.histogram(beamOb[k].u, bins=nBins, density = True)
        data2, binsI = np.histogram(beamIb[k].u, bins=nBins, density = True)

        widths1 = binsO[:-1] - binsO[1:]
        widths2 = binsI[:-1] - binsI[1:]
        
        TitleName = beamO5['FileName'].unique().tolist()[k]
        
        fig4, ax4 = plt.subplots(constrained_layout=True, dpi = 1200)
        fig4.suptitle(TitleName + ' - Distribution of Parallel Velocities')
        plt.bar(binsO[1:] + widths1/2, data1, width = widths1, alpha = 0.5, label= 'Parallel and Outside Beam')    
        plt.bar(binsI[1:] + widths2/2, data2, width = widths2, alpha = 0.5, label= 'Parallel and Inside Beam')
        plt.legend(loc = 1)
    
        plt.ylabel('Counts'), plt.xlabel('Velocity  ' r'$\mu$m/s');
       
        
    plt.figure(dpi=1200)
    plt.rc("font", size=20)


    fig, axes = joypy.joyplot(beamI6[:], by="wLength", column="u", kind="kde", 
                  range_style='own', tails=0.2,  x_range=[-1000,4000],
                  overlap=5, linewidth=4, fill=True, alpha=0.6, colormap=cm.autumn,
                  grid='y', figsize=(15,15))

        
        
        
        
        