

import glob
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from _Folders import folder_generator
import seaborn as sns
import math

palette_tab10 = sns.color_palette("colorblind", 6)
palette = sns.color_palette([palette_tab10[0], palette_tab10[1], palette_tab10[2],palette_tab10[3], palette_tab10[4], palette_tab10[5]])
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

# FolderList = []

DIR = (r'D:\\Data/20220517')
FolderList = glob.glob (DIR + '/*0mW')

# DIR2 = (r'D:\\Data/20220518')
# FolderList = FolderList + glob.glob (DIR2 + '/*100mW')

Set = 0

stat = 'Mean'
ptile = 25

#from _Folders import folder_generator
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


def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def Lorentz(f, a, b, c):
    return (a/np.pi)*((0.5*b)/((f-c)**2 + (0.5*b)**2))



# center = 0
# bWidth = 5e-6
# below = center - 2.355*bWidth
# above = center + 2.355*bWidth


# SiV Quantum Emitter
c=3e8 # speed of light
n_p = 2.417
b_width_scale = 10
# gam = (16.39)*(2*(math.pi)*10**6)*b_width_scale ## 1e3 to scale bandwidth
gam = (16.39 + 1.9*10**(-2)*(300**3))*(2*(math.pi)*10**6)*b_width_scale ## 1e3 to scale bandwidth
Gam0 = ((2*math.pi*10**9)/9.74)
Gam = Gam0*n_p



no_d = [0, 2e5] #number of defects
powers = [0.05, 0.1] ## # laser beam power

lam_start= 720e-9 ## # start wlen
lam_end = 745e-9  ## # end wlen
lam_del = 5e-9 ## # change wlen
lam1 = np.arange(lam_start, lam_end, lam_del)

lam_start= 730e-9 ## # start wlen
lam_end = 740e-9  ## # end wlen
lam_del = 1e-9  ## # change wlen
lam2 = np.arange(lam_start, lam_end, lam_del)

lam_start=750e-9 ## # start wlen
lam_end = 780e-9  ## # end wlen
lam_del = 10e-9  ## # change wlen
lam3 = np.arange(lam_start, lam_end, lam_del)

lam = np.concatenate((lam1, lam2, lam3))

lam.sort()
lam = list( dict.fromkeys(lam) ) 


# no_d = [no_d[1]]


stat = 'Mean'

FileNames = []

for i in FolderList[0]:
    Name = (os.path.basename(i))
    End = Name.find('mW')
    Name = Name[:End+2]
    FileNames.append(Name)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth




def Compiler(DIR, Set, stat, ptile):
   
    def gauss(x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    def Lorentz(f, a, b, c):
        return (a/np.pi)*((0.5*b)/((f-c)**2 + (0.5*b)**2))

   
    folder_generator(DIR, '/Compiled_Results')
    #folder_generator(Location + '/Compiled_Results/', Experiment)
    
    df0 = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()

    
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

        splitstats = pd.read_pickle(SubFolder[0] + '/Processing/Set' + str(Set)  + '/Datasets/' + DN1 + '_FFAnalysis_Statistics_Split.pickle')
        splitstats['FileName'] = os.path.basename(f)
        allstats = pd.read_pickle(SubFolder[0] + '/Processing/Set' + str(Set)  + '/Datasets/' + DN1 + '_Analysis_Statistics_All.pickle')
        allstats['FileName'] = os.path.basename(f)
        pushstats = pd.read_pickle(SubFolder[0] + '/Processing/Set' + str(Set)  + '/Datasets/' + DN1 + '_Push_Fitting_Parameters.pickle')
        pushstats['FileName'] = os.path.basename(f)        



#        dataIN = pd.read_pickle(FileLocation + '/' + f  + '/' + Experiment + '/Datasets/DataIN_'+ f +'.pickle')
#        dataOUT = pd.read_pickle(FileLocation  + '/' + f  + '/' + Experiment + '/Datasets/DataOUT_'+ f +'.pickle')
        dataALL = pd.read_pickle(SubFolder[0] + '/Processing/Set' + str(Set)  + '/Datasets/' + DN + '_t2.pickle')
        dataALL['FileName'] = os.path.basename(f)
        

        df1 = df1.append(splitstats, sort = False, ignore_index=True)
        df2 = df2.append(allstats, sort = False, ignore_index=True)
        df3 = df3.append(dataALL, sort = False, ignore_index = True)
        df4 = df4.append(pushstats, sort = False, ignore_index = True)
     
        
    CompName = 'Particle: ' + str(info[1]) + ', Power: ' + str(info[3])  + 'mW, Ptile: > ' + str(info[4]) + '%'
 
    df0.columns = ['FileName', 'pName', 'wLength', 'power', 'ptile']
    df1 = df1.sort_values(by=['Name', 'FileName']).reset_index(drop=True)
    df2 = df2.sort_values(by=['Name', 'FileName']).reset_index(drop=True)
    df3 = df3.sort_values(by=['FileName']).reset_index(drop=True)
    df4 = df4.sort_values(by=['FileName']).reset_index(drop=True)
    
    dfsplit1 = df1[df1['Name'] == 'FFLI'].reset_index(drop=True)
    dfsplit1a = pd.merge(dfsplit1, df0, on = 'FileName')
    
    dfsplit2 = df2[df2['Name'] == 'FA'].reset_index(drop=True)
    dfsplit2a = pd.merge(dfsplit2, df0, on = 'FileName')
    
    dfsplit3 = pd.merge(df3, df0, on = 'FileName')
    
    dfsplit3 = dfsplit3.dropna()

    # dfsplit4 = df4[df4['Name'] == 'PI'].reset_index(drop=True)
    # dfsplit4a = pd.merge(dfsplit4, df0, on = 'FileName')
    
    dfsplit4a = pd.merge(df4, df0, on = 'FileName')

    
    dfsplit1b = []
    dfsplit2b = []
    dfsplit3b = []
    dfsplit4b = []
    dfsplit5b = []
    
    grouped1 = dfsplit1a.groupby("power")
    grouped2 = dfsplit2a.groupby("power")
  
    grouped4 = dfsplit4a.groupby("power")
     
    powers = dfsplit1a['power'].unique().tolist()
    wLengths = dfsplit1a['wLength'].unique().tolist()
    
    for i in powers:
        group1 = grouped1.get_group(i)
        group2 = grouped2.get_group(i)

        group4 = grouped4.get_group(i)

        dfsplit1b.append(group1)
        dfsplit2b.append(group2)

        dfsplit4b.append(group4)
        
        
    grouped3 = dfsplit3.groupby("wLength")
    grouped5 = dfsplit4a.groupby("wLength")
    
    
    for j in wLengths:
        group3 = grouped3.get_group(j) 
        group5 = grouped5.get_group(j)         

        dfsplit3b.append(group3)
        dfsplit5b.append(group5)
    

    # grouped4 = dfsplit3b[0].groupby("wLength")
    # grouped5 = dfsplit3b[1].groupby("wLength")
     
    #for j in wLengths:

    #     group5 = grouped5.get_group(j)
        
    #     dfsplit5b.append(group5)
        
     


    dfsplit1b[0] = dfsplit1b[0].reset_index()
    dfsplit1b[1] = dfsplit1b[1].reset_index()

    
    # fig,(ax, ax2) = plt.subplots(1,2, sharey=True, gridspec_kw={'width_ratios': [1, 15]}, facecolor='w', dpi = 300, constrained_layout=True)  
    fig,ax2 = plt.subplots(dpi = 300, constrained_layout=True)  

    fig.suptitle('Experimental Measurement of the \n Scattering Force, Wavelength Dependence' )
    # ax.errorbar(wLengths[0], abs(dfsplit1b[0][stat])[0], yerr = 1.96*dfsplit1b[0]['StD'][0]/(np.sqrt(dfsplit1b[0]['N'][0])), fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
    # ax.errorbar(wLengths[0], abs(dfsplit1b[1][stat])[0], yerr = 1.96*dfsplit1b[1]['StD'][0]/(np.sqrt(dfsplit1b[1]['N'][0])), fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    # ax2.errorbar(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10], yerr = (1.96*dfsplit1b[0]['StD'][1:10]/(np.sqrt(dfsplit1b[0]['n'][1:10]))), fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
    # ax2.errorbar(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10], yerr = (1.96*dfsplit1b[1]['StD'][1:10]/(np.sqrt(dfsplit1b[1]['n'][1:10]))), fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    # ax.scatter(wLengths[0], abs(dfsplit1b[0][stat])[0], marker = 'o', label =  str(int(powers[0])) + ' mW')
    # ax.scatter(wLengths[0], abs(dfsplit1b[1][stat])[0], marker = 'o', label =  str(int(powers[1])) + ' mW')


    
    ax = ax2.twinx()
    
    df = pd.read_excel(r'Z:\labdata\project_optical_sorting\jimbo\jimbo_fluor.xlsx')
    dfT = df.transpose()
    ax.plot(dfT[1:][0], smooth((dfT[1:][1]-800),5)/smooth((dfT[1:][1]-800),5).max(), linestyle="-", color = 'black')
    

    ax2.plot(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10], marker = (5,2), markersize = 8, linewidth = 1.5, linestyle=":", label = str(int(powers[0])) + ' mW', color = palette_tab10[0])
    ax2.plot(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10], marker = 'o', linewidth = 1.5, linestyle="--", label = str(int(powers[1])) + ' mW', color = palette_tab10[3])

    
    #ax2.fill_between(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10] + dfsplit1b[1]['StD'][1:10],  abs(dfsplit1b[1][stat])[1:10] - dfsplit1b[1]['StD'][1:10], alpha=0.5, color = palette_tab10[3])
    #ax2.fill_between(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10] + dfsplit1b[0]['StD'][1:10],  abs(dfsplit1b[0][stat])[1:10] - dfsplit1b[0]['StD'][1:10], alpha=0.2, color = palette_tab10[0])

    ax2.fill_between(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10] + (1.96*dfsplit1b[0]['StD'][1:10]/(np.sqrt(dfsplit1b[0]['n'][1:10]))),  abs(dfsplit1b[0][stat])[1:10] - (1.96*dfsplit1b[0]['StD'][1:10]/(np.sqrt(dfsplit1b[0]['n'][1:10]))), alpha=0.2)
    ax2.fill_between(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10] + (1.96*dfsplit1b[1]['StD'][1:10]/(np.sqrt(dfsplit1b[1]['n'][1:10]))),  abs(dfsplit1b[1][stat])[1:10] - (1.96*dfsplit1b[1]['StD'][1:10]/(np.sqrt(dfsplit1b[0]['n'][1:10]))), alpha=0.2)
    # #ax2.fill_between([value * 1e9 for value in wLengths], abs(dfsplit1b[0][stat]) + (1.96*dfsplit1b[0]['StD']/(np.sqrt(dfsplit1b[0]['N']))),  abs(dfsplit1b[0][stat]) - (1.96*dfsplit1b[0]['StD']/(np.sqrt(dfsplit1b[0]['N']))), alpha=0.2)

    #ax2.errorbar(wLengths, abs(dfsplit1b[1][stat]), yerr = dfsplit1b[1]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    #plt.errorbar(np.arange(len(wLengths)), abs(dfsplit1b[2][stat]), yerr = dfsplit1b[2]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[2])
    ax2.legend(loc = 'best', title = 'Power')

    # ax.set_xticks([wLengths[0]])
    # ax2.set_xticks(wLengths[1:10])
    
    # ax.tick_params(axis='x', labelrotation=45)
    # ax2.tick_params(axis='x', labelrotation=45)
   
    # ax.set_xticks([wLengths[0]])
    
    ax2.set_xlabel('Wavelength (nm)'); ax2.set_ylabel('Force (fN)'); ax.set_ylabel('Normalised Intensity')
    # ax.set_ylabel('Force (fN)')
    ax2.axvline(x=737, ymin = 0, ymax = 1e10, linestyle="-", color = 'black', label = 'SiV ZPL')
    ax2.axvline(x=1200, ymin = 0, ymax = 1e10, linestyle="--", color = 'black', label = 'SiV Spectral Dist.')
    # ax.set_xlim(520,544)
    ax2.set_xlim(720,780)
    ax2.legend(loc = 'best', title = 'Powers')
    # hide the spines between ax and ax2
    # ax.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    #ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')
    # ax2.tick_params(labelleft='off')

    # d = 0.015
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # # ax.plot((1,1), (-d,+d), **kwargs)
    # # ax.plot((1,1),(1-d,1+d), **kwargs)
    
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((0,0), (1-d,1+d), **kwargs)
    # ax2.plot((0,0), (-d,+d), **kwargs)
    fig.savefig(DIR + '/Compiled_Results/' + stat + '_' + str(ptile) + 'FF_Wavelength_Velocity_Compiled.png')
    
    

    dfsplit4b[0] = dfsplit4b[0].reset_index()
    dfsplit4b[1] = dfsplit4b[1].reset_index()
      
    
    
    
    # fig,(ax, ax2) = plt.subplots(1,2, sharey=True, gridspec_kw={'width_ratios': [1, 15]}, facecolor='w', dpi = 300, constrained_layout=True)  
    fig,ax2 = plt.subplots(dpi = 300, constrained_layout=True)  
      
    fig.suptitle('Experimental Measurement of the \n Scattering Force, Wavelength Dependence' )
    # ax.errorbar(wLengths[0], abs(dfsplit1b[0][stat])[0], yerr = 1.96*dfsplit1b[0]['StD'][0]/(np.sqrt(dfsplit1b[0]['N'][0])), fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
    # ax.errorbar(wLengths[0], abs(dfsplit1b[1][stat])[0], yerr = 1.96*dfsplit1b[1]['StD'][0]/(np.sqrt(dfsplit1b[1]['N'][0])), fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    # ax2.errorbar(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10], yerr = (1.96*dfsplit1b[0]['StD'][1:10]/(np.sqrt(dfsplit1b[0]['n'][1:10]))), fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
    # ax2.errorbar(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10], yerr = (1.96*dfsplit1b[1]['StD'][1:10]/(np.sqrt(dfsplit1b[1]['n'][1:10]))), fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    # ax.scatter(wLengths[0], abs(dfsplit1b[0][stat])[0], marker = 'o', label =  str(int(powers[0])) + ' mW')
    # ax.scatter(wLengths[0], abs(dfsplit1b[1][stat])[0], marker = 'o', label =  str(int(powers[1])) + ' mW')
      
      
    
    ax = ax2.twinx()
    
    df = pd.read_excel(r'Z:\labdata\project_optical_sorting\jimbo\jimbo_fluor.xlsx')
    dfT = df.transpose()
    #ax.plot(dfT[1:][0], smooth((dfT[1:][1]-800),5)/smooth((dfT[1:][1]-800),5).max(), linestyle="-", color = 'black')
    
    
    wLengths1 = np.arange(600,850,0.1)
    ax.plot(wLengths1, Lorentz(wLengths1, 1e5, (737e-9)**2*(Gam0+gam)/(2*math.pi*c)*1e9, 737)/(Lorentz(wLengths1, 1e5, (737e-9)**2*(Gam0+gam)/(2*math.pi*c)*1e9, 737).max()), color = 'black', linestyle="--")

      
    ax2.plot(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10], marker = (5,2), markersize = 8, linewidth = 1.5, linestyle=":", label = str(int(powers[0])) + ' mW', color = palette_tab10[0])
    ax2.plot(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10], marker = 'o', linewidth = 1.5, linestyle="--", label = str(int(powers[1])) + ' mW', color = palette_tab10[3])
      
    
    ax2.fill_between(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10] + dfsplit1b[1]['StD'][1:10],  abs(dfsplit1b[1][stat])[1:10] - dfsplit1b[1]['StD'][1:10], alpha=0.5, color = palette_tab10[3])
    ax2.fill_between(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10] + dfsplit1b[0]['StD'][1:10],  abs(dfsplit1b[0][stat])[1:10] - dfsplit1b[0]['StD'][1:10], alpha=0.2, color = palette_tab10[0])
    
    #ax2.fill_between(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10] + dfsplit1b[1]['StD'][1:10]/dfsplit1b[1][stat][1:10],  abs(dfsplit1b[1][stat])[1:10] - dfsplit1b[1]['StD'][1:10]/dfsplit1b[1][stat][1:10], alpha=0.5, color = palette_tab10[3])
    #ax2.fill_between(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10] + dfsplit1b[0]['StD'][1:10]/dfsplit1b[1][stat][1:10],  abs(dfsplit1b[0][stat])[1:10] - dfsplit1b[0]['StD'][1:10]/dfsplit1b[1][stat][1:10], alpha=0.2, color = palette_tab10[0])
      
    
    # ax2.fill_between(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10] + (1.96*dfsplit1b[0]['StD'][1:10]/(np.sqrt(dfsplit1b[0]['n'][1:10]))),  abs(dfsplit1b[0][stat])[1:10] - (1.96*dfsplit1b[0]['StD'][1:10]/(np.sqrt(dfsplit1b[0]['n'][1:10]))), alpha=0.2)
    # ax2.fill_between(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10] + (1.96*dfsplit1b[1]['StD'][1:10]/(np.sqrt(dfsplit1b[1]['n'][1:10]))),  abs(dfsplit1b[1][stat])[1:10] - (1.96*dfsplit1b[1]['StD'][1:10]/(np.sqrt(dfsplit1b[0]['n'][1:10]))), alpha=0.2)
    # #ax2.fill_between([value * 1e9 for value in wLengths], abs(dfsplit1b[0][stat]) + (1.96*dfsplit1b[0]['StD']/(np.sqrt(dfsplit1b[0]['N']))),  abs(dfsplit1b[0][stat]) - (1.96*dfsplit1b[0]['StD']/(np.sqrt(dfsplit1b[0]['N']))), alpha=0.2)
      
    #ax2.errorbar(wLengths, abs(dfsplit1b[1][stat]), yerr = dfsplit1b[1]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    #plt.errorbar(np.arange(len(wLengths)), abs(dfsplit1b[2][stat]), yerr = dfsplit1b[2]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[2])
    ax2.legend(loc = 'best', title = 'Power')
      
    # ax.set_xticks([wLengths[0]])
    # ax2.set_xticks(wLengths[1:10])
    
    # ax.tick_params(axis='x', labelrotation=45)
    # ax2.tick_params(axis='x', labelrotation=45)
    ax2.legend(loc = 'best', title = 'Powers')
    # ax.set_xticks([wLengths[0]])
    
    ax2.set_xlabel('Wavelength (nm)'); ax2.set_ylabel('Force (fN)'); ax.set_ylabel('Normalised Intensity')
    # ax.set_ylabel('Force (fN)')
    ax2.axvline(x=737, ymin = 0, ymax = 1e10, linestyle="-", color = 'black', label = 'SiV ZPL')
    ax2.axvline(x=1200, ymin = 0, ymax = 1e10, linestyle="--", color = 'black', label = 'SiV Spectral Dist.')
    # ax.set_xlim(520,544)
    ax2.set_xlim(720,780)
    
    # hide the spines between ax and ax2
    # ax.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    #ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')
    # ax2.tick_params(labelleft='off')
      
    # d = 0.015
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # # ax.plot((1,1), (-d,+d), **kwargs)
    # # ax.plot((1,1),(1-d,1+d), **kwargs)
    
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((0,0), (1-d,1+d), **kwargs)
    # ax2.plot((0,0), (-d,+d), **kwargs)
    fig.savefig(DIR + '/Compiled_Results/' + stat + '_' + str(ptile) + 'FF_Wavelength_Velocity_Compiled.png')
    
    
      
    dfsplit4b[0] = dfsplit4b[0].reset_index()
    dfsplit4b[1] = dfsplit4b[1].reset_index()



    # fig2,(ax,ax2) = plt.subplots(1,2, sharey=True, gridspec_kw={'width_ratios': [1, 15]}, facecolor='w', dpi = 1200, constrained_layout=True)  
    # fig2.suptitle('Gaussian Fitting - Wavelength and Sigma \n' + CompName)
    # ax.errorbar(wLengths[0], abs(dfsplit4b[0]['Width'])[0]*2.4, yerr = dfsplit4b[0]['R2'][0], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
    # ax.errorbar(wLengths[0], abs(dfsplit4b[1]['Width'])[0]*2.4, yerr = dfsplit4b[1]['R2'][0], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    # ax2.errorbar(wLengths[1:10], abs(dfsplit4b[0]['Width'])[1:10]*2.4, yerr = dfsplit4b[0]['std'][1:10], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
    # ax2.errorbar(wLengths[1:10], abs(dfsplit4b[1]['Width'])[1:10]*2.4, yerr = dfsplit4b[1]['R2'][1:10], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
   
    
    # ax2.fill_between([value * 1e9 for value in wLengths], abs(dfsplit1b[1][stat]) + (1.96*dfsplit1b[1]['StD']/(np.sqrt(dfsplit1b[1]['N']))),  abs(dfsplit1b[1][stat]) - (1.96*dfsplit1b[1]['StD']/(np.sqrt(dfsplit1b[1]['N']))), alpha=0.2)
    # ax2.fill_between([value * 1e9 for value in wLengths], abs(dfsplit1b[0][stat]) + (1.96*dfsplit1b[0]['StD']/(np.sqrt(dfsplit1b[0]['N']))),  abs(dfsplit1b[0][stat]) - (1.96*dfsplit1b[0]['StD']/(np.sqrt(dfsplit1b[0]['N']))), alpha=0.2)

    
    # #ax2.errorbar(wLengths, abs(dfsplit1b[1][stat]), yerr = dfsplit1b[1]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    # #plt.errorbar(np.arange(len(wLengths)), abs(dfsplit1b[2][stat]), yerr = dfsplit1b[2]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[2])
    # ax2.legend(loc = 'best', title = 'Power (mW)')
    
    # ax.set_xticks([wLengths[0]])
    # ax2.set_xticks(wLengths[1:10])
    
    # ax.tick_params(axis='x', labelrotation=45)
    # ax2.tick_params(axis='x', labelrotation=45)
    # ax2.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Sigma  ' r'$\mu$m')
    
    # ax.set_xlim(520,544)
    # ax2.set_xlim(715,805)
    # ax.set_ylim(50,300)
    
    # # hide the spines between ax and ax2
    # ax.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # #ax.yaxis.tick_left()
    # # ax.tick_params(labelright='off')
    # # ax2.tick_params(labelleft='off')
    
    # d = 0.015
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # ax.plot((1,1), (-d,+d), **kwargs)
    # ax.plot((1,1),(1-d,1+d), **kwargs)
    
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((0,0), (1-d,1+d), **kwargs)
    # ax2.plot((0,0), (-d,+d), **kwargs)
    # fig2.savefig(DIR + '/Compiled_Results/' + stat + '_' + str(ptile) + 'FF_Wavelength_Width_Compiled.png')

    
    
    
    fig3, ax3 = plt.subplots(2, constrained_layout=True, dpi = 1200)
    fig3.suptitle('Gaussian Fitting - Velocities in parallel with the beam - f(y)')
    x_values = np.arange(0, 1000, 0.1)
    for i in range(len(dfsplit5b)):
        d = dfsplit5b[i].reset_index()
        ax3[0].plot(x_values, gauss(x_values, float(d['Amp'][0]), float(d['Center'][0]), float(d['Width'][0])), label = str(d.iloc[0]['wLength']))
        ax3[1].plot(x_values, gauss(x_values, float(d['Amp'][1]), float(d['Center'][1]), float(d['Width'][1])), label = str(d.iloc[1]['wLength']))
        
    ax3[0].legend(loc = 'best' , ncol=2)
    ax3[0].set_title('Power: ' + str(d.iloc[0]['power']) + ' mW')
    ax3[1].set_title('Power: ' + str(d.iloc[1]['power']) + ' mW')
    
    ax3[0].set_ylim(-100, 2200)
    ax3[1].set_ylim(-100, 2200)
    
    ax3[0].set_ylabel('Velocity  ' r'$\mu$m/s')
    ax3[1].set_xlabel('Vertical Position  ' r'$\mu$m'); ax3[1].set_ylabel('Velocity  ' r'$\mu$m/s')
    
    fig3.savefig(DIR + '/Compiled_Results/' + stat + '_' + str(ptile) +'_FF_Gaussian_Fitting.png')
    
    #ax.errorbar(wLengths[0], abs(dfsplit4b[0]['Width'])[0]*2.4, yerr = dfsplit4b[0]['R2'][0], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
    #ax.errorbar(wLengths, abs(dfsplit1b[1][stat]), yerr = dfsplit1b[1]['StD']◘, fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    #ax2.errorbar(wLengths[1:10], abs(dfsplit4b[0]['Width'])[1:10]*2.4, yerr = dfsplit4b[0]['R2'][1:10], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
   # ax2.errorbar(wLengths, abs(dfsplit1b[1][stat]), yerr = dfsplit1b[1]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    #plt.errorbar(np.arange(len(wLengths)), abs(dfsplit1b[2][stat]), yerr = dfsplit1b[2]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[2])
    #ax2.legend(loc = 'best', title = 'Power (mW)')
    
 


             
    # figX, axX = plt.subplots(dpi = 1200)    
    # figX.suptitle('PI Velocity [' + stat + '] - Wavelength and Power')
    
    # plt.errorbar(np.arange(len(wLengths)), abs(dfsplit4b[0][stat]), yerr = dfsplit4b[0]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[0])
    # #plt.errorbar(np.arange(len(wLengths)), abs(dfsplit4b[1][stat]), yerr = dfsplit4b[1]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[1])
    # #plt.errorbar(np.arange(len(wLengths)), abs(dfsplit4b[2][stat]), yerr = dfsplit4b[2]['StD'], fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = powers[2])
    # plt.legend(loc = 'best', title = 'Power (mW)')
    # plt.xticks(np.arange(len(wLengths)), wLengths)
    # plt.xlabel('Wavelength (nm)'); plt.ylabel('Velocity  ' r'$\mu$m/s')
    

    # figX.savefig(DIR + '/Compiled_Results/' + stat + 'PI_Wavelength_Velocity_Compiled.png')



   #  axs1[0].errorbar(dfsplit2b[1]['wLength'], dfsplit2b[1]['Mean'], yerr = dfsplit2b[1]['StD'], color = 'C0', fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = 'FA')
   # # axs1[0].fill_between(dfsplit1b[1]['wLength'], dfsplit1b[1]['Mean'] + dfsplit1b[1]['StD'], dfsplit1b[1]['Mean'] - dfsplit1b[1]['StD'], alpha = 0.1)
   #  axs1[0].set_ylabel('Velocity  ' r'$\mu$m/s')

   #  axs1[1].errorbar(dfsplit1b[0]['wLength'], dfsplit1b[0]['Mean'], yerr = dfsplit1b[0]['StD'], color = 'C1', fmt='o', capsize= 5, linewidth =  1.5, linestyle="--", label = 'FFLI')
   #  axs1[1].errorbar(dfsplit2b[0]['wLength'], dfsplit2b[0]['Mean'], yerr = dfsplit2b[0]['StD'], color = 'C0', fmt='o', capsize= 5, linewidth =  1.5, linestyle="--", label = 'FA')
   # # axs1[1].fill_between(dfsplit1b[0]['wLength'], dfsplit1b[0]['Mean'] + dfsplit1b[0]['StD'], dfsplit1b[0]['Mean'] - dfsplit1b[0]['StD'], alpha = 0.1)
   #  axs1[0].set_title('Laser Power: ' + str(powers[1]) + ' mW'); axs1[1].set_title('Laser Power: ' + str(powers[0]) + ' mW');
   #  axs1[1].set_xlabel('Wavelength (nm)');axs1[1].set_ylabel('Velocity  ' r'$\mu$m/s')    
   #  axs1[0].legend(loc = 'best', title = 'Dataset'), axs1[1].legend(loc = 'best', title = 'Dataset')

       
     
        
        
        
        
   #  fig1, axs1 = plt.subplots(2, constrained_layout=True)
   #  fig1.suptitle('Average Velocity for Each Dataset - Beam Orientation ')
    
   #  axs1[0].errorbar(dfsplit1b[1]['wLength'], dfsplit1b[1]['Mean'], yerr = dfsplit1b[1]['StD'], color = 'C1',fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = 'FFLI')
   #  axs1[0].errorbar(dfsplit2b[1]['wLength'], dfsplit2b[1]['Mean'], yerr = dfsplit2b[1]['StD'], color = 'C0', fmt='o', capsize= 5, linewidth = 1.5, linestyle="--", label = 'FA')
   # # axs1[0].fill_between(dfsplit1b[1]['wLength'], dfsplit1b[1]['Mean'] + dfsplit1b[1]['StD'], dfsplit1b[1]['Mean'] - dfsplit1b[1]['StD'], alpha = 0.1)
   #  axs1[0].set_ylabel('Velocity  ' r'$\mu$m/s')

   #  axs1[1].errorbar(dfsplit1b[0]['wLength'], dfsplit1b[0]['Mean'], yerr = dfsplit1b[0]['StD'], color = 'C1', fmt='o', capsize= 5, linewidth =  1.5, linestyle="--", label = 'FFLI')
   #  axs1[1].errorbar(dfsplit2b[0]['wLength'], dfsplit2b[0]['Mean'], yerr = dfsplit2b[0]['StD'], color = 'C0', fmt='o', capsize= 5, linewidth =  1.5, linestyle="--", label = 'FA')
   # # axs1[1].fill_between(dfsplit1b[0]['wLength'], dfsplit1b[0]['Mean'] + dfsplit1b[0]['StD'], dfsplit1b[0]['Mean'] - dfsplit1b[0]['StD'], alpha = 0.1)
   #  axs1[0].set_title('Laser Power: ' + str(powers[1]) + ' mW'); axs1[1].set_title('Laser Power: ' + str(powers[0]) + ' mW');
   #  axs1[1].set_xlabel('Wavelength (nm)');axs1[1].set_ylabel('Velocity  ' r'$\mu$m/s')    
   #  axs1[0].legend(loc = 'best', title = 'Dataset'), axs1[1].legend(loc = 'best', title = 'Dataset')

    
   #  fig1a, axs1a = plt.subplots(2, constrained_layout=True)
   #  fig1a.suptitle('Average Velocity for Each Dataset - Beam Orientation ')

   #  dflow = pd.concat([dfsplit1b[1], dfsplit2b[1]])
   #  dfhigh = pd.concat([dfsplit1b[0], dfsplit2b[0]])

   #  dflow.groupby(['wLength','Name'])['Mean'].sum().unstack().plot(kind="bar", ax = axs1a[0], alpha = 0.4, sharex='col', grid = True)
   #  dfhigh.groupby(['wLength','Name'])['Mean'].sum().unstack().plot(kind="bar", ax = axs1a[1], alpha = 0.4, sharex='col', grid = True)
   #  axs1a[0].set_title('Laser Power: ' + str(powers[1]) + ' mW'); axs1a[1].set_title('Laser Power: ' + str(powers[0]) + ' mW')
   #  axs1a[1].set_xlabel('Wavelength (nm)');axs1a[1].set_ylabel('Velocity  ' r'$\mu$m/s')
   #  axs1a[0].legend(loc = 'best', title = 'Dataset'), axs1a[1].legend(loc = 'best', title = 'Dataset')


    
   #  fig2, axs2 = plt.subplots(2, constrained_layout=True)
   #  fig2.suptitle('Distribution of Velocity - Parrallel with Beam')
   #  fig3, axs3 = plt.subplots(2, constrained_layout=True)
   #  fig3.suptitle('Distribution of Velocity - Perpendicular with Beam')
    
   #  for k in dfsplit5b:
   #      axs2[1].hist(k.u, density = True, bins= nBins , alpha = 0.6, histtype='step',  fill=False, label=k.wLength)
   #      axs3[1].hist(k.v, density = True, bins= nBins , alpha = 0.6, histtype='step',  fill=False, label=k.wLength)
        
   #  for j in dfsplit4b:
   #      axs2[0].hist(j.u, density = True, bins= nBins ,  alpha = 0.6, histtype='step',  fill=False, label=j.wLength)
   #      axs3[0].hist(j.v, density = True, bins= nBins ,  alpha = 0.6, histtype='step',  fill=False, label= j.wLength)
        
   #  axs2[1].set_ylabel('Counts %'), axs2[1].set_xlabel('Velocity  ' r'$\mu$m/s'), axs2[1].legend(loc = 1, title = 'WaveLengths')
   #  axs2[1].set_title('Laser Power: ' + str(powers[0]) + ' mW')

    
   #  axs3[1].set_ylabel('Counts %'), axs3[1].set_xlabel('Velocity  ' r'$\mu$m/s'), axs3[1].legend(loc = 1, title = 'WaveLengths')
   #  axs3[1].set_title('Laser Power: ' + str(powers[0]) + ' mW')

        
   #  axs2[0].set_ylabel('Counts %'), axs2[0].legend(loc = 1, title = 'WaveLengths')
   #  axs2[0].set_title('Laser Power: ' + str(powers[1]) + ' mW')


   #  axs3[0].set_ylabel('Counts %'), axs3[0].legend(loc = 1, title = 'WaveLengths')
   #  axs3[0].set_title('Laser Power: ' + str(powers[1]) + ' mW')
    
    # fig,(ax, ax2) = plt.subplots(1,2, sharey=True, gridspec_kw={'width_ratios': [1, 15]}, facecolor='w', dpi = 300, constrained_layout=True)  
    fig,ax2 = plt.subplots(dpi = 300, constrained_layout=True, figsize = (3,3))  
    fig.suptitle('B. Experimental Force' )

    
    
    df = pd.read_excel(r'Z:\labdata\project_optical_sorting\jimbo\jimbo_fluor.xlsx')
    dfT = df.transpose()
    

    # ax2.plot(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10], marker = (5,2), markersize = 8, linewidth = 1.5, linestyle=":", label = str(int(powers[0])) + ' mW', color = palette_tab10[0])
    ax2.plot(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10], marker = 'o', linewidth = 1.5, linestyle="--", label = str(int(powers[1])) + ' mW', color = palette_tab10[3])

    
    ax2.fill_between(wLengths[1:10], abs(dfsplit1b[1][stat])[1:10] + dfsplit1b[1]['StD'][1:10],  abs(dfsplit1b[1][stat])[1:10] - dfsplit1b[1]['StD'][1:10], alpha=0.5, color = palette_tab10[3])
    #ax2.fill_between(wLengths[1:10], abs(dfsplit1b[0][stat])[1:10] + dfsplit1b[0]['StD'][1:10],  abs(dfsplit1b[0][stat])[1:10] - dfsplit1b[0]['StD'][1:10], alpha=0.2, color = palette_tab10[0])


    # ax.set_xticks([wLengths[0]])
    # ax2.set_xticks(wLengths[1:10])
    
    # ax.tick_params(axis='x', labelrotation=45)
    # ax2.tick_params(axis='x', labelrotation=45)
    # ax.set_xticks([wLengths[0]])
    
    ax2.set_xlabel('Wavelength (nm)'); ax2.set_ylabel('Force (fN)')

    # ax.set_xlim(520,544)
    ax2.set_xlim(720,780)

    plt.show()
    
    # Saves the figures to the plots folder 
   
    # fig1a.savefig(Location + '/Compiled_Results/1a.Average_Velocity_for_Each_Dataset-Beam_Orientation-Grouped_Bar.png')
    # fig2.savefig(Location + '/Compiled_Results/2.Distribution_of_Velocity-Parrallel_with_Beam.png')
    # fig3.savefig(Location + '/Compiled_Results/3.Distribution_of_Velocity-Perpendicular_to_Beam.png') 
    
