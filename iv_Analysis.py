  

import glob
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy import stats
import trackpy as tp
import seaborn as sns
# sns.set_palette("RdBu")
sns.set_palette("Reds",2)
palette_tab10 = sns.color_palette("Reds",2)
palette = sns.color_palette([palette_tab10[0], palette_tab10[1]])

# Generates a dataframe containing ['FileName', 'Name','Max', 'Min', 'Mean', 'StD'] for a given histogram.

def Stats(binN, dataN, Ident):
    mids = pd.Series(0.5*(binN[1:] + binN[:-1]))  
    maxV = binN.max()
    minV = binN.min()
    probs = dataN / np.sum(dataN) 
    mean = np.sum(probs * mids)
    median = np.median(binN)
    std = np.sqrt(np.sum(probs * (mids - mean)**2))
    meanround = np.around(mean, 2)
    stdround = np.around(std, 2)
    skew = stats.skew(dataN)
    Name = (str(Ident))
    stats1 = [[Name, maxV, minV, meanround, median, stdround, skew]]
    df = pd.DataFrame(stats1, columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD', 'Skew'])
    return df

# Location = 'Z:\\labdata\\project_optical_sorting\\Data/20220415\\ShazND_740nm_200mW\\2022_04_15_13_54_52'
# DN = os.path.basename(os.path.dirname(Location))





Input = 'Python'
nBins = 50
center = 360
bWidth = 12*1e-6
smoo = 30
ptile = 5
cutoff = 30
roundV = 2
fits = True
lxdis = 40e-6



Location = 'D:\\Data/20220517/ShazND_735nm_050mW/2022_05_17_16_41_34/'
DN = os.path.basename(os.path.dirname(os.path.dirname(Location)))
DIR = os.path.dirname(os.path.dirname(os.path.dirname(Location)))

Set = 0

def Analysis(Location, Set, DN, DIR, Input, lxdis, nBins,  center, bWidth, smoo, ptile, cutoff, roundV, fits):
    
    
    # Fucntions for smoothing and Fitting the Gaussian and Lorentzian
    
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def gauss(x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    def gauss2(x, A, A2, x0, x02, sigma, sigma2):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + A2 * np.exp(-(x - x02) ** 2 / (2 * sigma2 ** 2))
    
    def lorentzian(x, a, x0, gam):
        return a * gam**2 / ( gam**2 + ( x - x0 )**2)


    below = center - bWidth;
    above = center + bWidth;
    
    quant = (100 - ptile)/100

    



    # Imported Datasets (Variables and Tracking data)
    
    # if (Input == 'ImageJ'):
        
    #     ## Data from ImageJ plugin mTrackj
    #     # Export the manual tracks by clicking "measure" key on the mTrackj control panel
    #     # Save Points.csv to the Data Folder (Location)
    #     # Save with filename ImageJ_Points.csv
     
    #     motion = pd.read_csv(Location + '/' + FileName + '/' + Experiment + '/Datasets/ImageJ_Points.csv')#, names =["u", "v", "x", "y", "t", "tr"])
    #     try:
    #         motion = motion.drop(columns=['Nr', 't [sec]', 'I [val]', 'Len [pixel]', 'D2S [pixel]', 'D2R [pixel]', 'D2P [pixel]', 'v [pixel/sec]', '? [deg]', '?? [deg]'])
    #     except:
    #         pass
        
    #     motion = motion.rename(columns={"TID": "particle", "PID": "frame", "x [pixel]" : "x", "y [pixel]" : "y"}, errors="raise")
        
    #     motion['xdis'] = motion.x.diff()
    #     motion['ydis'] = motion.y.diff()
        
    #     motion['xdis'] = motion['xdis'].where(motion.particle.diff() == 0, float('nan'))
    #     motion['ydis'] = motion['ydis'].where(motion.particle.diff() == 0, float('nan'))
        
    #     motion['u'] = (motion.x.diff()/motion.frame.diff()) 
    #     motion['v'] = (motion.y.diff()/motion.frame.diff())  
        
    #     motion['u'] = motion['u'].where(motion.particle.diff() == 0, float('nan'))
    #     motion['v'] = motion['v'].where(motion.particle.diff() == 0, float('nan'))
       
    #     motion = motion[["particle", "frame", "x", "y", "u", "v"]]
    #     motion = motion.sort_values(by=['y'])
        
        
    # elif (Input == 'Matlab'):
        
    #     ## Data from matlab
    #     # In Matlab > myCompiler('2019\MATLAB\Data\20190528\YG\',200)
    #     # Run through MyCompiler
    #     # > motion = [u,v,x,y,t,tr] 
    #     # > csvwrite('file.csv',motion)
        
    #     motion = pd.read_csv(Location + '/' + FileName, names =["u", "v", "x", "y", "t", "tr"])
    #     motion = motion.rename(columns={"t": "frame", "tr": "particle"}, errors="raise")
    #     motion = motion[["particle", "frame", "x", "y", "u", "v"]]
    #     motion = motion.sort_values(by=['y'])
        
    # elif (Input == 'Python'):
        
    #     ## Data from OB_Tracker.py
    
    t2 = pd.read_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_t2.pickle')
    
    t2 = t2.drop(columns=['txdis', 'tydis'])
    
    t2.index.name = None
    t2 = t2.sort_values(by=['particle', 'frame'])
    

    t2 = t2[["particle", "frame", "time", "x", "y", "mass", "size", "ecc"]]
    
    t2['x'] = (t2['x']*0.1 - 50)*1e-6
    # t2['y'] = (t2['y']*0.1 - 50)*1e-6
    t2['y'] = (t2['y']*0.1 - 44.5)*1e-6*1.1
      
    t2['xdis'] = t2.x.diff()
    t2['ydis'] = t2.y.diff()
    
    t2['xdis'] = t2['xdis'].where(t2.particle.diff() == 0, float('nan'))
    t2['ydis'] = t2['ydis'].where(t2.particle.diff() == 0, float('nan'))
    
    t2['u'] = (t2.x.diff()/t2.time.diff()) 
    t2['v'] = (t2.y.diff()/t2.time.diff())  
    
    t2['u'] = t2['u'].where(t2.particle.diff() == 0, float('nan'))
    t2['v'] = t2['v'].where(t2.particle.diff() == 0, float('nan'))
    
    t2['fx'] = t2['u']*(6*math.pi*0.001*120e-9)*1e15
    t2['fy'] = t2['v']*(6*math.pi*0.001*120e-9)*1e15
    
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
    
    t4 = pd.merge(t2, t3, on = 'particle')
    lxdis = 4.5e-5
    t4 = t4[(t4['txdis'] > lxdis)&(abs(t4['xdis'])<5e-5)&(abs(t4['xdis'])>-5e-5)&(abs(t4['ydis'])<1e-5)&(abs(t4['ydis'])>-1e-5)]
    
    
    motion = t4
    motion = motion.sort_values(by=['y'])
    
    FileList = glob.glob (Location + '/Frames_Edited/*.png')  
    FileList.sort()
    backgroundimg = cv2.imread(FileList[-1], cv2.IMREAD_GRAYSCALE)
    
    # t4['x']=t4['x']*1e6
    # t4['y']=t4['y']*1e6
    # fig, ax = plt.subplots(dpi = 300, figsize = [6,6])
    
    # plt.figure(dpi = 1200)
    # plt.title(DN + ' - Quiver of Velocity Vectors')
    # # plt.imshow(backgroundimg, cmap = 'gray')
    # tp.plot_traj(t4);
    
    
    # frameW = 100e-6
    # waist0 = 5e-6
    # t4['zscale']=t4['x']*1e6*1.1
    # t4['yscale']=(t4['y']*1e6*-1.2) - 10
    # fig, ax = plt.subplots(dpi = 300, figsize = [6,6])

    # for key, grp in t4.groupby(['particle']):
    #     ax = grp.plot(ax=ax, kind='line', x='zscale', y='yscale', linewidth=1.0, label=key, legend = False)
    #     ax.set_title('Experimental Trajectories')
    #     ax.set_xlabel('Z ($\mu$m)')
    #     ax.set_ylabel('Y ($\mu$m)')
    #     ax.set_xlim([-frameW/2*1e6+5,frameW/2*1e6-5])
    #     ax.set_ylim([-frameW/2*1e6+5,frameW/2*1e6-5])
    #     ax.axhline(waist0*1e6, linestyle = 'dashed', color = 'k')
    #     ax.axhline(-waist0*1e6,  linestyle ='dashed', color = 'k')   
    
    
    motion = motion.dropna()
    motion['smoothu'] = smooth(motion.fx, smoo)
    motion['smoothv'] = smooth(motion.fy, smoo)
    
    motion['scy'] = motion.y*1e6

    
    
    # Average velocity (perpendicular/ parallel) relative to vertical position.
    
    DN = (DN + '_' + str(ptile))

    
    # Plots the Parallel Velocity with fits and beam position
    fig1, axs = plt.subplots(2, constrained_layout=True, dpi = 300, figsize = (7,5))
    axs[0].plot(motion.scy, motion.smoothu, label = 'Experimental Force')
    axs[0].set_title('Experimental Force Profile - Parallel to Beam')
    axs[0].set_xlim([-47.5,47.5])
    axs[0].set_ylabel('Force (fN)'), axs[0].set_xlabel('Vertical Position (um)'), axs[0].legend(loc = 1)
    
    # Plots the Perpendicular Velocity with fits and beam position
    axs[1].plot(motion.scy, motion.smoothv - motion.smoothv.mean(), label = 'Experimental Force')
    axs[1].set_title('Experimental Force Profile - Perpendicular to Beam')
    axs[1].set_xlim([-47.5,47.5])
    axs[1].set_ylim([-50,50])
    
    axs[1].set_ylabel('Force (fN)'), axs[1].set_xlabel('Vertical Position (um)'), axs[1].legend(loc = 1)


    # Uses the pararllel Velocity plot to find the beam position by fitting a gaussian to the previous plot.
    
    if fits == True:
            
        # Fit the Gaussian and Lorentzian to the Data
    
        params, params_cov = op.curve_fit(gauss, motion.y, motion.fx, p0=[abs(motion.fx).max(), motion.y.mean(), motion.y.std()]) # bounds=([1e13,1e4,1e3],[1e16,1e7,1e5]))
        params1, params1_cov = op.curve_fit(lorentzian, motion.y, motion.fx, p0=[abs(motion.fx).max(), motion.y.mean(), motion.y.std()]) # bounds=([1e13,1e4,1e3],[1e16,1e7,1e5]))
    
        #RSquared    
        residuals = motion.fx - gauss(motion.y, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((motion.fx-np.mean(motion.fx))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        params = np.append(params, r_squared)  
        
        #R_Squared
        residuals1 = motion.fx - lorentzian(motion.y, *params1)
        ss_res1 = np.sum(residuals1**2)
        ss_tot1 = np.sum((motion.fx-np.mean(motion.fx))**2)
        r_squared1 = 1 - (ss_res1 / ss_tot1)
        
        params1 = np.append(params1, r_squared1)  
        
            
        
        # Finds the focus of the beam, assumes the beam boundaries
           
        center = params[1]*1e6
        bWidth = abs(params[2] * 2)
        below = center - 0.5*bWidth*1e6
        above = center + 0.5*bWidth*1e6
                
        axs[0].plot(motion.scy, gauss(motion.y, params[0], params[1], params[2]), label = 'Gaussian Fit')
        # axs[0].plot(motion.scy, lorentzian(motion.y, params1[0], params1[1], params1[2]), label = 'Lorentzian Fit')
        axs[0].plot([below, below], [motion.smoothu.min(),motion.smoothu.max()], linestyle = "--", color='k', label = 'Beam Position')
        axs[0].plot([above, above], [motion.smoothu.min(),motion.smoothu.max()], linestyle = "--", color='k')
        axs[1].axhline((motion.smoothv - motion.smoothv.mean()).mean(), label = 'Average Value', color = palette_tab10[1])
        axs[1].plot([below, below], [-50,50], linestyle = "--", color='k', label = 'Beam Position')
        axs[1].plot([above, above], [-50,50], linestyle = "--", color='k')


        axs[0].legend(loc = 1), axs[1].legend(loc = 1)
        
    else:
        pass
    

           
    # print(params[0], params[1], abs(params[2])*2)
    # print(motion.smoothv.mean())
   

          
    # Seperates the Data into Inside and Outside the Beam 
    
    beamI = motion[(motion['scy'] >= below) & (motion['scy'] <= above)]
    beamBelow = motion[(motion['scy'] <= below)] 
    beamAbove = motion[(motion['scy'] >= above)]
    beamO = pd.concat([beamBelow, beamAbove])
    beamA = motion
    
    
    # Creates Hisogram of the velocities Inside and Outside the Beam
    
    bins1 = np.linspace(beamO.fx.min(), beamO.fx.max(), int(round(1 + 3.3*np.log(len(beamO.fx)))))
    bins2 = np.linspace(beamI.fx.min(), beamI.fx.max(), int(round(1 + 3.3*np.log(len(beamI.fx)))))
    bins3 = np.linspace(beamO.fy.min(), beamO.fy.max(), int(round(1 + 3.3*np.log(len(beamO.fy)))))
    bins4 = np.linspace(beamI.fy.min(), beamI.fy.max(), int(round(1 + 3.3*np.log(len(beamI.fy)))))
    
    
    data1, bins1 = np.histogram(beamO.fx, bins=int(round(1 + 3.3*np.log(len(beamO.fx)))), density = 1)
    data2, bins2 = np.histogram(beamI.fx, bins=int(round(1 + 3.3*np.log(len(beamI.fx)))), density = 1)
    data3, bins3 = np.histogram(beamO.fy, bins=int(round(1 + 3.3*np.log(len(beamO.fy)))), density = 1)
    data4, bins4 = np.histogram(beamI.fy, bins=int(round(1 + 3.3*np.log(len(beamI.fy)))), density = 1)


    widths1 = bins1[:-1] - bins1[1:]
    widths2 = bins2[:-1] - bins2[1:]
    widths3 = bins3[:-1] - bins3[1:]
    widths4 = bins4[:-1] - bins4[1:]

    bincenters1 = np.array([0.5 * (bins1[i] + bins1[i+1]) for i in range(len(bins1)-1)])
    bincenters2 = np.array([0.5 * (bins2[i] + bins2[i+1]) for i in range(len(bins2)-1)])
#    bincenters3 = np.array([0.5 * (bins3[i] + bins3[i+1]) for i in range(len(bins3)-1)])
#    bincenters4 = np.array([0.5 * (bins4[i] + bins4[i+1]) for i in range(len(bins4)-1)])

    LO = Stats(bins1, data1, 'LO')
    LI = Stats(bins2, data2, 'LI')
    FO = Stats(bins3, data3, 'FO')
    FI = Stats(bins4, data4, 'FI')
    
    # Segments the Inside/ Outisde Datsets using a percentile cutoff
    
#    LO['Prob'] = round(100 - stats.percentileofscore(beamO['u'],cutoff),roundV)
#    LI['Prob'] = round(100 - stats.percentileofscore(beamI['u'],cutoff),roundV)
#    FO['Prob'] = round(100 - stats.percentileofscore(beamO['v'],cutoff),roundV)
#    FI['Prob'] = round(100 - stats.percentileofscore(beamI['v'],cutoff),roundV)
    
        
#    FFLO = beamO[(beamO.fx >= beamO.fx.quantile(quant))]
    FFLI = beamI[(beamI.fx >= beamI.fx.quantile(quant))]
#    FFFO= beamO[(beamO.fy >= beamO.fy.quantile(quant))]
#    FFFI = beamI[(beamI.fy >= beamI.fy.quantile(quant))]
    
    # sLO = pd.DataFrame([['LO', LO['Max'][0], LO['Min'][0], LO['Mean'][0], LO['Median'][0], LO['StD'][0]]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD'])
    # sFFLI = pd.DataFrame([['FFLI', FFLI.fx.max(), FFLI.fx.min(), FFLI.fx.mean(), FFLI.fx.median(), FFLI.fx.std(), len(FFLI)]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD', 'n'])
    # sFO = pd.DataFrame([['FO', FO['Max'][0], FO['Min'][0], FO['Mean'][0], FO['Median'][0], FO['StD'][0]]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD'])
    # sFI = pd.DataFrame([['FI', FI['Max'][0], FI['Min'][0], FI['Mean'][0], FI['Median'][0], FI['StD'][0]]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD'])
    
    
           
    sLO = pd.DataFrame([['LO', beamO.fx.max(), beamO.fx.min(), beamO.fx.mean(), beamO.fx.median(), beamO.fx.std(), len(beamO.fx)]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD', 'N'])
    #sLI = pd.DataFrame([['LI', beamI.fx.max(), beamI.fx.min(), beamI.fx.mean(), beamI.fx.median(), beamI.fx.std(), len(beamI.fx)]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD', 'N'])
    sFFLI = pd.DataFrame([['FFLI', FFLI.fx.max(), FFLI.fx.min(), FFLI.fx.mean(), FFLI.fx.median(), FFLI.fx.std(), len(FFLI)]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD', 'n'])
    sFO = pd.DataFrame([['FO', beamO.fy.max(), beamO.fy.min(), beamO.fy.mean(), beamO.fy.median(), beamO.fy.std(), len(beamO.fy)]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD', 'N'])
    sFI = pd.DataFrame([['FI', beamI.fy.max(), beamI.fy.min(), beamI.fy.mean(), beamI.fy.median(), beamI.fy.std(), len(beamI.fy)]], columns = ['Name','Max', 'Min', 'Mean', 'Median', 'StD', 'N'])
    

    DFSplit = pd.concat([LO, LI, FO, FI]).reset_index(drop=True)
    FFSplit = pd.concat([sLO, sFFLI, sFO, sFI]).reset_index(drop=True)
    
      
    # Fits a gaussian and the sum of 2 gaussians to the Histogram data
        
    #paramsa1, paramsa1_cov = op.curve_fit(gauss, bincenters1, data1, p0=[data1.max(), bins1.mean(), bins1.std()])
    #paramsa2, paramsa2_cov = op.curve_fit(gauss, bincenters2, data2, p0=[data2.max(), bins2.mean(), bins2.std()])
#    paramsa3, paramsa3_cov = op.curve_fit(gauss, bincenters3, data3, p0=[data3.max(), bins3.mean(), bins3.std()])
#    paramsa4, paramsa4_cov = op.curve_fit(gauss, bincenters4, data4, p0=[data4.max(), bins4.mean(), bins4.std()])
    
    
    # Plots the Histograms onto 2 subplots.
    

    
    str0 = 'Steps/ Trajectories - Laser All: ' +   str(len(beamA)) +  '/' +  str(len(beamA.particle.unique())) + ', FF' + str(ptile) + ': ' +   str(len(FFLI)) + '/'  + str(len(FFLI.particle.unique()))  
    
    str1 = 'Laser Out' + ' - MeanV: ' + str(LO['Mean'][0]) + ', StD: ' + str(LO['StD'][0]) + ', Range: ' + str(round(LO['Min'][0],roundV)) +  ' ~ ' + str(round(LO['Max'][0],roundV))

    str2 = 'FF' + str(ptile) + ' - MeanV: ' + str(round(FFLI.fx.mean(),roundV)) + ', StD: ' + str(round(FFLI.fx.std(),roundV)) + ', Range: ' + str(round(FFLI.fx.min(),roundV)) +  ' ~ ' + str(round(FFLI.fx.max(),roundV))
#    str3 = 'FF' + str(ptile) + ' - MeanV: ' + str(round(FFFO.fy.mean(),roundV)) + ', StD: ' + str(round(FFFO.fy.std(),roundV)) + ', Range: ' + str(round(FFFO.fy.min(),roundV)) +  ' ~ ' + str(round(FFFO.fy.max(),roundV))
#    str4 = 'FF' + str(ptile) + ' - MeanV: ' + str(round(FFFI.fy.mean(),roundV)) + ', StD: ' + str(round(FFFI.fy.std(),roundV)) + ', Range: ' + str(round(FFFI.fy.min(),roundV)) +  ' ~ ' + str(round(FFFI.fy.max(),roundV))

    # xMin = beamI.fx.min()
    # xMax = beamO.fx.max()
    
    
    fig2, axs2 = plt.subplots(2, constrained_layout=True, dpi = 300, figsize = (7,5))
    axs2[0].hist(beamO.fx, density = True, bins= 15*int(round(1 + 3.3*np.log(len(beamO.fx)))), alpha = 0.5, histtype='step', color = 'gray')
    axs2[0].hist(beamO.fx, density = True, bins= 15*int(round(1 + 3.3*np.log(len(beamO.fx)))), alpha = 1.0, label= 'Outside Beam')
    axs2[0].hist(beamI.fx, density = True, bins= 10*int(round(1 + 3.3*np.log(len(beamI.fx)))), alpha = 0.5, histtype='step', color = 'gray')
    axs2[0].hist(beamI.fx, density = True, bins= 10*int(round(1 + 3.3*np.log(len(beamI.fx)))), alpha = 1.0, label= 'Inside Beam')
    #axs2[0].bar(bins1[1:] + widths1/2, data1, width = widths1, alpha = 0.5, label= 'Parallel and Outside Beam')    
    #axs2[0].bar(bins2[1:] + widths2/2, data2, width = widths2, alpha = 0.5, label= 'Parallel and Inside Beam')
    # axs2[0].text(xMin + 10*conv, data1.max()/4 + data1.max()/ 10, str0, color='C2')
    # axs2[0].text(xMin + 10*conv, data1.max()/2 + data1.max()/ 20, str1, color='C0')
    # axs2[0].text(xMin + 10*conv, data1.max()/2 - data1.max()/ 20, str2,  color='C1')
    axs2[0].set_xlabel('Forces (fN)');
    axs2[0].set_xlim([-100,550])
    axs2[0].set_ylim([0,0.02])
    axs2[0].set_title('Experimental Force Distribution - Parallel to Beam')
    axs2[0].set_ylabel('Counts'), axs2[0].legend(loc = 1)
    

    axs2[1].hist(beamO.fy - beamO.fy.mean(), density = True, bins= 15*int(round(1 + 3.3*np.log(len(beamO.fy)))), alpha = 0.5, histtype='step', color = 'gray')
    axs2[1].hist(beamO.fy - beamO.fy.mean(), density = True, bins= 15*int(round(1 + 3.3*np.log(len(beamO.fy)))), alpha = 1.0, label= 'Outside Beam')
    axs2[1].hist(beamI.fy - beamI.fy.mean(), density = True, bins= 10*int(round(1 + 3.3*np.log(len(beamI.fy)))), alpha = 0.5, histtype='step', color = 'gray')
    axs2[1].hist(beamI.fy - beamI.fy.mean(), density = True, bins= 10*int(round(1 + 3.3*np.log(len(beamI.fy)))), alpha = 1.0, label= 'Inside Beam')
    # axs2[1].bar(bins3[1:] + widths3/2, data3, width = widths3, alpha = 0.5, label= 'Perpendicular and Outside Beam')    
    # axs2[1].bar(bins4[1:] + widths4/2, data4, width = widths4, alpha = 0.5, label= 'Perpendicular and Inside Beam')
#    axs2[1].text(20, data3.max()/2 + data3.max()/ 20, str3,  color='C0')
#    axs2[1].text(20, data3.max()/2 - data3.max()/ 20, str4,  color='C1')
    axs2[1].set_xlim([-100,550])
    axs2[1].set_ylim([0,0.02])
    axs2[1].set_title('Experimental Force Distribution - Perpendicular to Beam')
    axs2[1].legend(loc = 1)
    axs2[1].set_ylabel('Counts'), axs2[1].set_xlabel('Forces (fN)');
    
    if fits == True:        
#        axs2[0].plot(bincenters1, gauss(bincenters1, paramsa1[0], paramsa1[1], paramsa1[2]), label = 'Outside Beam Gaussian Fit')
        #axs2[0].plot(bincenters2, gauss(bincenters2, paramsa2[0], paramsa2[1], paramsa2[2]), label = 'Inside Beam Gaussian Fit')
#        axs2[1].plot(bincenters3, gauss(bincenters3, paramsa3[0], paramsa3[1], paramsa3[2]), label = 'Outside Beam Gaussian Fit')
#        axs2[1].plot(bincenters4, gauss(bincenters4, paramsa4[0], paramsa4[1], paramsa4[2]), label = 'Inside Beam Gaussian Fit')
    #else:
        pass
    

         
#     # Creates Hisogram of the velocities

    bins5 = np.linspace(motion.fx.min(), motion.fx.max(), nBins)
    data5, bins5 = np.histogram(motion.fx, bins=nBins, density = True)
    bincenters5 = np.array([0.5 * (bins5[i] + bins5[i+1]) for i in range(len(bins5)-1)])
    widths5 = bins5[:-1] - bins5[1:]

    bins6 = np.linspace(motion.fy.min(), motion.fy.max(), nBins)
    data6, bins6 = np.histogram(motion.fy, bins=nBins, density = True)
    bincenters6 = np.array([0.5 * (bins6[i] + bins6[i+1]) for i in range(len(bins6)-1)])
    widths6 = bins6[:-1] - bins6[1:]
    
    LA = Stats(bins5, data5, 'LA')
    FA = Stats(bins6, data6, 'FA')
    
#    LA['Prob'] = 100 - stats.percentileofscore(motion['u'],cutoff)
#    FA['Prob'] = 100 - stats.percentileofscore(motion['v'],cutoff)
    
#    LA = motion[(motion.fx >= motion.fx.quantile(quant))]
#    FA = motion[(motion.fy >= motion.fy.quantile(quant))]
    
#    sFFLA = pd.DataFrame([[FileName, 'LA', LA.fx.max(), LA.fx.min(), LA.fx.mean(), LA.fx.std()]], columns = ['FileName', 'Name','Max', 'Min', 'Mean', 'StD'])
#    sFFFA = pd.DataFrame([[FileName, 'FA', FA.fy.max(), FA.fy.min(), FA.fy.mean(), FA.fy.std()]], columns = ['FileName', 'Name','Max', 'Min', 'Mean', 'StD'])
    
#    LA['FF'] = motion[(motion.fx >= motion.fx.quantile(quant))].fx.mean()
#    FA['FF'] = motion[(motion.fy >= motion.fy.quantile(quant))].fy.mean()
    
    DFAll = pd.concat([LA, FA]).reset_index(drop=True)
#    FFAll = pd.concat([sFFLA, sFFFA]).reset_index(drop=True)
    
#     # Fits a gaussian and the sum of 2 gaussians to the Histogram data
   
#     paramsb5, paramsb5_cov = op.curve_fit(gauss, bincenters5, data5, p0=[data5.max(), bins5.mean(), bins5.std()])
#     paramsb6, paramsb6_cov = op.curve_fit(gauss, bincenters6, data6, p0=[data6.max(), bins6.mean(), bins6.std()])

    
#     # Plots the Histograms onto 2 subplots.
   
#     fig3, axs3 = plt.subplots(2, constrained_layout=True, dpi = 1200)
#     fig3.suptitle(DN + ' - Distribution of Velocities')
 
# #    str5 = 'Prob: ' + str(round(LA.Prob[0], roundV)) + ', FF: ' + str(round(LA.FFmean[0], roundV)) + ', Range: ' + str(round(LA.Min[0], roundV)) +  ' : ' + str(round(LA.Max[0], roundV))
# #    str6 = 'Prob: ' + str(round(FA.Prob[0], roundV)) + ', FF: ' + str(round(FA.FFmean[0], roundV)) + ', Range: ' + str(round(FA.Min[0], roundV)) +  ' : ' + str(round(FA.Max[0], roundV))

#     # str5 = 'Laser All' + ' - MeanV: ' + str(LA['Mean'][0]) + ', StD: ' + str(LA['StD'][0]) + ', Range: ' + str(round(LA['Min'][0],roundV)) +  ' ~ ' + str(round(LA['Max'][0],roundV))
#     # str6 = 'Flow All' + '- MeanV: ' + str(FA['Mean'][0]) + ', StD: ' + str(FA['StD'][0]) + ', Range: ' + str(round(FA['Min'][0],roundV)) +  ' ~ ' + str(round(FA['Max'][0],roundV))

    
#     #axs3[0].hist(motion.fx, density = True, bins= nBins , alpha = 0.5, label= 'Parallel All')
#     axs3[0].bar(bins5[1:] + widths5/2, data5, width = widths5, alpha = 0.5, label= 'Parallel All')
# #    axs3[0].text(, 100, FA.to_string(), ha='center', va='center')
#     axs3[0].set_title('Velocities in parallel with the beam - f(y)')
#     # axs3[0].text(motion.fx.min() + 100, data5.max()/2, str5,  color='C0')
#     #axs3[0].set_xlim([motion.fx.min() - 5 , motion.fx.max() + 5])
#     axs3[0].legend(loc = 1); axs3[0].set_ylabel('Counts')
    
#     #axs3[1].hist(motion.fy, density = True, bins= nBins, alpha = 0.5, label= 'Perpendicular All')
#     axs3[1].bar(bins6[1:] + widths6/2, data6, width = widths6, alpha = 0.5, label= 'Perpendicular All')
#     axs3[1].set_title('Velocities at a perpendicular to the beam - f(y)')
#     # axs3[1].text(motion.fx.min() + 100, data6.max()/2, str6,  color='C0')
#     #axs3[1].set_xlim([motion.fx.min() - 5 , motion.fx.max() + 5])
#     axs3[1].legend(loc = 1)
#     axs3[1].set_ylabel('Counts'), axs3[1].set_xlabel('Velocity  ' r'$\mu$m/s');
    
#     if fits == True:        
#        # axs3[0].plot(bincenters5, gauss(bincenters5, paramsb5[0], paramsb5[1], paramsb5[2]), label = 'Gaussian Fit')
#        # axs3[1].plot(bincenters6, gauss(bincenters6, paramsb6[0], paramsb6[1], paramsb6[2]), label = 'Gaussian Fit')
#     #else:
#         pass
     
        
#     # Subtract




#     # min1 = np.minimum(bins1.min(), bins2.min())
#     # max1 = np.maximum(bins1.max(), bins2.max())
    
#     # data7, bins7 = np.histogram(beamO.fx, range =(min1, max1), bins=nBins, density = True)
#     # data8, bins8 = np.histogram(beamI.fx, range =(min1, max1), bins=nBins, density = True)

#     # widths7 = bins7[:-1] - bins7[1:]
#     # widths8 = bins8[:-1] - bins8[1:]

#     # data9 = data8 - data7
#     # data9 = data9.clip(min=0)

    
#     fig4, ax4 = plt.subplots(constrained_layout=True, dpi = 1200)
#     fig4.suptitle(DN + ' - Distribution of Parallel Velocities')

#     xMin = -1000 #motion.fx.min()
#     xMax = 3000 #motion.fx.max()
    
#     # axs2[0].hist(beamO.fx, density = True, bins= nBins , alpha = 0.5, label= 'Parallel and Outside Beam')
#     # axs2[0].hist(beamI.fx, density = True, bins= nBins, alpha = 0.5, label= 'Parallel and Inside Beam')
#     plt.bar(bins1[1:] + widths1/2, data1, width = widths1, alpha = 0.5, label= 'Parallel and Outside Beam')    
#     plt.bar(bins2[1:] + widths2/2, data2, width = widths2, alpha = 0.5, label= 'Parallel and Inside Beam')
#     # plt.text(xMin + 100, data1.max()/4 + data1.max()/ 10, str0, color='C2')
#     # plt.text(xMin + 100, data1.max()/2 + data1.max()/ 20, str1, color='C0')
#     # plt.text(xMin + 100, data1.max()/2 - data1.max()/ 20, str2,  color='C1')

#     #plt.xlim([xMin - 5 , xMax + 5])
    

#     #plt.plot(bincenters7, gauss(bincenters7, paramsa1[0], paramsa1[1], paramsa1[2]))

#     plt.legend(loc = 1)

#     plt.ylabel('Counts'), plt.xlabel('Velocity  ' r'$\mu$m/s');
    
    
#     # FFPI = bins8[(abs(bins8) >= abs(data8).quantile(quant))]
    
#     # FFPI = Stats(abs(bins8[75:100]), data8[1:25], 'PI')

    
    

#     # bins7 = np.linspace(beamA.fx.min(), beamA.fx.max(), nBins)
#     # data7, bins7 = np.histogram(beamA.fx, bins=nBins, density = True)

#     # 

#     # widths = bins7[:-1] - bins7[1:]
#     # plt.bar(bins7[1:], data8, width=widths)
#     # plt.bar(bins1[1:], data1, width=widths)

#     # beamPush = beamA[(beamA['u'] >= 500)]
#     # plt.hist(beamPush.fx, density = True, bins= nBins , alpha = 0.5, label= 'Parallel All')
#     # # residual = np.absolute(data2 - data1)
    
#     # figX, ax = plt.subplots(dpi = 1200)    

#     # plt.bar(bins2[:-1], residual)
#     # plt.xlim(min(bins2), max(bins2))
#     # plt.grid(axis='y', alpha=0.75)
#     # plt.xlabel('Value',fontsize=15)
#     # plt.ylabel('Frequency',fontsize=15)
#     # plt.xticks(fontsize=15)
#     # plt.yticks(fontsize=15)
#     # plt.ylabel('Frequency',fontsize=15)
#     # plt.title('Normal Distribution Histogram',fontsize=15)
#     # plt.show()

    
#     # Saves the figures to the plots folder 
   
#     fig1.savefig(Location + '/Processing/Set' + str(Set) + '/Plots/9_' + DN + '_Velocities_f(y).png')
#     fig2.savefig(Location + '/Processing/Set' + str(Set) + '/Plots/10_' + DN + '_Velocity_Histogram_Split.png')
#     fig3.savefig(Location + '/Processing/Set' + str(Set) + '/Plots/11_' + DN + '_Velocity_Histogram_All.png') 
#     fig4.savefig(Location + '/Processing/Set' + str(Set) + '/Plots/12_' + DN + '_Velocity_Histogram_Push.png')
    
    
#     fig1.savefig(DIR + '/Compiled_Results/Plots/9_' + DN + '_Velocities_f(y).png')
#     fig4.savefig(DIR + '/Compiled_Results/Plots/12_' + DN + '_Velocity_Histogram_Push.png')


#     plt.show()
    
#     if fits == True:
#         Fitting_Params = pd.DataFrame([params], columns = ['Amp','Center','Width', 'R2'], index = ['Gauss'])
#         Fitting_Params.to_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Push_Fitting_Parameters.pickle')
#        # fitP = [paramsa2, paramsb5, paramsb6]# [params, params1, paramsa1, paramsa2, paramsa3, paramsa4, paramsb1, paramsb2]
#         #dfP = pd.DataFrame()
#         #dfP = dfP.append(pd.DataFrame(fitP))
#         #dfP.to_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Fitting_Parameters.pickle')
#     else:
#         pass
        


    
    DFSplit.to_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_Analysis_Statistics_Split.pickle')
    #PI.to_pickle(Location + '/Datasets/' + DN + '_Analysis_Statistics_Push.pickle')
    DFAll.to_pickle(Location + '/Processing/Set' + str(Set)  + '/Datasets/' + DN + '_Analysis_Statistics_ALL.pickle')
    FFSplit.to_pickle(Location + '/Processing/Set' + str(Set) + '/Datasets/' + DN + '_FFAnalysis_Statistics_Split.pickle')
#    FFAll.to_pickle(Location  + '/' + FileName + '/' + Experiment +  '/Datasets/Analysis_Statistics_ALL_'+ FileName +'.pickle')
    beamO.to_pickle(Location + '/Processing/Set' + str(Set)  + '/Datasets/' + DN + '_DataOUT.pickle')
    beamI.to_pickle(Location + '/Processing/Set' + str(Set)  + '/Datasets/' + DN + '_DataIN.pickle')
    beamA.to_pickle(Location + '/Processing/Set' + str(Set)  + '/Datasets/' + DN + '_DataALL.pickle')

    AnalysisVar = pd.DataFrame({'Input': Input, 'nBins': nBins, 'center': center, 'bWidth': bWidth, 'smoo': smoo, 'ptile': ptile, 'cutoff': cutoff, 'roundV': roundV, 'fits': fits}.items())
    AnalysisVar.to_pickle(Location + '/Processing/Set' + str(Set) + '/Variables/Analysis_V.pickle')