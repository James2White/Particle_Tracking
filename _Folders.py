
# This code makes a folder (fname) in location (floc) if there isn't a folder there already.

import os

floc = 'D:\Data/20220518/ShazND_800nm_100mW/2022_05_18_17_04_31/Processing/'
fname = 'Set'
x = 0



def folder_generator(floc, fname):
    dirName = floc + fname
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory: " , dirName ,  " (was created)")
        return dirName

    else:    
        print("Directory: " , dirName ,  " (already existed)")
        
        
        
def inc_folder_generator(floc, fname, x):
    while True:
       dir_name = (floc + fname + str(x) + '/')
       if not os.path.exists(dir_name):
           os.mkdir(dir_name)
           return dir_name
           print("Directory: " , dir_name ,  " (was created)")
       else:
           x = x + 1
        
        
        
 

