
# i_Setup.py creates the folders for storing the processing data and 
# extracts the experiment variables from the filenames

import pandas as pd
from _Folders import folder_generator


def Setup(FileName, Location, Experiment):
            
    df = pd.DataFrame()

    # Creates the folder and subfolders for storing processed data
    folder_generator(Location, '/Compiled_Results')
    folder_generator(Location + '/Compiled_Results', '/Plots')
    folder_generator(Location, '/' + FileName)
    folder_generator(Location + '/' + FileName, '/Frames')
    folder_generator(Location, '/' + FileName + '/' + Experiment)
    folder_generator(Location + '/' + FileName + '/' + Experiment, '/Datasets')
    folder_generator(Location + '/' + FileName + '/' + Experiment, '/Plots')

    # Uses the underscores to find "pName_wLength_power" could
    uScores = [i for i in range(len(FileName)) if FileName.startswith('_', i)] 
    pName = FileName[:uScores[0]]
    wLength = int(FileName[uScores[0]+1:uScores[1]-2])
    pend = FileName.find('mW')
    power = int(FileName[uScores[1]+1:pend])
    info = [pName, wLength, power]
    a_series = pd.Series(info).drop_duplicates()
    df = df.append(a_series, ignore_index=True)
        
    df.columns = ['pName', 'wLength', 'Power']
    df.to_pickle(Location + '/' + FileName + '/' + Experiment + '/Datasets/Experiment_Variables_' + FileName +'.pickle')
