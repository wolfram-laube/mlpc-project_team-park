# =============================================================================
# Incomplete:
# Fix ETA & add elapsed time
# =============================================================================

import numpy as numpy
import pandas
import matplotlib.pyplot as pyplot
from matplotlib.colors import LogNorm

import time
from enum import Enum
from os import path as os_path

# Terminology:
# Snippet - One audio segment, 1.1 seconds in length.
# Frame   - One spectral feature time slot. Every snippet contains 44 of these.
# Bin     - One spectral frequency slot. Every snippet has 64 of these.

FrameCount = 44
#Features    = Enum('Features', ['bandwidth', 'centroid', 'energy', 'flatness', 'flux', 'power', 'yin', 'zcr', 'contrast', 'melspect', 'mfcc', 'mfcc_d', 'mfcc_d2'], start = 1)
Features    = Enum('Features', ['bandwidth', 'centroid', 'energy', 'flatness', 'flux', 'power', 'yin', 'zcr', 'contrast', 'melspect', 'mfcc', 'mfcc_d', 'mfcc_d2'], start = 0)
Metrics     = Enum('Metrics',  ['mean', 'median', '10p percentile', 'std dev', 'variance', 'outliers'], start = 0)
FeaturePlotOrder = [Features.bandwidth.value, Features.centroid.value, Features.yin.value, Features.zcr.value,
                    Features.energy.value, Features.power.value, Features.flatness.value, Features.flux.value,
                    Features.melspect.value, Features.mfcc.value, Features.mfcc_d.value, Features.mfcc_d2.value, Features.contrast.value]
#WordPlotOrder = [1, 15, 6, 17, 11, 13, 7, 12, 4, 14, 8, 16, 2, 10, 5, 9, 19, 0, 3, 18, 20]
WordPlotOrder = [1, 15, 6, 17, 11, 13, 7, 12, 4, 14, 8, 16, 2, 10, 5, 9, 19, 0, 3, 18]

# Internals
SearchDict = dict()
Sigma      = 0.341
WordCount  = 0
Snippets   = 0

def SearchMetadata():
    global WordCount
    global Snippets
    DictIndex  = 0
    ReturnList = [] # List of lists
    PrintIndex = 0
    
    for Index in range(len(Metadata['word'])):
        if Metadata['word'][Index] not in SearchDict:
            SearchDict[Metadata['word'][Index]] = DictIndex
            ReturnList.append([])
            DictIndex = DictIndex + 1
        ReturnList[SearchDict[Metadata['word'][Index]]].append(Index) # ReturnList[WordID]
        
    print('[Info] SearchMetadata generated the following list with ' + str(len(ReturnList)) + ' categories:')
    
    for Key, Value in SearchDict.items():
        print(' - ' + str(len(ReturnList[PrintIndex])) + ' entries of word \'' + '{:<15}'.format(str(Key) + '\'') + ' with Index ' + str(Value))
        Snippets = Snippets + len(ReturnList[PrintIndex])
        PrintIndex = PrintIndex + 1

    WordCount = len(ReturnList)
    return ReturnList

# 2D arrays to create for each snippet:
# contrast (8  x 44)
# melspect (64 x 44)
# mfcc     (32 x 44)
# mfcc_d   (32 x 44)
# mfcc_d2  (32 x 44)
def Reconstruct2DFeature(SnippetID, Feature):
    ArrayYDim  = 0
    Offset     = 0
    match Feature:
        case Features.contrast.value:
            ArrayYDim = 8
            Offset    = 2
        case Features.melspect.value:
            ArrayYDim = 64
            Offset    = 12
        case Features.mfcc.value:
            ArrayYDim = 32
            Offset    = 76
        case Features.mfcc_d.value:
            ArrayYDim = 32
            Offset    = 108
        case Features.mfcc_d2.value:
            ArrayYDim = 32
            Offset    = 140
    ReturnArray = numpy.zeros((FrameCount, ArrayYDim))
    for TimeIndex in range(FrameCount):
        for FeatureIndex in range(ArrayYDim):
            ReturnArray[TimeIndex][FeatureIndex] = Dataset[SnippetID][Offset + FeatureIndex][TimeIndex]
    return ReturnArray

def Reconstruct1DFeature(SnippetID, Feature):
    Offset     = 0
    match Feature:
        case Features.bandwidth.value:
            Offset    = 0
        case Features.centroid.value:
            Offset    = 1
        case Features.energy.value:
            Offset    = 9
        case Features.flatness.value:
            Offset    = 10
        case Features.flux.value:
            Offset    = 11
        case Features.power.value:
            Offset    = 172
        case Features.yin.value:
            Offset    = 173
        case Features.zcr.value:
            Offset    = 174
    ReturnArray = numpy.zeros(FrameCount)
    for TimeIndex in range(FrameCount):
        ReturnArray[TimeIndex] = Dataset[SnippetID][Offset][TimeIndex]
    return ReturnArray

def FeatureStatistics(WordID, Feature):
    ReturnData = []
    ArrayYDim  = 1

    match Feature:
        case Features.contrast.value:
            ArrayYDim = 8
        case Features.melspect.value:
            ArrayYDim = 64
        case Features.mfcc.value:
            ArrayYDim = 32
        case Features.mfcc_d.value:
            ArrayYDim = 32
        case Features.mfcc_d2.value:
            ArrayYDim = 32
            
    WordData = numpy.zeros([len(SortedFeatures[WordID]), FrameCount, ArrayYDim])

    for Snippet in range(len(SortedFeatures[WordID])):
        WordData[Snippet] = numpy.array(SortedFeatures[WordID][Snippet][Feature]).reshape(FrameCount, ArrayYDim)
    
    #WordData *= 1.0 / WordData.max()
    ReturnData.append(numpy.mean      (WordData,       axis = 0)) # 0
    ReturnData.append(numpy.median    (WordData,       axis = 0)) # 1
    ReturnData.append(numpy.percentile(WordData, 10.0, axis = 0)) # 2
    ReturnData.append(numpy.std       (WordData,       axis = 0)) # 3
    ReturnData.append(numpy.var       (WordData,       axis = 0)) # 4
    
    # Compute the percentage of pixels which deviate from the median by more than 2 sigma
    ReturnData.append(numpy.zeros(len(WordData)))
    for Snippet in range(len(WordData)):
        OutlierCount = 0
        for TimeIndex in range(FrameCount):
            for Y in range(ArrayYDim):
                if ((WordData[Snippet][TimeIndex][Y] > (ReturnData[Metrics.median.value][TimeIndex][Y] * (1 + (2 * Sigma))))
                or  (WordData[Snippet][TimeIndex][Y] > (ReturnData[Metrics.median.value][TimeIndex][Y] * (1 - (2 * Sigma))))):
                    OutlierCount = OutlierCount + 1
        ReturnData[5][Snippet] = (OutlierCount / (FrameCount * ArrayYDim)) * 100 # 5

    return ReturnData

def AggregateFeatures(SnippetID):
    ReturnList = []
    #ReturnList.append(SnippetID)                                          # 0
    ReturnList.append(Reconstruct1DFeature(SnippetID, Features.bandwidth.value)) # 1
    ReturnList.append(Reconstruct1DFeature(SnippetID, Features.centroid.value )) # 2
    ReturnList.append(Reconstruct1DFeature(SnippetID, Features.energy.value   )) # 3
    ReturnList.append(Reconstruct1DFeature(SnippetID, Features.flatness.value )) # 4
    ReturnList.append(Reconstruct1DFeature(SnippetID, Features.flux.value     )) # 5
    ReturnList.append(Reconstruct1DFeature(SnippetID, Features.power.value    )) # 6
    ReturnList.append(Reconstruct1DFeature(SnippetID, Features.yin.value      )) # 7
    ReturnList.append(Reconstruct1DFeature(SnippetID, Features.zcr.value      )) # 8
    
    ReturnList.append(Reconstruct2DFeature(SnippetID, Features.contrast.value )) # 9
    ReturnList.append(Reconstruct2DFeature(SnippetID, Features.melspect.value )) # 10
    ReturnList.append(Reconstruct2DFeature(SnippetID, Features.mfcc.value     )) # 11
    ReturnList.append(Reconstruct2DFeature(SnippetID, Features.mfcc_d.value   )) # 12
    ReturnList.append(Reconstruct2DFeature(SnippetID, Features.mfcc_d2.value  )) # 13
    return ReturnList

# Assuming the path to your file is correct on your local system
WorkingDir = os_path.dirname(os_path.abspath(__file__))
File       = os_path.join(WorkingDir, 'dataset', 'development.npy')
Dataset    = numpy.load(File)
File       = os_path.join(WorkingDir, 'dataset', 'development.csv')
Metadata   = pandas.read_csv(File)
print('[Info] Opened dataset has ' + str(len(Dataset)) + ' entries.')

# Order the dataset with the given metadata
SortedFeatures = SearchMetadata()

ID        = 0
Processed = 0
Start     = time.time()
End       = time.time()
pDoneVal  = 0.0
pDoneStr  = ''
ETAsec    = 0
ETAstr    = ''

# Reorder features & reconstruct 2D data in memory
for Word in range(WordCount):
    for Snippet in range(len(SortedFeatures[Word])):
        Processed = Processed + 1
        ID = SortedFeatures[Word][Snippet]
        SortedFeatures[Word][Snippet] = AggregateFeatures(ID)

    End      = time.time()
    ETAsec   = End - Start
    pDoneVal = Processed / Snippets
    ETAsec   = (ETAsec * (1 / pDoneVal)) - ETAsec
    pDoneVal = pDoneVal * 100
    pDoneStr = '{:.1f}'.format(pDoneVal)
    pDoneStr = '{:<4}'.format(pDoneStr)
    ETAstr   = '{:.1f}'.format(ETAsec)
    ETAstr   = '{:<6}'.format(ETAstr)
    print('[Info] Reordering data for word: \'' + '{:<15}'.format(list(SearchDict.keys())[list(SearchDict.values()).index(Word)] + '\'') + '; last ID: ' + str(ID) + '; ' + pDoneStr + ' % done; ETA: ' + ETAstr + ' sec')

Processed = 0
Start     = time.time()
End       = time.time()
pDoneVal  = 0.0
pDoneStr  = ''
ETAsec    = 0
ETAstr    = ''
Data      = []

Rows      = len(Features)
Columns   = len(Metrics)

# Compute statistics
for Word in range(WordCount): 
    Data.append([])

    #for Feature in range(1, Rows):
    for Feature in range(Rows):
        Data[Word].append(FeatureStatistics(Word, Feature))
        
    Processed = Processed + 1
    End       = time.time()
    ETAsec    = End - Start
    pDoneVal  = Processed / WordCount
    ETAsec    = (ETAsec * (1 / pDoneVal)) - ETAsec
    pDoneVal  = pDoneVal * 100
    pDoneStr  = '{:.1f}'.format(pDoneVal)
    pDoneStr  = '{:<4}'.format(pDoneStr)
    ETAstr    = '{:.1f}'.format(ETAsec)
    ETAstr    = '{:<6}'.format(ETAstr)
    print('[Info] Computing statistics for word: \'' + '{:<15}'.format(list(SearchDict.keys())[list(SearchDict.values()).index(Word)] + '\'') + '; ' + pDoneStr + ' % done; ETA: ' + ETAstr + ' sec')

Processed = 0
Start     = time.time()
End       = time.time()
pDoneVal  = 0.0
pDoneStr  = ''
ETAsec    = 0
ETAstr    = ''

# Render statistics
for Word in range(WordCount): 
    Figure    = pyplot.figure(figsize = (25, 25))
    PlotCoord = 1
    is2D      = False

    Figure.suptitle(list(SearchDict.keys())[list(SearchDict.values()).index(Word)], fontsize=16)
    #for Feature in range(1, Rows):
    for Feature in range(Rows):
        match Feature:
            case Features.contrast.value:
                is2D = True
            case Features.melspect.value:
                is2D = True
            case Features.mfcc.value:
                is2D = True
            case Features.mfcc_d.value:
                is2D = True
            case Features.mfcc_d2.value:
                is2D = True

        for Metric in range(0, Columns):
            pyplot.subplot(Rows, Columns, PlotCoord)
            PlotCoord = PlotCoord + 1

            if (is2D == True) and (Metric != Metrics.outliers.value):
                #ArrayXForm = numpy.transpose(Data[Word][Feature - 1][Metric])
                ArrayXForm = numpy.transpose(Data[Word][Feature][Metric])
                ArrayXForm = numpy.flip(ArrayXForm, axis = 0)
                pyplot.imshow(ArrayXForm, cmap='hot', interpolation='nearest')
            else:
                #pyplot.plot(Data[Word][Feature - 1][Metric])
                pyplot.plot(Data[Word][Feature][Metric])
            pyplot.title(Features(Feature).name + ' ' + Metrics(Metric).name)

    Processed = Processed + 1
    End       = time.time()
    ETAsec    = End - Start
    pDoneVal  = Processed / WordCount
    ETAsec    = (ETAsec * (1 / pDoneVal)) - ETAsec
    pDoneVal  = pDoneVal * 100
    pDoneStr  = '{:.1f}'.format(pDoneVal)
    pDoneStr  = '{:<4}'.format(pDoneStr)
    ETAstr    = '{:.1f}'.format(ETAsec)
    ETAstr    = '{:<6}'.format(ETAstr)
    print('[Info] Rendering image for word: \'' + '{:<15}'.format(list(SearchDict.keys())[list(SearchDict.values()).index(Word)] + '\'') + '; ' + pDoneStr + ' % done; ETA: ' + ETAstr + ' sec')
    
    pyplot.show()

def SummaryPlot():
    Rows       = len(Features)
    Columns    = WordCount
    Figure     = pyplot.figure(figsize = (25, 25))
    GridSpec   = Figure.add_gridspec(Rows, Columns, hspace = 0, wspace = 0)
    Figure.suptitle('Median', fontsize=16)
    
    Axs = GridSpec.subplots(sharex='col', sharey='row')
    
    #constrast , mfcc log scale
    # add legend for 1 d plot
    #add color bar for 2d plots
    # add labels, fix overlap
    # fix time scale on bottom, add a title on top 'Time ->'

    for Feature in FeaturePlotOrder:
        print('[Debug] Feature: ' + str(Feature))
        is2D      = False
        isLog     = False
        match Feature:
            case Features.contrast.value:
                is2D  = True
                isLog = True
            case Features.melspect.value:
                is2D  = True
            case Features.mfcc.value:
                is2D  = True
                isLog = True
            case Features.mfcc_d.value:
                is2D  = True
                isLog = True
            case Features.mfcc_d2.value:
                is2D  = True
                isLog = True
                
        Axs[Feature, 0].set_ylabel(Features(Feature).name, rotation = 'horizontal')
        
        for Word in WordPlotOrder:   
            print('[Debug] Word: ' + str(Word))
            if (is2D == True):
                ArrayXForm = numpy.transpose(Data[Word][Feature][Metrics.median.value])
                ArrayXForm = numpy.flip(ArrayXForm, axis = 0)
                if (isLog == True):
                    #Axs[Feature, Word].imshow(ArrayXForm, cmap='plasma', norm = LogNorm(vmin=0.0, vmax=0.3), interpolation='nearest', aspect='auto')
                    Axs[Feature, Word].imshow(ArrayXForm, cmap='plasma', interpolation='nearest', aspect='auto')
                else:
                    Axs[Feature, Word].imshow(ArrayXForm, cmap='plasma', interpolation='nearest', aspect='auto')
            else:
                Axs[Feature, Word].plot(Data[Word][Feature][Metrics.median.value], zorder = 2.0, color = ('blue',  1.0))
                Axs[Feature, Word].plot(Data[Word][Feature][Metrics.mean.value],   zorder = 1.0, color = ('green', 0.3))

    for Word in WordPlotOrder:   
        Axs[0, Word].set_xlabel(list(SearchDict.keys())[list(SearchDict.values()).index(Word)])
    for ax in Figure.get_axes():
        ax.label_outer()
    pyplot.show()
    
SummaryPlot()
