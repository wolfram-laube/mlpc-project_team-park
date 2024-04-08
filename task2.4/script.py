import numpy as numpy
import pandas
import matplotlib.pyplot as pyplot
import matplotlib as matplot
#import csv

# Terminology:
# Snippet - One audio segment, 1.1 seconds in length.
# Frame   - One spectral feature time slot. Every snippet contains 44 of these.
# Bin     - One spectral frequency slot. Every snippet has 64 of these.

FrameCount = 44
Sigma      = 0.341

def SearchMetadata():
    SearchDict = dict()
    DictIndex  = 0
    ReturnList = [] # List of lists
    PrintIndex = 0
    
    for Index in range(len(Metadata['word'])):
        if Metadata['word'][Index] not in SearchDict:
            SearchDict[Metadata['word'][Index]] = DictIndex
            ReturnList.append([])
            DictIndex = DictIndex + 1
            #print('[Debug] Added ' + str(Metadata['word'][Index]) + ' to SearchDict.')
        ReturnList[SearchDict[Metadata['word'][Index]]].append(Index) # ReturnList[WordID]
        
    print('SearchMetadata generated the following list with ' + str(len(ReturnList)) + ' categories:')
    
    for Key, Value in SearchDict.items():
        print(str(len(ReturnList[PrintIndex])) + ' entries of word ' + str(Key) + ' with Index ' + str(Value))
        PrintIndex = PrintIndex + 1
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
        case 'contrast':
            ArrayYDim = 8
            Offset    = 2
        case 'melspect':
            ArrayYDim = 64
            Offset    = 12
        case 'mfcc':
            ArrayYDim = 32
            Offset    = 76
        case 'mfcc_d':
            ArrayYDim = 32
            Offset    = 108
        case 'mfcc_d2':
            ArrayYDim = 32
            Offset    = 140
    ReturnArray = numpy.zeros((FrameCount, ArrayYDim))
    for TimeIndex in range(FrameCount):
        for FeatureIndex in range(ArrayYDim):
            #print('Array is: ' + str(ReturnArray) + ', accessing Dataset[Snippet: ' + str(SnippetID) + '][Feature: ' + str(Offset + FeatureIndex) + '][Time: ' + str(TimeIndex) + '] and writing to ReturnArray[Time: ' + str(TimeIndex) + '][Y: ' + str(FeatureIndex) + ']...')
            ReturnArray[TimeIndex][FeatureIndex] = Dataset[SnippetID][Offset + FeatureIndex][TimeIndex]
    return ReturnArray

def Reconstruct1DFeature(SnippetID, Feature):
    Offset     = 0
    match Feature:
        case 'bandwidth':
            Offset    = 0
        case 'centroid':
            Offset    = 1
        case 'energy':
            Offset    = 9
        case 'flatness':
            Offset    = 10
        case 'flux':
            Offset    = 11
        case 'power':
            Offset    = 172
        case 'yin':
            Offset    = 173
        case 'zcr':
            Offset    = 174
    ReturnArray = numpy.zeros(FrameCount)
    for TimeIndex in range(FrameCount):
        ReturnArray[TimeIndex] = Dataset[SnippetID][Offset][TimeIndex]
    return ReturnArray

def FeatureStatistics(WordID, Feature):
    ReturnData = []
    FeatureID  = 0
    ArrayYDim  = 0
    match Feature:
        case 'bandwidth':
            FeatureID = 1
        case 'centroid':
            FeatureID = 2
        case 'energy':
            FeatureID = 3
        case 'flatness':
            FeatureID = 4
        case 'flux':
            FeatureID = 5
        case 'power':
            FeatureID = 6
        case 'yin':
            FeatureID = 7
        case 'zcr':
            FeatureID = 8
        case 'contrast':
            ArrayYDim = 8
            FeatureID = 9
        case 'melspect':
            ArrayYDim = 64
            FeatureID = 10
        case 'mfcc':
            ArrayYDim = 32
            FeatureID = 11
        case 'mfcc_d':
            ArrayYDim = 32
            FeatureID = 12
        case 'mfcc_d2':
            ArrayYDim = 32
            FeatureID = 13
    WordData = numpy.zeros([len(SortedFeatures[WordID]), FrameCount, ArrayYDim])

    for Snippet in range(len(SortedFeatures[WordID])):
        WordData[Snippet] = numpy.array(SortedFeatures[WordID][Snippet][FeatureID])
    #    for Time in range(44):
    #        for Y in range(64):
    #            Average[Snippet][Time][Y] = SortedFeatures[0][Snippet][10][Time][Y]
    
    #WordData *= 1.0 / WordData.max()
    ReturnData.append(numpy.mean(WordData, axis = 0))
    ReturnData.append(numpy.median(WordData, axis = 0))
    ReturnData.append(numpy.percentile(WordData, 10.0, axis = 0))
    ReturnData.append(numpy.std(WordData, axis = 0))
    ReturnData.append(numpy.var(WordData, axis = 0))
    
    ReturnData.append(numpy.zeros(len(WordData)))
    for Snippet in range(len(WordData)):
        OutlierCount = 0
        for TimeIndex in range(FrameCount):
            for Y in range(ArrayYDim):
                if ((WordData[Snippet][TimeIndex][Y] > (ReturnData[1][TimeIndex][Y] * (1 + Sigma)))
                    or (WordData[Snippet][TimeIndex][Y] > (ReturnData[1][TimeIndex][Y] * (1 - Sigma)))):
                    OutlierCount = OutlierCount + 1
        ReturnData[5][Snippet] = OutlierCount / (FrameCount * ArrayYDim)

    return ReturnData

def AggregateFeatures(SnippetID):
    ReturnList = []
    ReturnList.append(SnippetID)                                    # 0
    ReturnList.append(Reconstruct1DFeature(SnippetID, 'bandwidth')) # 1
    ReturnList.append(Reconstruct1DFeature(SnippetID, 'centroid' )) # 2
    ReturnList.append(Reconstruct1DFeature(SnippetID, 'energy'   )) # 3
    ReturnList.append(Reconstruct1DFeature(SnippetID, 'flatness' )) # 4
    ReturnList.append(Reconstruct1DFeature(SnippetID, 'flux'     )) # 5
    ReturnList.append(Reconstruct1DFeature(SnippetID, 'power'    )) # 6
    ReturnList.append(Reconstruct1DFeature(SnippetID, 'yin'      )) # 7
    ReturnList.append(Reconstruct1DFeature(SnippetID, 'zcr'      )) # 8
    ReturnList.append(Reconstruct2DFeature(SnippetID, 'contrast'))  # 9
    ReturnList.append(Reconstruct2DFeature(SnippetID, 'melspect'))  # 10
    ReturnList.append(Reconstruct2DFeature(SnippetID, 'mfcc'    ))  # 11
    ReturnList.append(Reconstruct2DFeature(SnippetID, 'mfcc_d'  ))  # 12
    ReturnList.append(Reconstruct2DFeature(SnippetID, 'mfcc_d2' ))  # 13
    return ReturnList

# Assuming the path to your file is correct on your local system
Dataset = numpy.load('C:\\Users\\InstallTest\\Documents\\development.npy')
print('Opened dataset has ' + str(len(Dataset)) + ' entries.')
Metadata = pandas.read_csv('C:\\Users\\InstallTest\\Documents\\development.csv')

SortedFeatures = SearchMetadata() # List of lists of ids for one word.

ID = 0
TotalCount = 45296
RunCount = 0
for Word in range(len(SortedFeatures)):
    print('Sorting & reconstructing data for word: \'' + str(Word) + '\'; ID: ' + str(ID) + '; % done: ' + str((RunCount / TotalCount) * 100))
    for Snippet in range(len(SortedFeatures[Word])):
        RunCount = RunCount + 1
        ID = SortedFeatures[Word][Snippet]
        SortedFeatures[Word][Snippet] = AggregateFeatures(ID)

for Word in range(len(SortedFeatures)):
    Columns = 6
    Rows = 2
    Figure = pyplot.figure(figsize = (8, 8))
    Feature = 0
    is2D = 0

    for Run in range(13):
        match Run:
            case 1:
                Feature = 'bandwidth'
                is2D    = 0
            case 2:
                Feature = 'centroid'
                is2D    = 0
            case 3:
                Feature = 'energy'
                is2D    = 0 
            case 4:
                Feature = 'flatness'
                is2D    = 0
            case 5:
                Feature = 'flux'
                is2D    = 0
            case 6:
                Feature = 'power'
                is2D    = 0
            case 7:
                Feature = 'yin'
                is2D    = 0
            case 8:
                Feature = 'zcr'
                is2D    = 0
            case 9:
                Feature = 'contrast'
                is2D    = 1
            case 10:
                Feature = 'melspect'
                is2D    = 1
            case 11:
                Feature = 'mfcc'
                is2D    = 1
            case 12:
                Feature = 'mfcc_d'
                is2D    = 1
            case 13:
                Feature = 'mfcc_d2'
                is2D    = 1  
        Data = FeatureStatistics(Word, Feature)
        for Index in range(1, 6):
            Figure.add_subplot(Rows, Columns, Index)
            match Index:
                case 1:
                    Title = 'mean'
                case 2:
                    Title = 'median'
                case 3:
                   Title = '10% percentile'
                case 4:
                    Title = 'std deviation'
                case 5:
                    Title = 'variance'
            if is2D == 1:
                pyplot.imshow(Data[Index - 1], cmap='hot', interpolation='nearest')
            else:
                pyplot.plot(Data[Index - 1])
            pyplot.title(str(Word) + Title)
    pyplot.show()
