# =============================================================================
# Note: Due to a python multiprocessing module bug, this script will hang indefinitely when one of the worker processes terminates unexpectedly.
# Incomplete:
# =============================================================================

import numpy
import pandas
import matplotlib.pyplot as pyplot
import multiprocessing
from multiprocessing import shared_memory
import time # for timestamps in output log
import os # for handling file paths
from enum import Enum
import random #  to generate random ids
import sys   # for some info messages
import secrets # DEPRECATED for randomised snippet distribution and id generation
import operator
import sklearn
import sklearn.neighbors
import sklearn.metrics

# Terminology:
# Snippet - One audio clip, 1.1 seconds in length.
# Frame   - One spectral feature time slot. Every snippet contains 44 of these.
# Bin     - One spectral frequency slot. Every snippet has 64 of these.

# Configurables:
global Config
Config = dict({
    'SystemThreadMul': 0.8,
    'FrameCount'     : 44,
    'Keywords'       : Enum('Keywords', ['Alarm', 'Staubsauger', 'Lüftung', 'Ofen', 'Heizung', 'Fernseher', 'Licht', 'aus', 'an', 'Radio'], start = 0),
    'OtherWordName'  : 'other',
    'OtherWordID'    : 20,
    'Features'       : Enum('Features', ['bandwidth', 'centroid', 'energy', 'flatness', 'flux', 'power', 'yin', 'zcr', 'contrast', 'melspect', 'mfcc', 'mfcc_d', 'mfcc_d2'], start = 0),
    'Metrics'        : Enum('Metrics',  ['mean', 'median', '10p percentile', 'std dev', 'variance', 'outliers'], start = 0),
    'DatasetRatios'  : (40, 20, 40),
    'TrainOutliers'  : False, # temp, not implemented yet
    'AllowSexBias'   : True,  # temp, not implemented yet
    })
Enums = dict({
    'Propagators'    : Enum('Propagators', ['RowByRow', 'ColByCol', 'Spiral', 'Minispiral', 'Nanospiral', 'Triangle', 'TriangleCol', 'Tile']),
    'Features'       : Enum('Features', ['bandwidth', 'centroid', 'energy', 'flatness', 'flux', 'power', 'yin', 'zcr', 'contrast', 'melspect', 'mfcc', 'mfcc_d', 'mfcc_d2'], start = 0),
})
Config.update([('FeaturePlotOrder', list({Enums['Features'].bandwidth.value, Enums['Features'].centroid.value, Enums['Features'].yin.value,  Enums['Features'].zcr.value, Enums['Features'].energy.value, Enums['Features'].power.value, Enums['Features'].flatness.value, Enums['Features'].flux.value,            Enums['Features'].melspect.value, Enums['Features'].mfcc.value, Enums['Features'].mfcc_d.value, Enums['Features'].mfcc_d2.value, Enums['Features'].contrast.value}))])

# Helper functions:
def Digits(Num):
    Num        = abs(Num)
    Iterations = 0

    while (Num >= 10**Iterations):
        Iterations = Iterations + 1
    return Iterations

def DictKey(Dict, Index):
    Key = list(Dict.keys())[list(Dict.values()).index(Index)]
    return Key

# Data processing functions:

# Builds an offset table for each feature from the provided metadata CSV.
# Returns a list of tuples with feature lengths and offsets.
def ParseOffsets(FeatureMetadata):
    for Entry in range(len(FeatureMetadata)):
        Feature = FeatureMetadata['feature_name'][Entry]
        Index   = FeatureMetadata['index'][Entry]

        FeatureName = Feature.rsplit('_', maxsplit = 1)[1]
    pass

# Builds several tables and derived counts on the dataset with help from the provided metadata CSV.
# Returns a dict with said data.
def ParseMetadata(Metadata):
    lOtherWordID  = Config['OtherWordID']
    WordDict      = dict() # Key - value memory for         word string - internal word ID    pairs.
    SpeakerDict   = dict() # Key - value memory for speaker metadata ID - internal speaker ID pairs.
    OtherSeqIDs   = dict() # Key - value memory for other snippet sequence metadata ID - corresponding internal ID pairs.
    NoiseSeqIDs   = dict() # Key - value memory for noise snippet sequence metadata ID - corresponding internal ID pairs.

    # Note: All IDs in the below tables are internal only. The metadata IDs are not used anywhere.
    WordIDTable   = list() # 2D table mapping word IDs to snippet IDs. Excludes snippets from other and noise.
    WordSpeakers  = list() # 3D table mapping snippets per speaker and per word to snippet IDs.
    OtherSeqs     = list() # 2D table mapping misc sequence snippets to internal IDs.
    OtherSpeakers = list() # 3D table mapping misc snippets per sequence and per speaker to snippet IDs.
    NoiseSeqs     = list() # 2D table mapping noise sequence snippets to internal IDs.
    NoiseSpeakers = list() # 3D table mapping noise snippets per sequence and per speaker to snippet IDs.

    RawAudioList  = list()

    WDictIndex      = 0 # WordDictionaryIndex
    SDictIndex      = 0 # SpeakerDictionaryIndex
    OtherSeqDictIdx = 0
    NoiseSeqDictIdx = 0
    MaxWordLen      = 0
    Snippets        = 0

    if ((len(Metadata['id']) != len(Metadata['word']))
    or (len(Metadata['id']) != len(Metadata['speaker_id']))):
        print('[Error] Invalid metadata table: uneven column lengths. Aborting.')
        raise SystemExit

    print('[Info] Parsing metadata...')

    # Note: The snippet ID's from the metadata table are intentionally not used here.
    for Snippet in range(len(Metadata)): # for each word ...
        Word       = Metadata['word'][Snippet]
        Speaker    = Metadata['speaker_id'][Snippet]
        WavPath    = Metadata['filename'][Snippet]

        # ... test whether a dictionary update is required ...
        if Speaker not in SpeakerDict:
            SpeakerDict.update([(int(Speaker), SDictIndex)]) # ... and assign each speaker an unique index ...
            WordSpeakers.append(list())
            OtherSpeakers.append(list())
            NoiseSpeakers.append(list())
            RawAudioList.append(list())
            for __ in range(len(WordIDTable)): # Add new word columns for the new speaker.
                WordSpeakers[SDictIndex].append(list())
            SDictIndex = SDictIndex + 1
        if Word not in WordDict:
            WordDict.update([(Word, WDictIndex)]) # ... then assign each word an unique index ...
            WordNum = WordDict[Word]
            if WordNum != lOtherWordID:
                WordIDTable.append(list())
                for SpeakerNum in range(len(WordSpeakers)): # Add a new word column for each speaker.
                    WordSpeakers[SpeakerNum].append(list())
            WDictIndex = WDictIndex + 1

            WordLen    = len(Word)
            MaxWordLen = max(MaxWordLen, WordLen)

        SpeakerNum = SpeakerDict[Speaker]
        RawAudioList[SpeakerNum].append(WavPath)
        RawAudioList[SpeakerNum].append('') # Fill in a placeholder for the inter snippet marker.
        WordNum    = WordDict[Word]
        # Note: The logic below fails if the same sequence ID is shared by multiple speakers, or if two sequences have their snippets interleaved.
        if '_speech_' in WavPath:
            SequenceID = int(WavPath.split('/', maxsplit = 2)[2].split('_', maxsplit = 1)[0]) # Improvement: Fix ugly hardcoding in the list indices.
            if '_true.' in WavPath:
                if SequenceID not in OtherSeqIDs:
                    OtherSeqIDs.update([(SequenceID, OtherSeqDictIdx)])
                    OtherSeqs.append(list())
                    OtherSpeakers[SpeakerNum].append(list())
                    OtherSeqDictIdx = OtherSeqDictIdx + 1
                OtherSeqs[OtherSeqIDs[SequenceID]].append(Snippet)
                OtherSpeakers[SpeakerNum][len(OtherSpeakers[SpeakerNum]) - 1].append(Snippet) # Append to the last sequence.
            elif '_false.' in WavPath:
                if SequenceID not in NoiseSeqIDs:
                    NoiseSeqIDs.update([(SequenceID, NoiseSeqDictIdx)])
                    NoiseSeqs.append(list())
                    NoiseSpeakers[SpeakerNum].append(list())
                    NoiseSeqDictIdx = NoiseSeqDictIdx + 1
                NoiseSeqs[NoiseSeqIDs[SequenceID]].append(Snippet)
                NoiseSpeakers[SpeakerNum][len(NoiseSpeakers[SpeakerNum]) - 1].append(Snippet)

        # ... then insert the snippet numbers into each table.
        if WordNum == lOtherWordID:
            continue
        WordIDTable[WordNum].append(Snippet)
        WordSpeakers[SpeakerNum][WordNum].append(Snippet)

    WordCount    = len(WordIDTable)
    SpeakerCount = len(WordSpeakers)

    # Display statistics. Do not test for data quality here.
    SpeakerWordList = list() # speakers per word rename
    for WordNum in range(WordCount): # For each word ...
        SpeakerWordList.append(0)
        for SpeakerNum in range(SpeakerCount):
            if len(WordSpeakers[SpeakerNum][WordNum]) > 0: # ... test if the current speaker's dataset contains any snippets for the current word.
                SpeakerWordList[WordNum] = SpeakerWordList[WordNum] + 1
    # Note: Due to the seperate handling of the 'other' snippets, a second loop is run here. The code below the loop uses entries from WordDict to avoid issues with WordCount not accounting for the 'other' word.
    SpeakerWordList.append(0) # Append ... the entire 'other' category as it appears in the dataset ...
    SpeakerWordList.append(0) # ... only the spoken 'other' sequences ...
    SpeakerWordList.append(0) # ... and only the 'noise' sequences.
    for SpeakerNum in range(SpeakerCount):
        CountedInOther  = False # Expected: 153
        CountedInNoise  = False # Expected: 39
        CountedInEither = False # Expected: 170
        for Sequence in OtherSpeakers[SpeakerNum]:
            if len(Sequence) > 0:
                if CountedInEither == False:
                    SpeakerWordList[len(SpeakerWordList) - 3] = SpeakerWordList[len(SpeakerWordList) - 3] + 1
                    CountedInEither = True
                if CountedInOther == False:
                    SpeakerWordList[len(SpeakerWordList) - 2] = SpeakerWordList[len(SpeakerWordList) - 2] + 1
                    CountedInOther = True
        for Sequence in NoiseSpeakers[SpeakerNum]:
            if len(Sequence) > 0:
                if CountedInEither == False:
                    SpeakerWordList[len(SpeakerWordList) - 3] = SpeakerWordList[len(SpeakerWordList) - 3] + 1
                    CountedInEither = True
                if CountedInNoise == False:
                    SpeakerWordList[len(SpeakerWordList) - 1] = SpeakerWordList[len(SpeakerWordList) - 1] + 1
                    CountedInNoise = True

    print('[Info] ParseMetadata has built word-snippet-speaker tables '
    + '{:.1f}'.format((sys.getsizeof(WordIDTable)
                     + sys.getsizeof(WordSpeakers)
                     + sys.getsizeof(OtherSeqs)
                     + sys.getsizeof(OtherSpeakers)
                     + sys.getsizeof(NoiseSeqs)
                     + sys.getsizeof(NoiseSpeakers)
                     + sys.getsizeof(WordDict)
                     + sys.getsizeof(SpeakerDict)
                     + sys.getsizeof(RawAudioList)) / 1024)
    + ' MiB long; and containing ' + str(WordCount + 1) + ' unique words from ' + str(SpeakerCount) + ' unique speakers:')

    # Find the required number of padding spaces for nicer text output.
    MaxEntries   = 0
    OtherEntries = 0
    NoiseEntries = 0
    for WordNum in range(WordCount):
        Entries    = len(WordIDTable[WordNum])
        MaxEntries = max(MaxEntries, Entries)
    for SpeakerNum in range(SpeakerCount):
        for Sequence in OtherSpeakers[SpeakerNum]:
            Entries      = len(Sequence)
            OtherEntries = OtherEntries + Entries
            MaxEntries   = max(MaxEntries, Entries)
        for Sequence in NoiseSpeakers[SpeakerNum]:
            Entries      = len(Sequence)
            NoiseEntries = NoiseEntries + Entries
            MaxEntries   = max(MaxEntries, Entries)

    Items = WordDict.items()
    for Word, WordNum in Items: # WordNum will account for 'other' snippets.
        if WordNum == (len(Items) - 1): # Value increments from 0, therefore the offset.
        # add % of total words in here i.e. 2038 (5%) snippets of word
            print(' ┣ '
            + ('{:<' + str(Digits(MaxEntries))         + '}').format(str(OtherEntries)) + ' other snippets '
            + ('{:<' + str(MaxWordLen + 1)             + '}').format('')                  + ' from ' # + 1 to the format instruction so that there is always one space between the word and subsequent text
            + ('{:<' + str(Digits(SpeakerCount))       + '}').format(str(SpeakerWordList[WordNum + 1]))
            + ' (' + str(SpeakerWordList[WordNum]) + ') speakers.')
            print(' ┗ '
            + ('{:<' + str(Digits(MaxEntries))         + '}').format(str(NoiseEntries)) + ' noise snippets    \''
            + ('{:<' + str(MaxWordLen + 1)             + '}').format('noise\'')         + ' with index '
            + ('{:<' + str(max(2, Digits(len(Items)))) + '}').format('NA')              + ' from '
            + ('{:<' + str(Digits(SpeakerCount))       + '}').format(str(SpeakerWordList[WordNum + 2]))
            + ' (' + str(SpeakerWordList[WordNum]) + ') speakers.')
            Snippets = Snippets + OtherEntries + NoiseEntries
        else:
            print(' ┣ '
            + ('{:<' + str(Digits(MaxEntries))         + '}').format(str(len(WordIDTable[WordNum]))) + ' snippets of word \''
            + ('{:<' + str(MaxWordLen + 1)             + '}').format(str(Word) + '\'')               + ' with index '
            + ('{:<' + str(max(2, Digits(len(Items)))) + '}').format(str(WordNum))           + ' from '
            + ('{:<' + str(Digits(SpeakerCount))       + '}').format(str(SpeakerWordList[WordNum]))  + ' speakers.')
            Snippets = Snippets + len(WordIDTable[WordNum])

    WordDict.update([('noise', 21)]) # temp hack
    Tables = dict({'WordIDTable'   : WordIDTable,
                   'WordSpeakers'  : WordSpeakers,
                   'OtherSeqs'     : OtherSeqs,
                   'OtherSpeakers' : OtherSpeakers,
                   'NoiseSeqs'     : NoiseSeqs,
                   'NoiseSpeakers' : NoiseSpeakers,
                   'WordDict'      : WordDict,
                   'SpeakerDict'   : SpeakerDict,
                   'RawAudioList'  : RawAudioList,
                   'WordCount'     : WordCount,
                   'Snippets'      : Snippets,
                   'MaxWordLen'    : MaxWordLen,
                   'SpeakerCount'  : SpeakerCount,
                   'OtherEntries'  : OtherEntries,
                   'NoiseEntries'  : NoiseEntries})
    return Tables

def PartitionDataset(Tables):#, WordSpeakers, MaxWordLen, WordDict, WordCount, Snippets):
    WordSpeakers  = Tables['WordSpeakers']
    WordDict      = Tables['WordDict']
    NoiseSeqs     = Tables['NoiseSeqs']
    OtherSeqs     = Tables['OtherSeqs']
    OtherSpeakers = Tables['OtherSpeakers']
    NoiseSpeakers = Tables['NoiseSpeakers']
    WordCount     = Tables['WordCount']
    Snippets      = Tables['Snippets']
    MaxWordLen    = Tables['MaxWordLen']
    OtherEntries  = Tables['OtherEntries']
    NoiseEntries  = Tables['NoiseEntries']
    TrainingWordNums = [2, 4, 5, 6, 8, 9, 11, 12, 15, 19] # compute from config enum
    # include outliers in the training data, include females in the training data so that there is little sex bias. make these toggleable.

    # Helper functions
    def UpdateHistogram(*, Data, Histogram):
        Length = len(Data)
        if Length == 0:
            return
        if Length not in [Tuple[0] for Tuple in Histogram]:
            Histogram.append(tuple([Length, 1]))
        else:
            for Bin in range(len(Histogram)):
                if Histogram[Bin][0] == Length:
                    Occurences = Histogram[Bin][1]
                    Histogram[Bin] = tuple([Length, Occurences + 1])
        return

    SpeakerCount     = len(WordSpeakers)
    WordsPerSpeaker  = list() # list which keeps track per speaker how many occurences of each word are in the dataset. should be 12 for all non other words
    Histogram        = list()
    NoiseSeqHist     = list()
    NoiseSnippetHist = list()
    OtherSeqHist     = list()
    OtherSnippetHist = list()
    IncompleteSpeakers = list() #fix
    MostOccurences   = list()
    ExcludedSpeakers = list()
    lDatasetRatios   = Config['DatasetRatios']

    for __ in range(WordCount): # Throwaway index
        Histogram.append(list())
        IncompleteSpeakers.append(list())
        MostOccurences.append(0)

    for SpeakerNum in range(SpeakerCount):
        WordsPerSpeaker.append(list())
        for WordNum in range(WordCount):
            WordsPerSpeaker[SpeakerNum].append(0)
            Words = len(WordSpeakers[SpeakerNum][WordNum])
            WordsPerSpeaker[SpeakerNum][WordNum] = Words

            # Note: These if/else blocks are slightly modified from the <UpdateHistogram> function
            if Words not in [Tuple[0] for Tuple in Histogram[WordNum]]:
                Histogram[WordNum].append(tuple([Words, 1]))
            else:
                for Bin in range(len(Histogram[WordNum])):
                    if Histogram[WordNum][Bin][0] == Words:
                        Occurences = Histogram[WordNum][Bin][1]
                        Histogram[WordNum][Bin] = tuple([Words, Occurences + 1])

            if Words == 0:
                if SpeakerNum not in ExcludedSpeakers:
                    ExcludedSpeakers.append(SpeakerNum)
                print('[Warning] Speaker '
                + ('{:<' + str(Digits(SpeakerCount)) + '}').format(str(SpeakerNum)+ '\'s') + ' dataset does not include any snippets for word \''
                + str(DictKey(WordDict, WordNum)) + '\'.')

                # nonsense, the datasets should be perfectly distributed, even in non keywords. any speaker which does not have the expected dataset is thrown into the validation set. if dataset quality is below the required threshold, error out
                if WordNum in TrainingWordNums:
                    print('[Warning] ' + str(DictKey(WordDict, WordNum)) + 'is a training keyword. Speaker ' + str(SpeakerNum) + ' will be excluded from the training and testing datasets.')
        # check if tests for zero length words/other/noise (parsemetadata cannot generate an empty list?) are nescessary and output here

    # histograms for sequence and snippet counts
    for SpeakerNum in range(len(NoiseSpeakers)):
        UpdateHistogram(Data = NoiseSpeakers[SpeakerNum], Histogram = NoiseSeqHist)
        if len(NoiseSpeakers[SpeakerNum]) == 0:
            print('[Warning] Speaker '
            + ('{:<' + str(Digits(SpeakerCount)) + '}').format(str(SpeakerNum)+ '\'s') + ' dataset does not include any noise sequences.')
    for SpeakerNum in range(len(OtherSpeakers)):
        UpdateHistogram(Data = OtherSpeakers[SpeakerNum], Histogram = OtherSeqHist)
        if len(OtherSpeakers[SpeakerNum]) == 0:
            print('[Warning] Speaker '
            + ('{:<' + str(Digits(SpeakerCount)) + '}').format(str(SpeakerNum)+ '\'s') + ' dataset does not include any other sequences.')

    for Sequence in NoiseSeqs:
        UpdateHistogram(Data = Sequence, Histogram = NoiseSnippetHist)
    for Sequence in OtherSeqs:
        UpdateHistogram(Data = Sequence, Histogram = OtherSnippetHist)

    # Note: This code assumes that the most common value of the histogram is the correct and expected one
    DatasetUneven = False
    for WordNum in range(WordCount):
        Histogram[WordNum].sort(key = operator.itemgetter(1), reverse = True)
        MostOccurences[WordNum] = Histogram[WordNum][0][0]
        if (len(Histogram[WordNum]) > 1):
            DatasetUneven = True

    NoiseSeqHist.sort(key = operator.itemgetter(1), reverse = True)
    NoiseSnippetHist.sort(key = operator.itemgetter(1), reverse = True)
    OtherSeqHist.sort(key = operator.itemgetter(1), reverse = True)
    OtherSnippetHist.sort(key = operator.itemgetter(1), reverse = True)
    if ((len(NoiseSeqHist)     > 1)
    or  (len(NoiseSnippetHist) > 1)
    or  (len(OtherSeqHist)     > 1)
    or  (len(OtherSnippetHist) > 1)):
        DatasetUneven = True

    if DatasetUneven is True:
        print('[Warning] Some datasets are unevenly distributed and will be trimmed or excluded from the training and testing datasets:')
        for SpeakerNum in range(SpeakerCount):
            #for WordNum in range(len(WordSpeakers[SpeakerNum])):
            for WordNum in range(WordCount):
                Word = DictKey(WordDict, WordNum) #list(WordDict.keys())[list(WordDict.values()).index(WordNum)]
                #if WordDict[Word] == OtherWordID:
                #    continue
                if WordsPerSpeaker[SpeakerNum][WordNum] != MostOccurences[WordNum]:
                    if SpeakerNum not in ExcludedSpeakers:
                        ExcludedSpeakers.append(SpeakerNum)
                    Line = '┣'
                    if (SpeakerNum == (SpeakerCount - 1)): # Value increments from 0, therefore the offset.
                        Line = '┗'
                    print(' ' + Line + ' Speaker '
                    + ('{:<' + str(Digits(SpeakerCount) + 2)   + '}').format(str(SpeakerNum) + '\'s') + ' dataset includes '
                    + ('{:<' + str(Digits(max(MostOccurences))) + '}').format(str(WordsPerSpeaker[SpeakerNum][WordNum])) + ' / '
                    + ('{:<' + str(Digits(max(MostOccurences))) + '}').format(str(MostOccurences[WordNum])) + ' snippets for word \''
                    + ('{:<' + str(MaxWordLen + 1)             + '}').format(Word + '\'.'))
            # add other& noise code here
    # display variance of histograms

    # compute quality at the end with len of snippets suitable for training vs total
    DatasetQuality    = (1 - (len(ExcludedSpeakers) / SpeakerCount)) * 100
    DatasetQualityStr = '{:.1f}'.format(DatasetQuality)
    # use snippets in the final reporting, as some speaker may be completely excluded but some only for certain words
    print('[Info] Final dataset quality is: ' + DatasetQualityStr + '% (Excluded ' + str(len(ExcludedSpeakers)) + ' snippets out of ' + str(SpeakerCount) + ').')
    if (DatasetQuality < (100 - lDatasetRatios[2])):
        print('[Warning] Dataset quality (' + DatasetQualityStr + '%) is below the ' + str(lDatasetRatios[0] + lDatasetRatios[1]) + '% required to satisfy the dataset split ratios bias free!')

    # now trim other words to avoid a bias in negative examples in the dataset
    TrainingSnippets   = list()
    TrainingSpeakers   = list()
    TestingSnippets    = list()
    TestingSpeakers    = list()
    ValidationSnippets = list()
    ValidationSpeakers = list()
    for __ in range(WordCount + 2):
        TrainingSnippets.append(list())
        TrainingSpeakers.append(list())
        TestingSnippets.append(list())
        TestingSpeakers.append(list())
        ValidationSnippets.append(list())
        ValidationSpeakers.append(list())

    TrainingSnippetCnt = 0
    TestingSnippetCnt = 0
    ValidationSnippetCnt = 0

    # try to refine the avail speakers table so that redrawing becomes unnescessary since the table is already sorted from best to worst
    def Fill(*, SpeakerTable, SeqHist, SnippetHist, Dataset, EntryCnt, DatasetIndex, RatioIndex):
        Blacklist      = list()
        TakenSpeakers  = list()
        AvailSpeakers  = list()
        FillRatio      = 0
        SeqHistIdx     = 0
        SnippetHistIdx = 0
        SnippetCnt     = 0
        DatasetIndex   = DatasetIndex - 1 # Offset due to zero based arrays
        Count = 0

        def ResetBlacklist():
            nonlocal SeqHistIdx

            print('[Warning] There are insufficient speakers with sequences of length ' + str(SeqHist[SeqHistIdx][0]) + '. Attempting to satisty the requested fill ratio with sequences of length ' + str(SeqHist[SeqHistIdx + 1][0]) + '.')
            SeqHistIdx = SeqHistIdx + 1
            for SpeakerNum in Blacklist: # Then remove any speakers previously blacklisted.
                if len(SpeakerTable[SpeakerNum]) == SeqHist[SeqHistIdx][0]:
                    Blacklist.remove(SpeakerNum)
                    AvailSpeakers.append(SpeakerNum)
            return

        for Index in range(len(SpeakerTable)):
            if len(SpeakerTable[Index]) != 0:
                AvailSpeakers.append(Index)

        while (FillRatio < lDatasetRatios[RatioIndex]):
            SpeakerNum = random.choice(AvailSpeakers) # Note: <FillRatio> should never be able to reach 100% so <AvailSpeakers> will never be empty
            while SpeakerNum in Blacklist:
                SpeakerNum = random.choice(AvailSpeakers)
                Count = Count + 1
                if Count > 10:
                    print('Script has probably entered an infinite loop.')
                    raise SystemExit
            Count = 0
            Sequences = SpeakerTable[SpeakerNum]
            # First try to fill the dataset only with the most common sequence length to avoid bias in the noise distribution
            # Note: The current logic will prefer the most common sequence count, then the second most common one etc. ignoring that some other histogram bin might be closer to the first (most optimal) one.
            # Note: So far there is no test if there even is an SeqHistIdx + 1
            if len(Sequences) != SeqHist[SeqHistIdx][0]:
                Blacklist.append(SpeakerNum)
                AvailSpeakers.remove(SpeakerNum)
                if len(AvailSpeakers) == 0: # Test if there are not enough speakers with the most common number of noise sequences
                    ResetBlacklist()
                continue
            for Sequence in Sequences:
                if (len(Sequence) > (SnippetHist[SnippetHistIdx][0] * (1 + 0.341))
                or len(Sequence) < (SnippetHist[SnippetHistIdx][0] * (1 - 0.341))):
                    print('[Warning] Sequence length deviates by more than 1 sigma (' + str(len(Sequence)) + ' / ' + str(SnippetHist[SnippetHistIdx][0]) + ').')
                    # insert new blacklist code, try to exhaust the median sequence lengths first
                TakenSpeakers.append(SpeakerNum)
                random.shuffle(Sequence)
                for Snippet in Sequence:
                    Dataset[DatasetIndex].append(Snippet)
                    SnippetCnt = SnippetCnt + 1
            FillRatio = (SnippetCnt / EntryCnt) * 100
            AvailSpeakers.remove(SpeakerNum)
            if len(AvailSpeakers) == 0:
                ResetBlacklist()
        print('Blacklisted ' + str(len(Blacklist)) + ' speakers.')
        # print some stats, take the current word (other or noise) from word dict
        return TakenSpeakers

    TrainingSpeakers[WordCount + 1].append(Fill(SpeakerTable = NoiseSpeakers,
                                                SeqHist      = NoiseSeqHist,
                                                SnippetHist  = NoiseSnippetHist,
                                                Dataset      = TrainingSnippets,
                                                EntryCnt     = NoiseEntries,
                                                DatasetIndex = WordCount + 2,
                                                RatioIndex   = 0))
    TrainingSpeakers[WordCount + 0].append(Fill(SpeakerTable = OtherSpeakers,
                                                SeqHist      = OtherSeqHist,
                                                SnippetHist  = OtherSnippetHist,
                                                Dataset      = TrainingSnippets,
                                                EntryCnt     = OtherEntries,
                                                DatasetIndex = WordCount + 1,
                                                RatioIndex   = 0))
    ValidationSpeakers[WordCount + 1].append(Fill(SpeakerTable = NoiseSpeakers,
                                                  SeqHist      = NoiseSeqHist,
                                                  SnippetHist  = NoiseSnippetHist,
                                                  Dataset      = TestingSnippets,
                                                  EntryCnt     = NoiseEntries,
                                                  DatasetIndex = WordCount + 2,
                                                  RatioIndex   = 1))
    ValidationSpeakers[WordCount + 0].append(Fill(SpeakerTable = OtherSpeakers,
                                                  SeqHist      = OtherSeqHist,
                                                  SnippetHist  = OtherSnippetHist,
                                                  Dataset      = TestingSnippets,
                                                  EntryCnt     = OtherEntries,
                                                  DatasetIndex = WordCount + 1,
                                                  RatioIndex   = 1))

    TrainingFillRatio = 0
    CompleteSpeakers = list()
    for SpeakerNum in range(SpeakerCount):
        if SpeakerNum not in ExcludedSpeakers:
            CompleteSpeakers.append(SpeakerNum)

    Speakers = random.sample(CompleteSpeakers, int(SpeakerCount * (lDatasetRatios[0] / 100 )))
    for SpeakerNum in Speakers:
        CompleteSpeakers.remove(SpeakerNum)
        for WordNum in range(WordCount):
            for Snippet in random.sample(WordSpeakers[SpeakerNum][WordNum], Histogram[WordNum][0][0]):
                TrainingSnippets[WordNum].append(Snippet)
                TrainingSnippetCnt = TrainingSnippetCnt + 1
    Speakers = random.sample(CompleteSpeakers, int(SpeakerCount * (lDatasetRatios[1] / 100 )))
    for SpeakerNum in Speakers:
        CompleteSpeakers.remove(SpeakerNum)
        for WordNum in range(WordCount):
            for Snippet in random.sample(WordSpeakers[SpeakerNum][WordNum], Histogram[WordNum][0][0]):
                TestingSnippets[WordNum].append(Snippet)
                TestingSnippetCnt = TestingSnippetCnt + 1

    for SpeakerNum in CompleteSpeakers:
        for WordNum in range(WordCount): # <WordCount> will never result in out of bounds memory access due to the empty lists in place
            Randomised = random.sample(WordSpeakers[SpeakerNum][WordNum], k = len(WordSpeakers[SpeakerNum][WordNum]))
            for Snippet in Randomised:
                ValidationSnippets[WordNum].append(Snippet)
    for SpeakerNum in ExcludedSpeakers:
        for WordNum in range(WordCount): # <WordCount> will never result in out of bounds memory access due to the empty lists in place
            Randomised = random.sample(WordSpeakers[SpeakerNum][WordNum], k = len(WordSpeakers[SpeakerNum][WordNum]))
            for Snippet in Randomised:
                ValidationSnippets[WordNum].append(Snippet)
    for SpeakerNum in range(SpeakerCount):
        if (SpeakerNum not in TrainingSpeakers[WordCount + 1]
        and SpeakerNum not in TestingSpeakers[WordCount + 1]):
            for SequenceNum in range(len(NoiseSpeakers[SpeakerNum])):
                Randomised = random.sample(NoiseSpeakers[SpeakerNum][SequenceNum], k = len(NoiseSpeakers[SpeakerNum][SequenceNum]))
                for Snippet in Randomised:
                    ValidationSnippets[WordCount + 1].append(Snippet)
                    ValidationSnippetCnt = ValidationSnippetCnt + 1
        if (SpeakerNum not in TrainingSpeakers[WordCount + 0]
        and SpeakerNum not in TestingSpeakers[WordCount + 0]):
            for SequenceNum in range(len(OtherSpeakers[SpeakerNum])):
                Randomised = random.sample(OtherSpeakers[SpeakerNum][SequenceNum], k = len(OtherSpeakers[SpeakerNum][SequenceNum]))
                for Snippet in Randomised:
                    ValidationSnippets[WordCount + 0].append(Snippet)
                    ValidationSnippetCnt = ValidationSnippetCnt + 1
                
    rWordCounts = list()
    for Idx in range(WordCount + 2): # Throwaway index
        rWordCounts.append(list())
        rWordCounts[Idx].append(0)

    rTrainingSnippetCnt = 0
    for WordNum in range(len(TrainingSnippets)):
        rTrainingSnippetCnt = rTrainingSnippetCnt + len(TrainingSnippets[WordNum])
        rWordCounts[WordNum][0] = rWordCounts[WordNum][0] + len(TrainingSnippets[WordNum])
    rTestingSnippetCnt = 0
    for WordNum in range(len(TestingSnippets)):
        rTestingSnippetCnt = rTestingSnippetCnt + len(TestingSnippets[WordNum])
        rWordCounts[WordNum][0] = rWordCounts[WordNum][0] + len(TestingSnippets[WordNum])
    rValidationSnippetCnt = 0
    for WordNum in range(len(ValidationSnippets)):
        rValidationSnippetCnt = rValidationSnippetCnt + len(ValidationSnippets[WordNum])
        rWordCounts[WordNum][0] = rWordCounts[WordNum][0] + len(ValidationSnippets[WordNum])

    percentages = list()
    percentages.append(rTrainingSnippetCnt / Snippets)
    percentages.append(rTestingSnippetCnt / Snippets)
    percentages.append(rValidationSnippetCnt / Snippets)
        
    if (rTrainingSnippetCnt + rTestingSnippetCnt + rValidationSnippetCnt) != Snippets:
        print ('distributed ' + str(rTrainingSnippetCnt + rTestingSnippetCnt + rValidationSnippetCnt) + ' out of ' + str(Snippets) + ' snippets.')

    # do final randomisation, for sequences, each sequence is ramdom but the sequences are not interleaved
    return (TrainingSnippets, TestingSnippets, ValidationSnippets)

# scales all input data per category from 0-1 & sets numpy precision from fp64
#def NormaliseDataset:

    
#def FindOutliers:
    
def BuildValidationVectors(WordCount, DatasetTables, WordDict):
    TrainingWordNums = [2, 4, 5, 6, 8, 9, 11, 12, 15, 19] # compute from config enum
    DatasetStr = ['Train', 'Test', 'Valid']
    
    VectorLength = len(TrainingWordNums)
    #Index = 0
    for Dataset in range(len(DatasetTables)):
        for WordNum in range(WordCount + 2):
            Length = len(DatasetTables[Dataset][WordNum])
            array = numpy.zeros(shape = (Length, VectorLength), dtype = numpy.double)
            for Snippet in range(Length):
                for Element in range(VectorLength):
                    if WordNum == TrainingWordNums[Element]:
                        array[Snippet][Element] = 1.0
            Filename = 'Vectors-' + DatasetStr[Dataset] + '-' + DictKey(WordDict, WordNum)
            print('[Info] Writing ' + Filename + '...')
            numpy.save(file = Filename, arr = array)
# the final classifier needs a linear list of arrays for data and expected output
    
#degrades the validation dataset to check how robust the model ist
#def degradeData

# concats each speakers raw audio. adds short beep in between samples
#def ConcatRawSnippets:

# performs various triangle & spiral decompositions to preprocess 2d data
def LineariseData(Matrix, Propagator = Enums['Propagators'].Nanospiral.value):
#column, row, spiral, minispiral, nanospiral, triangle, column triangle, frequency rectangles (fft, high frequency regions get subdivided into small squares)
    XTile = 0
    YTile = 0
    PatternXDim = 1
    PatternYDim = 1
    XTilesRemaining = Matrix.shape[0] / PatternXDim
    YTilesRemaining = Matrix.shape[1] / PatternYDim
    #LinearData = numpy.ndarray(shape = Matrix.shape[0] * Matrix.shape[1])
    LinearData = numpy.ravel(Matrix)


    #def Propagate():
    
    #def WritePattern():
        
    #while(XTilesRemaining > 0):
    #    while(YTilesRemaining > 0):
    #        WritePattern()
    #        Propagate()
    return LinearData



def BuildDatasets(WordNum, TaskID, iMems, DatasetTables):
    FrameCount = 44
    # 2D arrays to create for each snippet:
    # contrast (8  x 44)
    # melspect (64 x 44)
    # mfcc     (32 x 44)
    # mfcc_d   (32 x 44)
    # mfcc_d2  (32 x 44)
    def Reconstruct2DFeature(SnippetID, Feature, Features, FrameCount, Mem, MemIdx):
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
    
        #ReturnArray = numpy.zeros((FrameCount, ArrayYDim))
        Buffer = numpy.zeros((FrameCount, ArrayYDim))
        # try to operate directly in imems
        for TimeIndex in range(FrameCount):
            for FeatureIndex in range(ArrayYDim):
                Buffer[TimeIndex][FeatureIndex] = SharedDataset[SnippetID][Offset + FeatureIndex][TimeIndex]
        Buffer = LineariseData(Buffer)
        for Val in range(len(Buffer)):
            Mem[1][MemIdx][Val] = Buffer[Val]
    
        return
    #lMem = WorkerMemory['Mems'][WordNum]
    lMem = iMems[WordNum]
    for MemIdx, SnippetNum in zip(range(len(DatasetTables[0][WordNum])), DatasetTables[0][WordNum]):
        Reconstruct2DFeature(SnippetNum, Enums['Features'].melspect.value, Enums['Features'], FrameCount, lMem[0], MemIdx) # Train
        Reconstruct2DFeature(SnippetNum, Enums['Features'].mfcc_d2.value, Enums['Features'], FrameCount, lMem[3], MemIdx) # Train
        #Reconstruct2DFeature(SnippetNum, Enums['Features'].zcr.value, Enums['Features'], FrameCount, lMem[6], MemIdx) # Train
    for MemIdx, SnippetNum in zip(range(len(DatasetTables[1][WordNum])), DatasetTables[1][WordNum]):
        Reconstruct2DFeature(SnippetNum, Enums['Features'].melspect.value, Enums['Features'], FrameCount, lMem[1], MemIdx) # Test
        Reconstruct2DFeature(SnippetNum, Enums['Features'].mfcc_d2.value, Enums['Features'], FrameCount, lMem[4], MemIdx)# Test
        #Reconstruct2DFeature(SnippetNum, Enums['Features'].zcr.value, Enums['Features'], FrameCount, lMem[7], MemIdx) # Test
    for MemIdx, SnippetNum in zip(range(len(DatasetTables[2][WordNum])), DatasetTables[2][WordNum]):
        Reconstruct2DFeature(SnippetNum, Enums['Features'].melspect.value, Enums['Features'], FrameCount, lMem[2], MemIdx) # Test
        Reconstruct2DFeature(SnippetNum, Enums['Features'].mfcc_d2.value, Enums['Features'], FrameCount, lMem[5], MemIdx)# Test
        #Reconstruct2DFeature(SnippetNum, Enums['Features'].zcr.value, Enums['Features'], FrameCount, lMem[7], MemIdx) # Test
    #return (WordNum, TaskID, WorkerMemory['ID'])
    return (WordNum, TaskID)
# Auxiliary functions
def GetETA(*, Rebase = False, Workload = 1):
    global RefTimestamp

    if (Rebase is True):
        RefTimestamp = time.perf_counter_ns()
        return

    End         = time.perf_counter_ns() #lookup time documentation
    ETAsec      = End - RefTimestamp
    tElapsedStr = '{:.1f}'.format(ETAsec)     # one decimal
    tElapsedStr = '{:<6}'.format(tElapsedStr) # padding
    pDoneVal    = Workload  # replace with workload
    #ETAsec      = (ETAsec * (1 / pDoneVal)) - ETAsec
    ETAsec      = (ETAsec * (1 / Workload)) - ETAsec
    pDoneVal    = pDoneVal * 100
    #pDoneStr    = '{:.1f}'.format(pDoneVal)
    pDoneStr    = '{:.1f}'.format(Workload)
    pDoneStr    = '{:<4}'.format(pDoneStr)
    ETAstr      = '{:.1f}'.format(ETAsec)
    ETAstr      = '{:<6}'.format(ETAstr)

    ReturnString = '; progress: ' + pDoneStr + ' %; ETA: ' + ETAstr + ' sec ; elapsed ' + tElapsedStr + ' sec.'
    #return some sort of object with the two strings

# Multiprocessing helper functions

def PickWorkerID(*, List):
    Index = 0
    while List[Index] == '':
        Index = Index + 1
    ReturnID = List[Index]
    List[Index] = ''
    return (Index, ReturnID)

def WorkerTest(Word, TaskID):
    print('[Debug] Testing worker ' + str(WorkerMemory['ID']) + '(' + str(WorkerMemory['Number']) + '); Word: ' + str(Word) + '; TaskID: ' + str(TaskID) + '.', flush = True)
    CloseSharedMemory()
    return (Word, TaskID, WorkerMemory['ID'])

def CloseSharedMemory():
    for Memory in WorkerMemory['SharedMems']:
        Memory.close()
    global DatasetMemory

    DatasetMemory.close()

    return

def WorkerSetup(SharedList, pMems, DatasetTables, DatasetShape):
    global WorkerMemory
    global DatasetMemory # global for <CloseSharedMemory()>
    global SharedDataset

    global wDatasetTables
    wDatasetTables = DatasetTables
    
    Mems          = list()
    WorkerParams  = PickWorkerID(List = SharedList)
    WorkerNumber  = WorkerParams[0]
    WorkerID      = WorkerParams[1]
    DatasetMemory = multiprocessing.shared_memory.SharedMemory(name = 'DatasetMem')
    SharedDataset = numpy.ndarray(shape = DatasetShape, dtype = numpy.double, buffer = DatasetMemory.buf)

    for WordNum in range(len(pMems)):
        Mems.append(list())
        for Mem in pMems[WordNum]:
            Shared = multiprocessing.shared_memory.SharedMemory(name = Mem[0])
            Numpy  = numpy.ndarray(shape = Mem[1], dtype = numpy.double, buffer = Shared.buf)
            Mems[WordNum].append(tuple([Shared, Numpy]))

    TrainingSnippets   = DatasetTables[0]
    TestingSnippets    = DatasetTables[1]
    ValidationSnippets = DatasetTables[2]
    WorkerProcess = multiprocessing.current_process()
    ParentProcess = multiprocessing.parent_process()
    PID           = WorkerProcess.pid

    print('[Info] Worker number / ID: ' + str(WorkerNumber) + ' / ' + str(WorkerID) + '; with name: ' + str(WorkerProcess.name) + '; and parent / worker PID: ' + str(ParentProcess.pid) + ' / ' + str(PID) + '; has spawned.', flush = True)

    WorkerMemory = dict({'Number': WorkerNumber, 'ID': WorkerID, 'PID': PID, 'Mems': Mems})
    return

def Main():
    global DatasetMemory

    #global WorkerDict
    WorkerDict = dict()
    # goto parent dir
    WorkingDir = os.path.dirname(os.path.abspath(__file__))
    File       = os.path.join(WorkingDir, 'dataset', 'development.npy')
    #log loading dataset x...
    Dataset    = numpy.load(File)
    File       = os.path.join(WorkingDir, 'dataset', 'development.csv')
    Metadata   = pandas.read_csv(File)
    print('[Info] Opened dataset has ' + str(Dataset.size) + ' ' + str(Dataset.dtype) + ' ' + str(Dataset.itemsize) + ' byte entries; is ' + '{:.1f}'.format(Dataset.nbytes / 1048576) + ' MiB long; and contains ' + str(Dataset.shape[0]) + ' snippets.')

    # Order the dataset with the given metadata
    Tables    = ParseMetadata(Metadata)
    WordCount = Tables['WordCount']
    WordDict  = Tables['WordDict']

    DatasetTables = PartitionDataset(Tables)
    BuildValidationVectors(WordCount, DatasetTables, WordDict)
    # Helper functions
    def GenerateWorkerID(WorkerCount):
        for Index in range(WorkerCount):
            WorkerID = '%0x' % random.getrandbits(16)
            while WorkerID in list(WorkerDict.values()):
                WorkerID = '%0x' % random.getrandbits(16)

            WorkerNumber = len(WorkerDict) + 1
            WorkerDict.update([(WorkerNumber, WorkerID)]) # Update with a key, value tuple

            yield (WorkerNumber, WorkerID)

    def GenerateTaskID(*, Stage = 0, TaskCount = 1, TaskDict):
        for Index in range(TaskCount):
            TaskID = '%0x' % random.getrandbits(16)
            while TaskID in list(WorkerDict.values()):
                TaskID = '%0x' % random.getrandbits(16)

            TaskDict.update([(TaskID, Stage)]) # Update with a key, value tuple
            yield TaskID

    def TaskEvent(Result):
        print('debug task event' + str(Result))
        Word     = Result[0]
        TaskID   = Result[1]
        WorkerID = Result[2]

        match Stage:
            case 1:
                String = 'Reordering feature data'
        
        #ETAString = GetETA(Rebase = False, Workload = len(WordIDTable[Word]) / len(WordIDTable))
        # log here



    # Setup a worker pool
    Stage         = 1

    # Allocate & setup shared memory
    BlowupFactor  = 1 # todo compute actual size of all new numpy features (derivatives and what not). do this per snippet and then multiply by snippet count + safety factor
        # out of memory guard, cancel execution of system memory is too small
    DatasetMemory = multiprocessing.shared_memory.SharedMemory(name = 'DatasetMem', create = True, size = Dataset.nbytes)
    global SharedDataset
    SharedDataset = numpy.ndarray(shape = Dataset.shape, dtype = Dataset.dtype, buffer = DatasetMemory.buf)
    numpy.copyto(dst = SharedDataset, src = Dataset, casting = 'safe')
    print('[Info] Copied the source dataset to shared memory.')
    print(DatasetMemory.buf) # memory view object documentation
    del(Dataset) # Mark the now duplicate dataset obsolete to reclaim some memory

    Mems = list()
    for WordNum in range(WordCount + 2):
        Mems.append(list())
        #for Dataset in range(2):
            #for Feature in range(2):
        Shape = tuple([len(DatasetTables[0][WordNum]), 44 * 64])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Train' + '-' + 'Melspect'
        Mems[WordNum].append(tuple([String, Shape]))
        Shape = tuple([len(DatasetTables[1][WordNum]), 44 * 64])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Test' + '-' + 'Melspect'
        Mems[WordNum].append(tuple([String, Shape]))
        Shape = tuple([len(DatasetTables[2][WordNum]), 44 * 64])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Validate' + '-' + 'Melspect'
        Mems[WordNum].append(tuple([String, Shape]))
        
        Shape = tuple([len(DatasetTables[0][WordNum]), 44 * 32])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Train' + '-' + 'MFCC'
        Mems[WordNum].append(tuple([String, Shape]))
        Shape = tuple([len(DatasetTables[1][WordNum]), 44 * 32])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Test' + '-' + 'MFCC'
        Mems[WordNum].append(tuple([String, Shape]))
        Shape = tuple([len(DatasetTables[2][WordNum]), 44 * 32])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Validate' + '-' + 'MFCC'
        Mems[WordNum].append(tuple([String, Shape]))
        
        Shape = tuple([len(DatasetTables[0][WordNum]), 44 * 44])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Train' + '-' + 'ZcrBand'
        Mems[WordNum].append(tuple([String, Shape]))
        Shape = tuple([len(DatasetTables[1][WordNum]), 44 * 44])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Test' + '-' + 'ZcrBand'
        Mems[WordNum].append(tuple([String, Shape]))
        Shape = tuple([len(DatasetTables[2][WordNum]), 44 * 44])
        String = 'ResultMemory-' + DictKey(WordDict, WordNum) + '-' + 'Validate' + '-' + 'ZcrBand'
        Mems[WordNum].append(tuple([String, Shape]))

    iMems = list() #initialised mems
    for WordNum in range(len(Mems)):
        iMems.append(list())
        for Mem in Mems[WordNum]:
            Shared = multiprocessing.shared_memory.SharedMemory(name = Mem[0], create = True, size = (Mem[1][0] * Mem[1][1]) * 8)
            print('Created buffer ' + str(((Mem[1][0] * Mem[1][1]) * 8) / 1048576) + ' MiB long')
            Numpy  = numpy.ndarray(shape = Mem[1], dtype = numpy.double, buffer = Shared.buf)
            iMems[WordNum].append(tuple([Shared, Numpy]))

    # Only works with offline data
    print('Starting offline ML')
    ML(WordCount, DatasetTables, WordDict, iMems, Mems)
    # COPYPASTE
    for WordNum in range(WordCount + 2):
        for Memory in [Tuple[0] for Tuple in iMems[WordNum]]:
              Memory.close()
    DatasetMemory.close()
    raise SystemExit
    # end COPYPASTE
    # note for many shared arrays: (which is the better approach!:) due to the difficulties of passing data to one individual worker via the initialiser and due to the fact that i dont know which worker gets which tasks, the best way would be to setup the shared mem at the bebinning of each workers task function with a known name (so resultMemory-21-2) for word 21 and feature 2
    # will need resultmem 1-22, features 1-3 (spect, mfccd2, zcr - bandwidth 2d merge)
    # each worker will need to keep a record of which numpy array position is which snippet. these tables then need to be used to write the final 198 [(22*3) features*3 train/test/valid] numpy arrays to the disk. before writing to the disk, each numpy array has to be randomised once more so that the same speaker is not adjacent to itself
    print('[Info] Setup shared memory buffers for the source and processed datasets; ' + '{:.1f}'.format(SharedDataset.nbytes / 1048576) + ' and ' + '{:.1f}'.format((SharedDataset.nbytes * BlowupFactor) / 1048576) + ' MiB long respectively.')

    # get the number of system processes. multiply actual worker count by global factor.
    WorkerCount = int(18 * Config['SystemThreadMul'])
    WorkerIDs   = list(GenerateWorkerID(WorkerCount))
    SManager    = multiprocessing.Manager()
    SharedList  = SManager.list(WorkerIDs)
    Workers     = multiprocessing.Pool(processes = WorkerCount, initializer = WorkerSetup, initargs = (SharedList, Mems, DatasetTables, SharedDataset.shape))
    print('[Info] Spawned ' + str(WorkerCount) + ' worker processes.')
    print(SharedList)

    GetETA(Rebase = True) # Setup the reference timestamp
    Results       = list()
    TaskDict      = dict()
    # build a new list for words + noise + other. also build a list with the relative sizes (in % of total) for eta
    for WordNum, TaskID in zip(range(WordCount + 2), GenerateTaskID(Stage = Stage, TaskCount = WordCount + 2, TaskDict = TaskDict)):
        DeferredResult = Workers.apply_async(func = WorkerTest, args = (WordNum, TaskID), callback = TaskEvent)
        #DeferredResult = Workers.apply_async(func = BuildDatasets, args = (WordNum, TaskID), callback = TaskEvent)
        Results.append(DeferredResult)
    
    [DeferredResult.wait() for DeferredResult in Results] # Wait for all workers to finish their assigned tasks.

    for WordNum in range(WordCount + 2):
        BuildDatasets(WordNum, 0, iMems, DatasetTables)
        for Elem in range(len(iMems[WordNum])):
            print('[Info] Writing ' + Mems[WordNum][Elem][0])
            numpy.save(file = Mems[WordNum][Elem][0], arr = iMems[WordNum][Elem][1])

    #PlotSummary()

    for WordNum in range(WordCount + 2):
        for Memory in [Tuple[0] for Tuple in iMems[WordNum]]:
              Memory.close()
    DatasetMemory.close()
    Workers.close()
    Workers.join()
    print('Completed')

def ML(WordCount, DatasetTables, WordDict, iMems, Mems):
    Vlen = (44 * 64) + (44 * 32) # + (44 * 44)
    TrainIdx = 0
    TestIdx = 0
    ValidIdx = 0
    DatasetSnippets = list()
    for Dataset in range(len(DatasetTables)):
        DatasetSnippets.append(0)
        for Word in DatasetTables[Dataset]:
            DatasetSnippets[Dataset] += len(Word)
    DatasetSnippets[0] = int(DatasetSnippets[0] * 1.2)
    DatasetSnippets[1] = int(DatasetSnippets[1] * 1.2)
    DatasetSnippets[2] = int(DatasetSnippets[2] * 1.2)
    Training  = numpy.ndarray(shape = tuple([DatasetSnippets[0], Vlen]))
    TrainingV = numpy.ndarray(shape = tuple([DatasetSnippets[0], 10]))
    Testing  = numpy.ndarray(shape = tuple([DatasetSnippets[1], Vlen]))
    TestingV = numpy.ndarray(shape = tuple([DatasetSnippets[1], 10]))
    Validation  = numpy.ndarray(shape = tuple([DatasetSnippets[2], Vlen]))
    ValidationV = numpy.ndarray(shape = tuple([DatasetSnippets[2], 10]))
    for WordNum in range(WordCount + 2):
        for DSFeature in range(0, 3):
            Filename = Mems[WordNum][DSFeature][0] + '.npy'
            Dat1 = numpy.load(file = Filename)
            Filename = Mems[WordNum][DSFeature + 3][0] + '.npy'
            Dat2 = numpy.load(file = Filename)
            Shape = Dat1.shape[0]
            for Snippet in range(Shape):
                if DSFeature in (0, 3):
                    #Training = numpy.append(arr = Training, values = numpy.load(file = Filename))
                    #if TrainIdx == 18000:
                    #    breakpoint()
                    Training[TrainIdx] = numpy.concatenate([Dat1[Snippet], Dat2[Snippet]])
                    TrainIdx += 1
                if DSFeature in (1, 4):
                    #Testing = numpy.append(arr = Testing, values = numpy.load(file = Filename))
                    Testing[TestIdx] = numpy.concatenate([Dat1[Snippet], Dat2[Snippet]])
                    TestIdx += 1
                if DSFeature in (2, 5):
                    #Validation = numpy.append(arr = Validation, values = numpy.load(file = Filename))
                    Validation[ValidIdx] = numpy.concatenate([Dat1[Snippet], Dat2[Snippet]])
                    ValidIdx += 1
        #numpy.concatenate(iMems[WordNum][0][1], iMems[WordNum][3][1], axis = 1, out = Training)
        #numpy.concatenate(iMems[WordNum][1][1], iMems[WordNum][4][1], axis = 1, out = Testing)
        #numpy.concatenate(iMems[WordNum][2][1], iMems[WordNum][5][1], axis = 1, out = Validation)
    DatasetStr = ['Train', 'Test', 'Valid']
    TrainIdx = 0
    TestIdx = 0
    ValidIdx = 0
    for Dataset in range(len(DatasetTables)):
        for WordNum in range(WordCount + 2):
            Filename = 'Vectors-' + DatasetStr[Dataset] + '-' + DictKey(WordDict, WordNum) + '.npy'
            Vec = numpy.load(file = Filename)
            Shape = Vec.shape[0]
            for Snippet in range(Shape):
                if Dataset == 0:
                    #TrainingV = numpy.append(arr = TrainingV, values = numpy.load(file = Filename))
                    TrainingV[TrainIdx] = Vec[Snippet]
                    TrainIdx += 1
                if Dataset == 1:
                    #TestingV = numpy.append(arr = TestingV, values = numpy.load(file = Filename))
                    TestingV[TestIdx] = Vec[Snippet]
                    TestIdx += 1
                if Dataset == 2:
                    #ValidationV = numpy.append(arr = ValidationV, values = numpy.load(file = Filename))
                    ValidationV[ValidIdx] = Vec[Snippet]
                    ValidIdx += 1
    print('Training...')
    TrainingWordNums = [2, 4, 5, 6, 8, 9, 11, 12, 15, 19]
    WordStrList = list()
    for n in TrainingWordNums:
        WordStrList.append(DictKey(WordDict, n))
    for i in range(1, 50, 5):
        print('k = ' + str(i))
        var1 = kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i)
        var2 = kNN.fit(Training, TrainingV)
        var3 = kNN.predict(Testing)
        var4 = kNN.predict(Validation)
        var5 = kNN.score(Validation, ValidationV)
        print(sklearn.metrics.classification_report(y_true = ValidationV, y_pred = var4, target_names = WordStrList))
        print(str(var5) + ' k = ' + str(i))
    print('training done!')
    return
    

# Entry
if __name__ == '__main__':
    multiprocessing.freeze_support() # Support frozen windows executables
    if multiprocessing.get_start_method(allow_none = False) != 'spawn':
        multiprocessing.set_start_method('spawn')
    Main()
