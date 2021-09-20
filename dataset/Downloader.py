# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 02:27:23 2021

Custom Downloader for xR-EgoPose dataset.

I was having problems with the downloader shell script that came with
xR-EgoPose, so I made my own.

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
@author: Andrew Huard
"""

from wget import download


class downloader:
    # Create a downloadfile method
    # Accepting the url and the file storage location
    # Set the location to an empty string by default.
    def downloadFile(self, url, dest=""):
         # Download file and with a custom progress bar
        download(url, out = dest)
        

dataset_dir = "A:\\xR-EgoPose\\data\\Dataset\\"
train_dir = dataset_dir + "TrainSet\\"
test_dir = dataset_dir + "TestSet\\"
val_dir = dataset_dir + "ValSet\\"

fileDef_val = list([("male_008_a_a", 'i')])

fileDef_test = list([("female_004_a_a", 'i'),
                   ("female_008_a_a", 'i'),
                   ("female_010_a_a", 'i'),
                   ("female_012_a_a", 'f'),
                   ("female_012_f_s", 'a'),
                   ("male_001_a_a", 'i'),
                   ("male_002_a_a", 'j'),
                   ("male_004_f_s", 'a'),
                   ("male_006_a_a", 'i'),
                   ("male_007_f_s", 'a'),
                   ("male_010_a_a", 'i'),
                   ("male_014_f_s", 'a')])

fileDef_train = list([("female_001_a_a", 'i'),
                   ("female_002_a_a", 'j'),
                   ("female_002_f_s", 'a'),
                   ("female_003_a_a", 'f'),
                   ("female_005_a_a", 'i'),
                   ("female_006_a_a", 'i'),
                   ("female_007_a_a", 'h'),
                   ("female_009_a_a", 'i'),
                   ("female_011_a_a", 'f'),
                   ("female_014_a_a", 'f'),
                   ("female_015_a_a", 'j'),
                   ("male_003_f_s", 'a'),
                   ("male_004_a_a", 'i'),
                   ("male_005_a_a", 'j'),
                   ("male_006_f_s", 'a'),
                   ("male_007_a_a", 'i'),
                   ("male_008_f_s", 'a'),
                   ("male_009_a_a", 'h'),
                   ("male_010_f_s", 'a'),
                   ("male_011_f_s", 'a'),
                   ("male_014_a_a", 'i')])


def process(fileDef_list, dest_str):
    downloadObj = downloader()
    
    for _, fileDef_tup in enumerate(fileDef_list):
        prefix, maxIndex = fileDef_tup
        
        index_list = []
        alpha = 'a'
        for _ in range(0,26):
            index_list.append(alpha)
            if alpha == maxIndex:
                break
            alpha = chr(ord(alpha)+1)
            
        
        for _, index in enumerate(index_list):
            url="https://github.com/facebookresearch/xR-EgoPose/releases/download/v1.0/{}.tar.gz.parta{}".format(prefix, index)
            print("Downloading: " + url)
            downloadObj.downloadFile(url, dest_str)


#process(fileDef_test, test_dir)
#process(fileDef_train, train_dir)
process(fileDef_val, val_dir)