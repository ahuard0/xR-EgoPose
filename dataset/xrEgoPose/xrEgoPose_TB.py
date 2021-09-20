# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 04:09:59 2021

A test script for working with the xR-EgoPose dataset.

@author: Andrew Huard
"""

from dataset.xrEgoPose.xrEgoPose import xrEgoPose

dataset = xrEgoPose(bool_train=False, bool_tensor=False,
                    str_rootdir=r'A:\xR-EgoPose\data\Dataset\ValSet',
                    bool_json=True, bool_rot=True, bool_rgba=True,
                    bool_worldp=True, bool_depth=True, bool_objectId=True,
                    bool_tarArchive=True)

#dataset.printDatasetSampleDictRange(13, 15)

#dataset[0]['data']['depth'].show()
dataset[5201]['data']['depth'].show()
#dataset[10301]['data']['depth'].show()

#dataset[0]['data']['rgba'].show()
#dataset[5201]['data']['rgba'].show()
#dataset[10301]['data']['rgba'].show()

#print(dataset[0]['data']['rot'])
#print(dataset[5201]['data']['rot'])
#print(dataset[10301]['data']['rot'])

#dataset[1000]['data']['depth'].show()

#print(dataset[10301]['data'].keys())
#dataset.GetDatasetFilePath()

#dataset[10301]['data']['worldp'].show()

#dataset.ToString()
