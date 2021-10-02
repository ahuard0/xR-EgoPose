# -*- coding: utf-8 -*-
r"""
Created on Mon Sep 20 04:09:59 2021

A test script for working with the xR-EgoPose dataset.

For fast reuse of loaded datasets in memory, configure the python runtime to
use the current namespace of the console.  Then, if the dataset is already
defined, it will not be defined again.

@author: Andrew Huard
"""

from dataset.xrEgoPose.xrEgoPose import xrEgoPose

try:
    dataset
except NameError:
    dataset = xrEgoPose(bool_train=True, bool_tensor=False,
                        str_rootdir=r'A:\xR-EgoPose\data\Dataset\TrainSet',
                        bool_json=True, bool_rot=False, bool_rgba=False,
                        bool_worldp=False, bool_depth=False,
                        bool_objectId=False, bool_tarArchive=True,
                        bool_camera=True)

print("Length: " + str(len(dataset)))

print(dataset[134788]['data']['camera']['char_height'])