# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 07:12:06 2021

A test script for extracting data from a TAR file archive of the xR-EgoPose
dataset.

@author: Andrew Huard
"""

import tarfile
from PIL import Image


tarpath = r"A:\xR-EgoPose\data\Dataset\ValSet\male_008_a_a.tar"
tar = tarfile.open(tarpath, "r")

# Example of Extracting a File
member = tar.getmember("env_001/cam_down/depth/male_008_a_a.depth.000001.png")  # slow. don't use getmember. good for example only
f=tar.extractfile(member)
img = Image.open(f)
img.show()


# Example of Parsing Directories Quickly
for tarinfo in tar:
    if tarinfo.isdir():
        print("Directory: " + tarinfo.name)
tar.close()

