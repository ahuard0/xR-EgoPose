# -*- coding: utf-8 -*-
r"""
Created on Sun Sep 12 02:09:26 2021

The xR-EgoPose dataset consists of hundreds of thousands of files, which cannot
be stored on a remote server except at great cost ($$$ per file).  Therefore,
we archive each subset of the greater dataset to reduce the number of files
stored on the server.  The original dataset directories are deleted to save
disk space.

Before running this script, follow the download and unpacking instructions:

Download and Unpacking: 
        Download this dataset using Downloader.py, then unarchive using the
        bash script, unarchive.sh. To run the script, navidate to the root
        directory of the project and type: ./unarchive.sh
    
Running this script:
    1) Change the root directory for each dataset to correspond to your
       filesystem at the bottom of this script file.
    2) Directly run this file in python.

Output Archive Format:
    uncompressed tar files

Example:
    1) Specify Root Directory: A:\xR-EgoPose\data\Dataset\ValSet
    2) Processes subdirectory of name: male_008_a_a
    3) Creates an archive file: male_008_a_a.tar
    4) Deletes the original data directory: male_008_a_a

@author: Andrew Huard
"""

import tarfile
from pathlib import Path
import os
import shutil


def archiveFolders(rootdir):
    r"""
    Archives all subdirectories, creating a tar file with the same name as each
    directory archived.  The original data is deleted upon successful archive
    creation.

    Parameters
    ----------
    rootdir : string
        Path to the root directory for archiving all immediate subdirectories.

    Returns
    -------
    None.

    """
    checkDirectoryTree(rootdir, bool_removeCorrupt=True)
    root = Path(rootdir)
    dirpaths = [f for f in root.iterdir() if f.is_dir()]
    for _, dirpath in enumerate(dirpaths):
        print("Archiving: " + str(dirpath))
        archiveFolder(dirpath)
        deleteFolder(dirpath)


def archiveFolder(source_dirpath):
    r"""
    Creates a RAR archive of the source directory.

    Parameters
    ----------
    source_dirpath : string
        Path to the directory to be archived.

    Returns
    -------
    None.

    """
    dest_filepath = Path(str(source_dirpath) + '.tar')
    print("Saving: " + str(dest_filepath))
    with tarfile.open(str(dest_filepath), 'w') as tar:
        tar.add(str(source_dirpath), arcname=os.path.sep)
        tar.close()
        
def deleteFolder(dirpath):
    r"""
    Deletes a folder and all its' contents.

    Parameters
    ----------
    dirpath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    shutil.rmtree(str(dirpath))
    
    
def checkDirectoryTree(rootpath, bool_removeCorrupt=False):
    r"""
    Recursively checks every subdirectory to ensure it is actually a directory.
    
    If a directory is corrupted, the directory is removed if it is empty.
    Corrupted directories do not pass os.path.isdir() checks despite os.walk()
    believing it is a directory. Corrupted directory removal may be suppressed
    using bool_removeCorrupt=False (Default).

    Parameters
    ----------
    rootpath : string
        The root directory where subdirectories will be checked for integrity.
    bool_removeCorrupt : boolean, optional
        Flag to suppress corrupt directory removal. The default is False.

    Returns
    -------
    None.

    """
    print("Starting Directory Check: " + str(rootpath))
    for (path,dirs,files) in os.walk(rootpath):
        for d in dirs:
            p = Path(path,d)
            if os.path.isdir(str(p)) is not True:  # Corrupt directory found
                if bool_removeCorrupt:
                    print("Removing Corrupted Directory: " + str(p))
                    Path.rmdir(p)  # Remove empty, corrupted directory
                else:
                    print("Found Corrupted Directory: " + str(p))


if __name__ == "__main__":
    r"""
    Instructions:
        Change the dataset directories here to correspond with your filesystem.
        Then, directly run this python file.
    """
    archiveFolders(r"A:\xR-EgoPose\data\Dataset\ValSet")
    archiveFolders(r"A:\xR-EgoPose\data\Dataset\TrainSet")
    archiveFolders(r"A:\xR-EgoPose\data\Dataset\TestSet")