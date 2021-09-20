# -*- coding: utf-8 -*-
r"""
Created on Mon Sep 13 01:48:39 2021

Class for parsing and accessing an xR-EgoPose Dataset Tar-Archive File.

The Tar-Archived version of xR-EgoPose is an extension of the dataset to reduce
the number of files stored, which is desirable on some cluster-based high
performance computing systems.

The xR-EgoPose Dataset (Denis Tome, 2019)
https://github.com/facebookresearch/xR-EgoPose

Steps to prepare the xR-EgoPose dataset:
    1) Download the dataset by running Downloader.py
    2) Unarchive the dataset by running the bash script ./unarchive.sh
    3) Re-archive the dataset in TAR format by running archive.py

Download and Unarchive:
    Download this dataset using Downloader.py, then unarchive using the
    bash script, unarchive.sh. To run the script, navidate to the root
    directory of the project and type: ./unarchive.sh

Re-Archive:
    This implementation requires a TAR archived version of the xR-EgoPose
    dataset.  First download and unarchive as described above, then run
    archive.py to convert the dataset to uncompressed TAR archives.

@author: Andrew Huard
"""
import os
import tarfile
import pickle
from pathlib import PurePosixPath, Path
from treelib import Tree


class Tar():
    r"""
    A class representing a tar file and its contents.  This class stores
    pointers in memory that may be used to access data directly from within a
    TAR archive without extracting (unarchiving) the entire tar file, a very
    expensive operation this implementation avoids.

    A treelib file tree is created when parsing a new tar archive, which
    attaches references in memory (tarinfo) to each leaf in the tree.  This
    tarinfo object is used to quickly and seamlessly access files within the
    archive without reparsing or extracting the entire archive's contents.

    The initial parsing operation creates a pickled byte file representing the
    tar file's structure including file pointers to file within the archive.
    If present, this pickled byte file is loaded, bypassing the archive parsing
    operation (an expensive, slow process).  The result is near-instantaneous
    initialization after the first, slow initialization.  The byte file itself
    stores the treelib tree object, which includes tarinfo objects attached to
    each leaf's data parameter.
    """

    def __init__(self, tarfilepath):
        """
        Initialization member for the Tar class.

        Checks whether a pickled byte file representing the tar archive is
        present and bypasses parsing the archive's file tree if found.  If a
        pickled byte file is not present, parse the archive (an expensive
        operation).  The pickled byte file has the extention: .tree

        Parameters
        ----------
        tarfilepath : string
            Path to a tar archive.

        Returns
        -------
        None.

        """
        self.path = Path(tarfilepath)  # Set the path to the tar archive
        self.tar_obj = tarfile.open(tarfilepath, "r")  # Open the tar archive file for reading
        if self.GetTreeFilePath().is_file():  # Check for a pickle file of the file tree
            self.LoadTree()  # Pickle file found, load the tree and bypass parsing the archive
        else:  # Pickle file not found, proceed to parse the archive file tree
            self.GenerateTreeTAR()  # Parse the file tree, an expensive operation
            self.SaveTree()  # Save the treelib tree object as a pickled byte file for faster loading in the future

    def GenerateTreeTAR(self):
        """
        Parses the tar archive and generates the treelib tree object
        representing the files in the archive.  A reference in memory to each
        object in the archive (tarinfo objects) are stored on each leaf of the
        tree in the data parameter.

        The treelib tree object respresents the data model of the tar archive.

        Returns
        -------
        None.

        """
        print("Parsing File: " + self.path.parts[-1])  # Output the filename being parsed
        count = 0  # Counter for displaying updates on file parsing progress
        self.tree = Tree()  # Instantiate a new treelib tree object
        for tarinfo in self.tar_obj:  # Iterate over the objects in the tar archive.  This is the only fast method found by the author to efficiently access the members of a tar file.
            path_str = tarinfo.name  # Get the path string of the item (file or directory) within the archive.  The path is relative to within the archive.
            path = PurePosixPath(path_str)  # Use pathlib to ensure the path is a good representation and to gain access to members such as path.parts
            if tarinfo.isdir() or tarinfo.isreg():  # Exclude symbolic links and corrupt files/directories
                count += 1  # Increment item counter
                if count % 2500 == 0:  # Display the number of items processed every 2500 items
                    print("Item " + str(count))
                parts = path.parts  # Get the tuple: prefix (parent directory path) and suffix (current item) of the file path
                if tarinfo.name:  # Not the root element
                    self.tree.create_node(tag=str(parts[-1]), identifier=str(path), parent=str(path.parent), data=tarinfo)  # Tree Node Identifier is always the Posix path, attach the tarinfo pointer as a data parameter of the tree node
                else:  # The root directory is the only one with a blank name (name is None)
                    self.tree.create_node(tag=str(path), identifier=str(path), parent=None, data=tarinfo)  # Root Node

    def extractfile(self, tarinfo):
        """
        Passthrough method for accessing the tarfile object extractfile()
        method.  This function provides a quick method for extracting a single
        file from an archive without unarchiving it first.  File extraction is
        almost instantaneous using this method versus waiting several minutes
        to extract the entire archive.

        Parameters
        ----------
        tarinfo : tarfile tarinfo object
            A pointer in memory to objects within a tar archive.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.tar_obj.extractfile(tarinfo)

    def GetTreeFilePath(self):
        """
        Generates a default file name and path for the pickled byte file
        storing the contents of the tarfile tree and corresponding tarinfo
        pointers.

        Returns
        -------
        pathlib Path
            Default filename and path to file tree pickled byte file, which
            allows bypassing of expensive file and directory parsing operations
            within the tar archive.

        """
        return Path(os.path.splitext(str(self.path))[0] + '.tree')

    def SaveTree(self, savepath_pickle=None):
        """
        Saves a pickled byte file containing the tar archive's file tree and
        associated tarinfo pointers to each object in the archive.

        Parameters
        ----------
        savepath_pickle : string, optional
            Destination filename and path to file tree pickled byte file. The
            default is None.  If unset, the default path is used.

        Returns
        -------
        None.

        """
        if savepath_pickle is None:
            savepath_pickle = self.GetTreeFilePath()
        with open(savepath_pickle, "wb") as file:
            pickle.dump(self.tree, file)

    def LoadTree(self, loadpath_pickle=None):
        """
        Loads a pickled byte file containing the tar archive's file tree and
        associated tarinfo pointers to each object in the archive.

        Parameters
        ----------
        loadpath_pickle : string, optional
            Destination filename and path to file tree pickled byte file. The
            default is None.  If unset, the default path is used.

        Returns
        -------
        None.

        """
        if loadpath_pickle is None:
            loadpath_pickle = self.GetTreeFilePath()
        with open(loadpath_pickle, 'rb') as file:
            self.tree = pickle.load(file)

    def GetSubTree(self, subroot_nid):
        """
        Passthrough function used to retrieve a subtree of a treelib tree 
        object.  This function is used to reference subdirectories and their
        corresponding tarinfo pointers attached to each leaf of the tree.

        Parameters
        ----------
        subroot_nid : string
            Unique node identifier corresponding to the filepath in the tar
            archive.

        Returns
        -------
        treelib tree
            A subtree of a treelib tree corresponding to a subdirectory of the
            tar archive.

        """
        return self.tree.subtree(subroot_nid)

    def GetLeaves(self, subroot_nid):
        """
        Passthrough function used to retrieve the Nodes of a treelib tree 
        object.  This function is used to reference subdirectories and their
        corresponding tarinfo pointers attached to each leaf of the tree.
        
        Example Output:
            Node(tag=male_008_a_a.depth.000001.png,
                 identifier=env_001/cam_down/depth/male_008_a_a.depth.000001.png,
                 data=<TarInfo 'env_001/cam_down/depth/male_008_a_a.depth.000001.png' at 0x1ea7f55d7c0>)
                
        Parameters
        ----------
        subroot_nid : string
            Unique node identifier corresponding to the filepath in the tar
            archive.

        Returns
        -------
        treelib Node object
            A treelib node object corresponding to files and directories in the
            tar archive.  Each node contains a data parameter that stores a
            TarInfo object, which is used to quickly reference or extract the
            file in the tar archive.  The Node identifier is the string path to
            the corresponding file or directory within the tar archive.

        """
        return self.tree.leaves(subroot_nid)

    def listPaths(self, level=2):
        """
        Provides a string list of filepaths within the tar archive with
        corresponding level.  For example, level 2 corresponds to the
        directories:
            ["env_001/cam_down", "env_002/cam_down", "env_003/cam_down"]

        Parameters
        ----------
        level : integer, optional
            Directory depth level to search. The default is 2.

        Returns
        -------
        paths : string list
            A list of files or directories with corresponding depth level.

        """
        paths = list()
        for node in self.tree.filter_nodes(func=lambda x: self.tree.depth(x.identifier) == level):
            paths.append(node.identifier)
        return paths

    def Close(self):
        """
        Close the tar file in the filesystem.  The tar file cannot be deleted
        or moved while open.  Normally, the tar file remains open for the
        duration of this script because data may be retrieved, Just In Time
        (JIT), as needed.

        Returns
        -------
        None.

        """
        self.tar_obj.close()

    def printTree(self, levels=3):
        """
        A diagnostic function that prints the contents of the tree, up to a
        specified depth level.

        Parameters
        ----------
        level : integer, optional
            Directory depth level to search. The default is 3.

        Returns
        -------
        None.

        """
        self.tree.show(filter=lambda x: self.tree.depth(x.identifier) <= levels)


def main():
    """
    Main function used for diagnostic testing of the Tar Class.  Main executes
    if python directly calls this class file.  Several use cases are shown
    here as examples.
    
    This function is intended for diagnostic purposes only.  This function is
    completely unused by PyTorch DataLoader.
    """
    tarpath = r"A:\xR-EgoPose\data\Dataset\ValSet\male_008_a_a.tar"
    tar = Tar(tarpath)
    
    print("Testing Tar File: " + tarpath)
    tar.printTree(levels=3)
    print(tar.listPaths(level=2))
    print("\n")
    print(tar.GetLeaves(tar.listPaths(level=2)[0])[0])
    
    tar.Close()


if __name__ == "__main__":
    """
    Main Function executes if python executes directly on this class file.
    This function is intended for diagnostic purposes only.  This block is
    completely unused by PyTorch DataLoader.
    """
    main()
