# -*- coding: utf-8 -*-
r"""
Created on Wed Sep  8 23:48:10 2021

xR-EgoPose Dataset Implementation, Built from Scratch
    xR-EgoPose dataset downloaded locally to A:\xR-EgoPose\data\Dataset

This dataset is an implementation of a PyTorch Dataset, which is intended to be
used with a PyTorch DataLoader.

@author: Andrew Huard
"""

import json
from pathlib import Path, PurePosixPath
import os.path
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.xrEgoPose.Tar import Tar
from treelib.exceptions import NodeIDAbsentError


class xrEgoPose(Dataset):
    r"""
    The xR-EgoPose Dataset (Denis Tome, 2019)
    https://github.com/facebookresearch/xR-EgoPose

    Steps to prepare the dataset:
        1) Download the dataset by running Downloader.py
        2) Unarchive the dataset by running the bash script ./unarchive.sh
        3) Re-archive the dataset in TAR format by running archive.py

    Download and Unarchive:
        Download this dataset using Downloader.py, then unarchive using the
        bash script, unarchive.sh. To run the script, navigate to the root
        directory of the project and type: ./unarchive.sh

    Re-Archive:
        This implementation requires a TAR archived version of the xR-EgoPose
        dataset.  First download and unarchive as described above, then run
        archive.py to convert the dataset to uncompressed TAR archives.
    """

    def __init__(self, str_rootdir=r'A:\xR-EgoPose\data\Dataset\ValSet',
                 bool_train=False, bool_tensor=False, bool_depth=False,
                 bool_json=False, bool_objectId=False, bool_rgba=True,
                 bool_rot=False, bool_worldp=False, bool_tarArchive=True):
        r"""
        Initialization function for the dataset.  The user must specify a root
        directory of the dataset in their filesystem.  The xR-EgoPose datasets
        available are: TestSet, TrainSet, ValSet.

        Parameters
        ----------
        str_rootdir : string, optional
            The path to the root directory of the dataset.  The default value
            may be changed by the user to correspond with their filesystem.
            The default value is: A:\xR-EgoPose\data\Dataset\ValSet
        bool_train : TYPE, optional
            A boolean switch used to specify either a training dataset (True)
            or a validation dataset (False). This value is provided for
            reference by others using this script, such as by a PyTorch model.
            The default is False.
        bool_tensor : TYPE, optional
            A boolean switch used to output either a PIL image (False) or a
            PyTorch Tensor (True) when retrieving an image from the dataset.
            The default is False.
        bool_depth : TYPE, optional
            A boolean switch used to output depth image data. The default is
            False.
        bool_rgba : TYPE, optional
            A boolean switch used to output RGBA image data. The default is
            False.
        bool_objectId : TYPE, optional
            A boolean switch used to output ObjectID image data. The default is
            False.
        bool_worldp : TYPE, optional
            A boolean switch used to output WorldP image data. The default is
            False.
        bool_json : TYPE, optional
            A boolean switch used to output position data. The default is
            False.
        bool_rot : TYPE, optional
            A boolean switch used to output rotation data. The default is
            False.
        bool_tarArchive : TYPE, optional
            A boolean switch used to specify the source of raw data for the
            dataset.  When set to false, the dataset is retrieved from
            unarchived folders the way Denis Tome organized it. When set to
            True, the dataset is received from TAR archives (the new method).
            The default is True.

        Returns
        -------
        None.

        """
        self.bool_train = bool_train
        self.str_rootdir = str_rootdir
        self.bool_tensor = bool_tensor
        self.bool_depth = bool_depth
        self.bool_json = bool_json
        self.bool_objectId = bool_objectId
        self.bool_rgba = bool_rgba
        self.bool_rot = bool_rot
        self.bool_worldp = bool_worldp
        self.bool_tarArchive = bool_tarArchive

        # Initialize
        self.tar_dict = dict()  # Used to store references to Tar Class objects.  Key is the string tar filepath.
        self.sample_dict = dict()  # Multilevel dictionary with integer primary index key value corresponding to samples in the dataset.
        self.path_list = list()  # List of paths to level 2 directories within the dataset: e.g., "env_001\cam_down"
        
        if self.bool_tarArchive:
            self.instantiateTarFromPathList()  # Opens each tar archive file and stores a handle in the corresponding Tar Class instantiation

        if self.GetDatasetFilePath().is_file():  # Checks if a '.data' file is present corresponding to the name of the dataset.
            self.LoadDataset()  # Reuse prior parsing results, bypassing the parsing operation
        else:
            self.generateDatasetFileDict()  # Parse Files (Get Dictionary of Files in Subdirectories)
            self.SaveDataset()  # Save parsing results for later reuse, bypassing the parsing operation in the future

    def __len__(self):
        """
        A required method by PyTorch Datasets, which is used to interface
        with a PyTorch DataLoader.

        Usage:
            length = len(dataset)

        Returns
        -------
        count : integer
            Returns the number of samples in the dataset.

        """
        return len(self.sample_dict)

    def __getitem__(self, ndx):
        """
        A required method by PyTorch Datasets, which is used to interface
        with PyTorch Dataloader.

        Usage:
            img, labelID = dataset[ndx]

        Parameters
        ----------
        ndx : integer
            Index of the item to be returned by the dataset.

        Returns
        -------
        output : python dictionary
            Returns a dictionary containing PIL Images, PyTorch tensors, or
            other meta information.  Data is added or omitted from the returned
            python dictionary depending on the selected boolean flags on
            initialization of the dataset.  Either PIL images or PyTorch
            tensors are returned depending on how the dataset is initialized by
            the boolean flag bool_tensor=True/False.
        labelID : integer
            The integer index of the image label, which is indexed from 0-9.
            Use getMetaLabelsList for a list of string labels ordered by ID
            index value.

        """
        item = self.sample_dict[ndx]
        item_meta = item['meta']
        
        tarpath = item_meta['path']  # pathlib Path Object
        tar_obj = self.tar_dict[str(tarpath)]
        
        item['data'] = dict()
        if self.bool_depth:
            if self.bool_tarArchive:
                item['data']['depth']['img'] = self.getImagePILfromTAR(tar=tar_obj, tarinfo=item_meta['depth'])
            else:
                item['data']['depth']['img'] = self.getImagePIL(item_meta['depth'])
            if self.bool_tensor:
                item['data']['depth']['tensor'] = self.TensorFromImagePIL(item['data']['depth']['img'])
            else:
                item['data']['depth']['tensor'] = None
        if self.bool_rgba:
            if self.bool_tarArchive:
                item['data']['rgba']['img'] = self.getImagePILfromTAR(tar=tar_obj, tarinfo=item_meta['rgba'])
            else:
                item['data']['rgba']['img'] = self.getImagePIL(item_meta['rgba'])
            if self.bool_tensor:
                item['data']['rgba']['tensor'] = self.TensorFromImagePIL(item['data']['rgba']['img'])
            else:
                item['data']['rgba']['tensor'] = None
        if self.bool_objectId:
            if self.bool_tarArchive:
                item['data']['objectId']['img'] = self.getImagePILfromTAR(tar=tar_obj, tarinfo=item_meta['objectId'])
            else:
                item['data']['objectId']['img'] = self.getImagePIL(item_meta['objectId'])
            if self.bool_tensor:
                item['data']['objectId']['tensor'] = self.TensorFromImagePIL(item['data']['objectId']['img'])
            else:
                item['data']['objectId']['tensor'] = None
        if self.bool_worldp:
            if self.bool_tarArchive:
                item['data']['worldp']['img'] = self.getImagePILfromTAR(tar=tar_obj, tarinfo=item_meta['worldp'])
            else:
                item['data']['worldp']['img'] = self.getImagePIL(item_meta['worldp'])
            if self.bool_tensor:
                item['data']['worldp']['tensor'] = self.TensorFromImagePIL(item['data']['worldp']['img'])
            else:
                item['data']['worldp']['tensor'] = None
        if self.bool_json:
            if self.bool_tarArchive:
                item['data']['json'] = self.getDataJSONfromTAR(tar=tar_obj, tarinfo=item_meta['json'])
            else:
                item['data']['json'] = self.getDataJSON(item_meta['json'])
        if self.bool_rot:
            if self.bool_tarArchive:
                item['data']['rot'] = self.getDataJSONfromTAR(tar=tar_obj, tarinfo=item_meta['rot'])
            else:
                item['data']['rot'] = self.getDataJSON(item_meta['rot'])
        return item

    def TensorFromImagePIL(self, img):
        """
        Transforms a PIL Image to a PyTorch tensor.

        Parameters
        ----------
        img : PIL Image
            A PIL Image to transform.

        Returns
        -------
        img_t : Tensor
            PyTorch tensor representing an image.

        """
        to_tensor = transforms.ToTensor()
        img_t = to_tensor(img)
        return img_t
    
    def PlotImageTensor(self, img_t):
        """
        Plots an image from a tensor.

        Parameters
        ----------
        img_t : Tensor
            PyTorch tensor representing an image.

        Returns
        -------
        None

        """
        import matplotlib.pyplot as plt
        plt.imshow(img_t.permute(1, 2, 0))  # Change C x H x W to H x W x C
        plt.show()

    def GetDatasetFilePath(self):
        """
        A pickled copy of the data pointers and meta information used by this
        dataset are saved to a file for quick reloading.  This function
        provides the filename to use for this data file.

        Returns
        -------
        output : pathlib Path object
            The filename of the data file.

        """
        root = Path(self.str_rootdir)
        parts = root.parts
        dataset_name = parts[-1]
        return Path(self.str_rootdir, dataset_name + '.data')

    def SaveDataset(self, savepath_pickle=None):
        """
        Saves the dataset meta information and TAR file pointers in a pickled
        byte file.  This file is used for fast reloading of the dataset to
        bypass time consuming parsing of the TAR archives.  To re-parse the TAR
        files and rebuild the meta information stored, simply delete the
        pickled data file.  This pickled file has the extension: .data

        Parameters
        ----------
        savepath_pickle : pathlib Path object or string path, optional
            The filepath destination to save the dataset meta file.  The
            default is None.

        Returns
        -------
        None.

        """
        if savepath_pickle is None:
            savepath_pickle = self.GetDatasetFilePath()
        with open(savepath_pickle, "wb") as file:
            pickle.dump((self.sample_dict, self.path_list), file)

    def LoadDataset(self, loadpath_pickle=None):
        """
        Loads the dataset meta information and TAR file pointers from a pickled
        byte file.  This file is used for fast reloading of the dataset to
        bypass time consuming parsing of the TAR archives.  To re-parse the TAR
        files and rebuild the meta information stored, simply delete the
        pickled data file.  This pickled file has the extension: .data

        Parameters
        ----------
        loadpath_pickle : pathlib Path object or string path, optional
            The filepath to the dataset meta file.  The default is None.

        Returns
        -------
        None.

        """
        if loadpath_pickle is None:
            loadpath_pickle = self.GetDatasetFilePath()
        with open(loadpath_pickle, 'rb') as file:
            self.sample_dict, self.path_list = pickle.load(file)

    def getImagePIL(self, filepath):
        """
        Retrieves a PIL image from the dataset.  This function is used to
        retrieve image data from non-archived datasets (e.g., unarchived raw
        datafiles organized in folders the way Denis Tome intended).  This
        function is not to be used with the TAR archived version of the dataset
        (Huard implementation).

        Parameters
        ----------
        filepath : pathlib Path object or string path
            The filepath of the data file within the dataset folder structure.

        Returns
        -------
        If self.bool_tensor=False:
            PIL Image Object
                A PIL Image representing the image data.

        If self.bool_tensor=True:
            PyTorch tensor
                A PyTorch tensor representing the image data.

        """
        if filepath is None:
            return None  # Propagate None if encountered
        return Image.open(str(Path(filepath)))

    def getImagePILfromTAR(self, tar, tarinfo):
        """
        Retrieves a PIL image from the dataset.  This function is used to
        retrieve image data from TAR-archived datasets (Huard Implementation).
        Refer to the procedure for re-archiving the xR-EgoPose dataset before
        using this function.

        Parameters
        ----------
        tar : Instantiated Tar Class Object
            An object that represents a tar file.  This parameter specifies the
            Tar.py class instantiation and is not a tarfile library object
            directly.  Tar.py is a wrapper around a tarfile library object,
            which exposes additional, custom functionality.
        tarinfo : tarinfo object from the tarfile library
            An object that represents a member of a tar file.  This parameter
            is returned while parsing the tar file directory structure and is
            stored while constructing the directory tree. It is the most
            efficient method to access files within a tar file archive that has
            been found by the author.  Tarinfo objects are attached to treelib
            nodes using the data parameter on each leaf of the tree object.

        Returns
        -------
        If self.bool_tensor=False:
            PIL Image Object
                A PIL Image representing the image data.

        If self.bool_tensor=True:
            PyTorch tensor
                A PyTorch tensor representing the image data.

        """
        if tarinfo is None:
            return None  # Propagate None if encountered
        if self.bool_tarArchive:
            f = tar.extractfile(tarinfo)
            img = Image.open(f)
            return img
        else:
            raise Exception("bool_tarArchive must be set to True")

    def getDataJSON(self, filepath):
        """
        Retrieves a JSON from the dataset as a python dictionary.  This
        function is used to retrieve JSON data from non-archived datasets
        (e.g., unarchived raw datafiles organized in folders the way Denis Tome
        intended).  This function is not to be used with the TAR archived
        version of the dataset (Huard implementation).

        Parameters
        ----------
        filepath : pathlib Path object or string path
            The filepath of the data file within the dataset folder structure.

        Returns
        -------
        Python Dictionary
            A dictionary representing the meta data stored in a JSON file.
            This file generally contains position data for each of the skeleton
            joints used in pose estimation.

        """
        if filepath is None:
            return None  # Propagate None if encountered
        with open(filepath) as file:
            data = json.load(file)
            return data

    def getDataJSONfromTAR(self, tar, tarinfo):
        """
        Retrieves a JSON data file from the dataset.  This function is used to
        retrieve JSON data from TAR-archived datasets (Huard Implementation).
        Refer to the procedure for re-archiving the xR-EgoPose dataset before
        using this function.

        Parameters
        ----------
        tar : Instantiated Tar Class Object
            An object that represents a tar file.  This parameter specifies the
            Tar.py class instantiation and is not a tarfile library object
            directly.  Tar.py is a wrapper around a tarfile library object,
            which exposes additional, custom functionality.
        tarinfo : tarinfo object from the tarfile library
            An object that represents a member of a tar file.  This parameter
            is returned while parsing the tar file directory structure and is
            stored while constructing the directory tree. It is the most
            efficient method to access files within a tar file archive that has
            been found by the author.  Tarinfo objects are attached to treelib
            nodes using the data parameter on each leaf of the tree object.

        Returns
        -------
        filepath : pathlib Path object or string path
            The filepath of the data file within the dataset folder structure.

        """
        if tarinfo is None:
            return None  # Propagate None if encountered
        if self.bool_tarArchive:
            f = tar.extractfile(tarinfo)
            data = json.load(f)
            return data
        else:
            raise Exception("bool_tarArchive must be set to True")

    def instantiateTarFromPathList(self):
        """
        Instantiate Tar Class objects from a list of filesystem paths, which
        are generated using getFilesList(rootdir, '.tar').

        Returns
        -------
        None.

        """
        tarpath_list = self.getFilesList(self.str_rootdir, ext='.tar')  # List of pathlib Paths
        
        for _, tarpath in enumerate(tarpath_list):
            self.tar_dict[str(tarpath)] = Tar(str(tarpath))

    def generateDatasetFileDict(self):
        """
        The main initialization dataset parsing function.  This function is
        computationally expensive.  This function may be bypassed if a pickled
        data file containing the python dictionary sample_dict and python list
        path_list are found.  The extension of this pickled data file is: .data

        When bool_tarArchive=False, this function parses the directory
        structure of a non-archived dataset (Denis Tome implementation).

        When bool_tarArchive=True, this function parses each TAR file,
        constructing a treelib node tree, and attaches a tarinfo pointer as a
        data parameter to each leaf of the tree.

        The output of this function is a python dictionary containing meta
        information and file pointers to data stored in the dataset.  This
        function does not retrieve data from the dataset directly; it merely
        provides the pointers and instructions necessary to retrieve the
        desired data.  The returned dictionary is indexed by a positive integer
        representing a specific sample's position in the dataset, which is the
        convention used by PyTorch datasets.  The __getitem__() member builds
        on this python dictionary to attach and return a corresponding 'data'
        key/value pair containing the Just in Time (JIT) data requested.

        Returns
        -------
        None.

        """
        self.sample_dict.clear()  # Dictionary representing the dataset
        self.path_list.clear()  # List of archive subfolders
        self.path_list = self.getDatasetDirectoriesList(self.str_rootdir)

        if self.bool_tarArchive:  # Parse TAR archives
            index = 0  # the sample index number in the PyTorch dataset
            for _, file_dict in enumerate(self.path_list):  # Iterate over archive subfolders: ["env_001\cam_down", "env_002\cam_down", "env_003\cam_down"]
                p_tarfile = file_dict['path_tarfile']  # Path to a tar file as a pathlib Path object
                tar = self.tar_dict[str(p_tarfile)]  # Instantiation of Tar.py class
                path_str = file_dict['path']  # Path to the data subfolder within the TAR archive: "env_001\cam_down"
                num_items = 0  # Initialize count to zero -> provides the maximum number of items in the archive subfolder.

                # Get RGBA Data
                try:
                    file_dict['rgba'] = tar.GetLeaves(str(PurePosixPath(path_str, 'rgba')))  # Get a list of treelib nodes corresponding to the files in the TAR archive subfolder: "env_001\cam_down\rgba"
                    num_items = max(num_items, len(file_dict['rgba']))  # Update the file count
                except NodeIDAbsentError:
                    file_dict['rgba'] = None

                # Get Depth Data
                try:
                    file_dict['depth'] = tar.GetLeaves(str(PurePosixPath(path_str, 'depth')))  # Get a list of treelib nodes corresponding to the files in the TAR archive subfolder: "env_001\cam_down\depth"
                    num_items = max(num_items, len(file_dict['depth']))  # Update the file count
                except NodeIDAbsentError:
                    file_dict['depth'] = None

                # Get WorldP Data
                try:
                    file_dict['worldp'] = tar.GetLeaves(str(PurePosixPath(path_str, 'worldp')))  # Get a list of treelib nodes corresponding to the files in the TAR archive subfolder: "env_001\cam_down\worldp"
                    num_items = max(num_items, len(file_dict['worldp']))  # Update the file count
                except NodeIDAbsentError:
                    file_dict['worldp'] = None

                # Get objectID Data
                try:
                    file_dict['objectId'] = tar.GetLeaves(str(PurePosixPath(path_str, 'objectId')))  # Get a list of treelib nodes corresponding to the files in the TAR archive subfolder: "env_001\cam_down\objectId"
                    num_items = max(num_items, len(file_dict['objectId']))  # Update the file count
                except NodeIDAbsentError:
                    file_dict['objectId'] = None

                # Get JSON Data
                try:
                    file_dict['json'] = tar.GetLeaves(str(PurePosixPath(path_str, 'json')))  # Get a list of treelib nodes corresponding to the files in the TAR archive subfolder: "env_001\cam_down\json"
                    num_items = max(num_items, len(file_dict['json']))  # Update the file count
                except NodeIDAbsentError:
                    file_dict['json'] = None

                # Get Rot Data
                try:
                    file_dict['rot'] = tar.GetLeaves(str(PurePosixPath(path_str, 'rot')))  # Get a list of treelib nodes corresponding to the files in the TAR archive subfolder: "env_001\cam_down\rot"
                    num_items = max(num_items, len(file_dict['rot']))  # Update the file count
                except NodeIDAbsentError:
                    file_dict['rot'] = None

                # Output to Dictionary
                for ndx, _ in enumerate(range(0, num_items)):  # Iterate over the maximum file count.  Used to gather a one-to-one matrix of related file pointers for each dataset index.
                    self.sample_dict[index] = dict()  # Initialize dictionary representing the dataset item
                    self.sample_dict[index]['meta'] = dict()  # Initialize meta data
                    self.sample_dict[index]['index'] = index  # Store index value for reference and easy printing of diagnostics to the console. Easily tells what index the data belongs to.
                    self.sample_dict[index]['meta']['path'] = p_tarfile  # Store the path to the tar archive file.  This is a pathlib Path object.
                    #self.sample_dict[index]['meta']['tar'] = tar  # Store a reference to the tar class instantiation.  This does not copy the class, but merely attaches a low cost reference in memory to it.
                    if self.bool_depth:  # Only store information requested when instantiating the dataset.
                        if file_dict['depth']:
                            self.sample_dict[index]['meta']['depth'] = file_dict['depth'][ndx].data  # TarInfo object is attached to each treelib node using the data parameter
                        else:  # Check if the files returned are None
                            self.sample_dict[index]['meta']['depth'] = None  # Propagate None to the output dictionary of the dataset if no relevant files are found.
                    if self.bool_rgba:
                        if file_dict['rgba']:
                            self.sample_dict[index]['meta']['rgba'] = file_dict['rgba'][ndx].data  # TarInfo object is attached to each treelib node using the data parameter
                        else:  # Check if the files returned are None
                            self.sample_dict[index]['meta']['rgba'] = None  # Propagate None to the output dictionary of the dataset if no relevant files are found.
                    if self.bool_json:
                        if file_dict['json']:
                            self.sample_dict[index]['meta']['json'] = file_dict['json'][ndx].data  # TarInfo object is attached to each treelib node using the data parameter
                        else:  # Check if the files returned are None
                            self.sample_dict[index]['meta']['json'] = None  # Propagate None to the output dictionary of the dataset if no relevant files are found.
                    if self.bool_objectId:
                        if file_dict['objectId']:
                            self.sample_dict[index]['meta']['objectId'] = file_dict['objectId'][ndx].data  # TarInfo object is attached to each treelib node using the data parameter
                        else:  # Check if the files returned are None
                            self.sample_dict[index]['meta']['objectId'] = None  # Propagate None to the output dictionary of the dataset if no relevant files are found.
                    if self.bool_rot:
                        if file_dict['rot']:
                            self.sample_dict[index]['meta']['rot'] = file_dict['rot'][ndx].data  # TarInfo object is attached to each treelib node using the data parameter
                        else:  # Check if the files returned are None
                            self.sample_dict[index]['meta']['rot'] = None  # Propagate None to the output dictionary of the dataset if no relevant files are found.
                    if self.bool_worldp:
                        if file_dict['worldp']:
                            self.sample_dict[index]['meta']['worldp'] = file_dict['worldp'][ndx].data  # TarInfo object is attached to each treelib node using the data parameter
                        else:  # Check if the files returned are None
                            self.sample_dict[index]['meta']['worldp'] = None  # Propagate None to the output dictionary of the dataset if no relevant files are found.
                    index += 1  # Increment the PyTorch dataset index value stored

        else:
            index = 0  # the sample index number in the PyTorch dataset
            for _, p in enumerate(self.path_list):
                file_dict = dict()
                items_dict = dict()

                # Path Dictionary
                file_dict['dirpath'] = Path(p)

                # Get RGBA Data
                if self.bool_rgba:
                    items_dict.clear()
                    for ndx, f in enumerate(self.getFilesList(Path(p, 'rgba'))):
                        items_dict[ndx] = f
                        ndx += 1
                    file_dict['rgba'] = items_dict.copy()

                # Get Depth Data
                if self.bool_depth:
                    items_dict.clear()
                    for ndx, f in enumerate(self.getFilesList(Path(p, 'depth'))):
                        items_dict[ndx] = f
                        ndx += 1
                    file_dict['depth'] = items_dict.copy()

                # Get JSON Data
                if self.bool_json:
                    items_dict.clear()
                    for ndx, f in enumerate(self.getFilesList(Path(p, 'json'))):
                        items_dict[ndx] = f
                        ndx += 1
                    file_dict['json'] = items_dict.copy()

                # Get ObjectID Data
                if self.bool_objectId:
                    items_dict.clear()
                    for ndx, f in enumerate(self.getFilesList(Path(p, 'objectId'))):
                        items_dict[ndx] = f
                        ndx += 1
                    file_dict['objectId'] = items_dict.copy()

                # Get Rot Data
                if self.bool_rot:
                    items_dict.clear()
                    for ndx, f in enumerate(self.getFilesList(Path(p, 'rot'))):
                        items_dict[ndx] = f
                        ndx += 1
                    file_dict['rot'] = items_dict.copy()

                # Get World Point Data
                if self.bool_worldp:
                    items_dict.clear()
                    for ndx, f in enumerate(self.getFilesList(Path(p, 'worldp'))):
                        items_dict[ndx] = f
                        ndx += 1
                    file_dict['worldp'] = items_dict.copy()

                # Output to Dictionary
                for ndx, _ in enumerate(file_dict['rgba']):
                    self.sample_dict[index] = dict()
                    self.sample_dict[index]['meta'] = dict()
                    self.sample_dict[index]['index'] = index
                    self.sample_dict[index]['meta']['path'] = file_dict['dirpath']
                    if self.bool_depth:
                        if file_dict['depth']:
                            self.sample_dict[index]['meta']['depth'] = file_dict['depth'][ndx]
                        else:
                            self.sample_dict[index]['meta']['depth'] = None
                    if self.bool_rgba:
                        if file_dict['rgba']:
                            self.sample_dict[index]['meta']['rgba'] = file_dict['rgba'][ndx]
                        else:
                            self.sample_dict[index]['meta']['rgba'] = None
                    if self.bool_json:
                        if file_dict['json']:
                            self.sample_dict[index]['meta']['json'] = file_dict['json'][ndx]
                        else:
                            self.sample_dict[index]['meta']['json'] = None
                    if self.bool_objectId:
                        if file_dict['objectId']:
                            self.sample_dict[index]['meta']['objectId'] = file_dict['objectId'][ndx]
                        else:
                            self.sample_dict[index]['meta']['objectId'] = None
                    if self.bool_rot:
                        if file_dict['rot']:
                            self.sample_dict[index]['meta']['rot'] = file_dict['rot'][ndx]
                        else:
                            self.sample_dict[index]['meta']['rot'] = None
                    if self.bool_worldp:
                        if file_dict['worldp']:
                            self.sample_dict[index]['meta']['worldp'] = file_dict['worldp'][ndx]
                        else:
                            self.sample_dict[index]['meta']['worldp'] = None
                    index += 1

        return None

    def getDatasetDirectoriesList(self, ROOT):
        r"""
        Create a list of directories to the data folders.

        Parameters
        ----------
        ROOT : String
            Path to the root directory of the dataset.
            Example: r"A:\xR-EgoPose\data\Dataset\ValSet"

        Returns
        -------
        path_list : list
            When bool_tarArchive=False:
                A list of string-type paths are returned:
                    ['A:\\xR-EgoPose\\data\\Dataset\\ValSet\\male_008_a_a\\env_001\\cam_down',
                     'A:\\xR-EgoPose\\data\\Dataset\\ValSet\\male_008_a_a\\env_002\\cam_down',
                     'A:\\xR-EgoPose\\data\\Dataset\\ValSet\\male_008_a_a\\env_003\\cam_down']
            When bool_tarArchive=True:
                A list of dictionaries is returned:
                    [{'path_tarfile' : 'A:\\xR-EgoPose\\data\\Dataset\\ValSet\\male_008_a_a.tar',
                     'path' : 'env_002\\cam_down'}, ...]
        """
        if self.bool_tarArchive:
            tarfiles_list = self.getFilesList(ROOT, '.tar')  # Get List of .tar files in ROOT

            # Get Internal Tar File Paths
            path_list = list()
            for tarfile_path in tarfiles_list:  # Iterate over each string list item of tar-archive file paths comprising the dataset
                tar = Tar(tarfile_path)  # Instantiate Tar.py Class Object
                for path in tar.listPaths(level=2):  # Returns a list of greater tar-archive subdirectory paths, limited to level 2 only.
                    entry = dict()
                    entry['path_tarfile'] = tarfile_path  # String path to the tar-archive
                    entry['path'] = path  # String path to a level 2 subdirectory within the TAR archive.
                    path_list.append(entry.copy())
            return path_list  # List of Dictionaries
        else:
            path_list = list()
            subdirs_dict = self.getImmediateSubDirectoriesDict(ROOT)  # Returns directory list: "male_008_a_a"
            for _, sub2dirname_p in subdirs_dict.items():
                sub2dirs_dict = self.getImmediateSubDirectoriesDict(str(sub2dirname_p))  # Returns directory list: "env_001"
                for _, sub3dirname_p in sub2dirs_dict.items():
                    sub3dirs_dict = self.getImmediateSubDirectoriesDict(str(sub3dirname_p))  # Returns directory list: "cam_down"
                    for _, item_p in sub3dirs_dict.items():
                        path_list.append(str(item_p))  # Adds to path list: "male_008_a_a/env_001/cam_down"
            return path_list  # Return List of String Paths

    def getImmediateSubDirectoriesDict(self, PATH):
        """
        Gets a list of immediate subdirectories within a given directory path.
        Only called when parsing a non-archieve dataset when
        bool_tarArchive=False (Denis Tome implementation)

        Parameters
        ----------
        PATH : string
            Path to a directory in the filesystem.

        Returns
        -------
        path_dict : python dictionary
            A key value pair correspond to shortened directory names (keys) and
            full directory paths (values).

        """
        path_dict = dict()
        p = Path(PATH)  # The root directory where immediate subdirectories are found

        gen = p.iterdir()  # Generator object used to iterate over the file/directory contents
        dirs = list()  # Perform a first pass to eliminate corrected entries
        while True:
            try:
                value = next(gen)  # Attempt to get a value from the iterator
                dirs.append(value)  # Append to the first pass list
            except StopIteration:  # Enumerable generator has run out of items to return
                break  # Stop iterating over the directories in the list.  Nothing left to iterate over.
            except FileNotFoundError:  # Sometimes a directory is corrupted
                continue  # Skip

        for x in sorted(dirs):  # Perform second pass over the directory list, sorted by name in ascending order
            if x.is_dir():  # Only process directories
                _, dirname_str = os.path.split(x)  # Get shortened directory name (last portion of the full path)
                path_dict[dirname_str] = x  # Create Key/Value pair.  Key is the shorted directory name.  Value is the full directory path.
        return path_dict

    def getFilesList(self, PATH, ext=None):
        """
        Returns a list of files in a directory.

        Parameters
        ----------
        PATH : string
            Directory path to search.
        ext : string, optional
            Limit search to only the provided file extentions.  Other file
            extensions are ignored.  If no extensions are specified, then all
            file extensions are returned.  The default is None.

        Returns
        -------
        path_list : python list
            A list of file paths.

        """
        path_list = list()
        p = Path(PATH)  # The root directory where files are found

        gen = p.iterdir()  # Generator object used to iterate over the file/directory contents
        files = list()  # Perform a first pass to eliminate corrected entries
        while True:
            try:
                value = next(gen)  # Attempt to get a value from the iterator
                files.append(value)  # Append to the first pass list
            except StopIteration:  # Enumerable generator has run out of items to return
                break  # Stop iterating over the directories in the list.  Nothing left to iterate over.
            except FileNotFoundError:  # Sometimes files are corrupted
                files.append(None)  # List corrupted files as None
                continue  # Skip

        for x in files:  # Do not sort, process in order returned by OS
            if x is None:  # File is corrupt
                path_list.append(None)  # List corrupted files in order
            elif x.is_file():  # File is not corrupt (passes checks)
                _, dirname_str = os.path.split(x)  # Get shortened file name (last portion of the full path)
                if ext is not None:  # Search by file extension if provided
                    if x.suffix == ext:  # Exclude non-specified file extensions
                        path_list.append(x)  # Add to output list
                else:
                    path_list.append(x)  # Add to output list

        if len(path_list) == 1:  # Additional check to clarify single, empty entries -> set to empty (ensure count is zero, not one in an empty list)
            if path_list[0] is None:  # Check if entry is none
                path_list.clear()  # If no files are found, return empty

        return path_list

    def printDatasetDirectoriesList(self):
        r"""
        A diagnostic function that outputs the contents of a directory list.

        Example output:
            A:\xR-EgoPose\data\Dataset\ValSet\male_008_a_a\env_001\cam_down
            A:\xR-EgoPose\data\Dataset\ValSet\male_008_a_a\env_002\cam_down
            A:\xR-EgoPose\data\Dataset\ValSet\male_008_a_a\env_003\cam_down

        Returns
        -------
        None.

        """
        for _, x in enumerate(self.path_list):
            print(x)

    def printDatasetSampleDictRange(self, start_ndx, end_ndx):
        """
        Prints the contents of the sample dictionary.

        Parameters
        ----------
        start_ndx : integer
            Start index of the range to be printed to the console.
        end_ndx : integer
            End index of the range to be printed to the console.

        Returns
        -------
        None.

        """
        for _, i in enumerate(range(start_ndx, end_ndx + 1)):
            print(str(i) + ": ")
            print(self.sample_dict[i])
        return None

    def ToString(self):
        """
        A diagnostic function used to display a set of tests of functionality
        of this dataset object instantiation.

        Returns
        -------
        None.

        """
        print("xR-EgoPose Dataset Implementation by Andrew Huard")
        print("\nTesting Item 0 of " + str(len(self)))
        item = self[0]

        print("\nIndex: " + str(item['index']))
        print("\n Meta: ")
        print(item['meta'])
        print("\n Data: ")
        print(item['data'])

        print("\nTesting Items 13 through 15:")
        self.printDatasetSampleDictRange(13, 15)
        self.printDatasetSampleDictRange(5013, 5015)
        self.printDatasetSampleDictRange(10013, 10015)

        print("\nDataset Directories:")
        self.printDatasetDirectoriesList()


def main():
    """
    Main method used for testing the xrEgoPose class. This function is used
    for testing purposes and to provide very basic examples of how to access
    data within the xR-EgoPose dataset.  This function is completely unused by
    this dataset and PyTorch dataloaders.

    Use this function to test the functionality of an instantiation of this
    dataset.

    Returns
    -------
    None.

    """
    dataset = xrEgoPose(bool_train=False, bool_tensor=False,
                        str_rootdir=r'A:\xR-EgoPose\data\Dataset\ValSet',
                        bool_json=True, bool_rot=True, bool_rgba=True,
                        bool_worldp=True, bool_depth=True, bool_objectId=True,
                        bool_tarArchive=True)

    #dataset.printDatasetSampleDictRange(13, 15)

    #dataset[0]['data']['depth'].show()
    #dataset[5201]['data']['depth'].show()
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

    dataset.ToString()
    


if __name__ == "__main__":
    """
    Main Function executes if python executes directly on this class file.
    This function is intended for diagnostic purposes only.  This block is
    completely unused by PyTorch DataLoader.
    """
    main()
