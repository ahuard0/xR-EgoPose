# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:36:30 2021

Motion History Image (MHI)

@author: Andrew Huard
"""
from torch.utils.data import Dataset
from dataset.xrEgoPose.xrEgoPose import xrEgoPose
import numpy as np
from pathlib import Path
import pickle

class MHI(Dataset):
    r"""
    Implementation of Motion History Image from the paper, "Egocentric Pose
    Estimation from Human Vision Span" by Hao Jiang and Vamsi Krishna Ithapu,
    12 April 2021.  See arXiv:2104.05167v1.
    https://arxiv.org/abs/2104.05167
    """
    
    def __init__(self, dataset, n_seq=13):
        """
        Initialization function for the Motion History Image (MHI) dataset.
        The MHI dataset accepts an xR-EgoPose dataset class object to form 
        sequential images including rotation and translation data.

        Parameters
        ----------
        dataset : xrEgoPose class object
            The xR-EgoPose datasets available are: TestSet, TrainSet, ValSet.
            See the xrEgoPose class for more information.

        n_seq : integer, optional
            The number of frames in a sequence forming an MHI block.  Each
            MHI block returned by the dataset is size (13,n_seq).

        Returns
        -------
        None.

        """
        self.dataset = dataset
        self.n_seq = n_seq
        self.blocks = dict()
        
        index = 0  # MHI index
        for setID in np.arange(min(dataset.set_index),max(dataset.set_index)+1):
            set_info = dataset.set_dict[setID]
            index_set_start = set_info['index_start']
            index_set_end = set_info['index_end']
            set_length = index_set_end - index_set_start
            num_blocks = np.floor(set_length/n_seq)
            
            # Form Blocks (13 x n_seq)
            i = index_set_start  # xR-EgoPose dataset index
            for b in np.arange(0, num_blocks):
                block = np.zeros(shape=(13, n_seq), dtype=float)
                for c in np.arange(0, n_seq):
                    camera = dataset[i]['data']['camera']
                    rot = camera['rot_matrix'][0:3,0:3]
                    block[0:9, c] = np.reshape(rot, (9))
                    block[9:12, c] = camera['translation']
                    block[12, c] = camera['char_height']
                    i += 1
                self.blocks[index] = block.copy()
                index += 1
        
    
    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, ndx):
        return self.blocks[ndx]

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
            pickle.dump((self.sample_dict, self.path_list, self.set_dict, self.set_index), file)

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
            self.sample_dict, self.path_list, self.set_dict, self.set_index = pickle.load(file)


if __name__ == "__main__":
    # try:
    #     dataset_xregopose_val
    # except NameError:
    #     dataset_xregopose_val = xrEgoPose(bool_train=False, bool_tensor=True,
    #                         str_rootdir=r'A:\xR-EgoPose\data\Dataset\ValSet',
    #                         bool_json=False, bool_rot=False, bool_rgba=False,
    #                         bool_worldp=False, bool_depth=False,
    #                         bool_objectId=False, bool_tarArchive=True,
    #                         bool_camera=True)
    
    try:
        dataset_xregopose_train
    except NameError:
        dataset_xregopose_train = xrEgoPose(bool_train=True, bool_tensor=True,
                            str_rootdir=r'A:\xR-EgoPose\data\Dataset\TrainSet',
                            bool_json=False, bool_rot=False, bool_rgba=False,
                            bool_worldp=False, bool_depth=False,
                            bool_objectId=False, bool_tarArchive=True,
                            bool_camera=True)
        
    # dataset_mhi_val = MHI(dataset=dataset_xregopose_val, n_seq=13)
    dataset_mhi_train = MHI(dataset=dataset_xregopose_train, n_seq=13)
    
    # print("Val: " + str(len(dataset_mhi_val)))
    print("Train: " + str(len(dataset_mhi_train)))
