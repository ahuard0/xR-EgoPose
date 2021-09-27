# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:36:30 2021

Motion History Image (MHI)

@author: Andrew Huard
"""
from torch.utils.data import Dataset
from dataset.xrEgoPose.xrEgoPose import xrEgoPose

class MHI(Dataset):
    r"""
    Implementation of Motion History Image from the paper, "Egocentric Pose
    Estimation from Human Vision Span" by Hao Jiang and Vamsi Krishna Ithapu,
    12 April 2021.  See arXiv:2104.05167v1.
    https://arxiv.org/abs/2104.05167
    """
    
    def __init__(self, dataset, n_seq):
        """
        Initialization function for the Motion History Image (MHI) dataset.
        The MHI dataset accepts an xR-EgoPose dataset class object to form 
        sequential images including rotation and translation data.

        Parameters
        ----------
        dataset : xrEgoPose class object
            The xR-EgoPose datasets available are: TestSet, TrainSet, ValSet.
            See the xrEgoPose class for more information.

        Returns
        -------
        None.

        """
        self.dataset = dataset
        self.n_seq = n_seq
        
    
    def __len__(self):
        return 0
    
    def __getitem__(self, ndx):
        return None


def main():
    dataset_xregopose_val = xrEgoPose(bool_train=False, bool_tensor=True,
                        str_rootdir=r'A:\xR-EgoPose\data\Dataset\ValSet',
                        bool_json=False, bool_rot=False, bool_rgba=False,
                        bool_worldp=False, bool_depth=False, bool_objectId=False,
                        bool_tarArchive=True, bool_camera=True)
    dataset_mhi_val = MHI(dataset=dataset_xregopose_val, n_seq=13)
    print(len(dataset_mhi_val))


if __name__ == "__main__":
    main()
