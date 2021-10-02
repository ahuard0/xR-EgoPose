# xR-EgoPose
 An implementation of the xR-EgoPose dataset except the dataset is archived in tar files to reduce the number of files stored.  This is an advantage on some cluster based High Performance Computing (HPC) platforms, which charge per file stored.
 
 For the original dataset, see: https://github.com/facebookresearch/xR-EgoPose (Denis Tome, 2019)
 
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
