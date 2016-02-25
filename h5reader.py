"""
allows access to hdf5 files
"""

import sys
import h5py

class H5Reader(object):
    """object representation of an hdf5file, has very high level commands
    """
    def __init__(self, file_path):
        """opens the hdf5 file at file_path
        """
        self.h5file = h5py.File(file_path, 'r')

    def list(self):
        """outputs a list of all groups at the first level of the directory tree
        """
        labels = []
        for group in self.h5file:
            labels.append(str(group))
        return labels

    def get_dset(self, label):
        """returns all dset in a group as a concatenated list
        """
        group = self.h5file[label]
        concat = []
        for dset in group:
            data = group[dset][:].tolist()
            concat.append(data)
        return concat
        """
        group = self.h5file[label]
        concat = []
        for dset in group:
            data = group[dset][:].tolist()
            concat += data
        return concat
        """

if __name__ == '__main__':
    INPUT_PATH = sys.argv[1]
    H5 = H5Reader(INPUT_PATH)
    print H5.get_dset(H5.list()[0])
