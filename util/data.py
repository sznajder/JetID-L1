# Handles the data importing and small preprocessing for the interaction network.

import os
import numpy as np
import tensorflow as tf
import functools

#from .terminal_colors import tcols


class Data:
    """Data class to store the data to be used in learning for the interaction network.

    Attributes:
        fpath: Path to where the data files are located.
        fname: The base name of the data you want to load.
        fnames_train: The file names of the training data sets in case of kfold.
        fnames_test: The file name of the testing kfold.
    """

    def __init__(
        self,
        fpath: str,
        fname: str = None,
        fnames_train: list = None,
        fname_test: str = None,
        only_test: bool = False
    ):
        self._fpath = fpath
        self._fname = fname
        self._fnames_train = fnames_train
        self._fname_test = fname_test
        self._only_test = only_test

        self.train_data = self._load_data("train")
        self.train_target = self._load_target("train")
        self.ntrain_jets = self.train_data.shape[0]

        self.test_data = self._load_data("test")
        self.test_target = self._load_target("test")
        self.ntest_jets = self.test_data.shape[0]

        self.ncons = self.train_data.shape[1]
        if self.ncons == 0:
            self.ncons = self.test_data.shape[1]

        self.nfeat = self.train_data.shape[2]
        if self.nfeat == 0:
            self.nfeat = self.test_data.shape[2]

        self._success_message()

        # WIP tensorflow record implementation.
        # self.feats = fname.split("_")[4]
        # self.buffer_size = buffer_size
        # self.batch_size = batch_size
        # self.jet_seed = jet_seed
        # self.seed = seed

    @classmethod
    def load_kfolds(cls, fpath: str, fnames_train: list, fname_test: str):
        """Alternative constructor for kfolded data."""
        return cls(fpath, fnames_train=fnames_train, fname_test=fname_test)

    def _load_train_data_kfolds(self) -> np.ndarray:
        """Loads data from multiple data files."""
        datafile_name = "x_" + self._fnames_train[0] + ".npy"
        datafile_path = os.path.join(self._fpath, datafile_name)
        data = np.load(datafile_path)
        for fname in self._fnames_train[1:]:
            datafile_name = "x_" + fname + ".npy"
            datafile_path = os.path.join(self._fpath, datafile_name)
            data_kfold = np.load(datafile_path)
            data = np.concatenate((data, data_kfold), axis=0)

        return data

    def _load_train_target_kfolds(self) -> np.ndarray:
        """Loads the target corrsponding from multiple target files."""
        datafile_name = "y_" + self._fnames_train[0] + ".npy"
        datafile_path = os.path.join(self._fpath, datafile_name)
        target = np.load(datafile_path)
        for fname in self._fnames_train[1:]:
            targetfile_name = "y_" + fname + ".npy"
            targetfile_path = os.path.join(self._fpath, targetfile_name)
            target_kfold = np.load(targetfile_path)
            target = np.concatenate((target, target_kfold), axis=0)

        return target

    def _load_test_data_kfold(self) -> np.ndarray:
        """Loads the one kfold data reserved for testing."""
        datafile_name = "x_" + self._fname_test + ".npy"
        datafile_path = os.path.join(self._fpath, datafile_name)

        return np.load(datafile_path)

    def _load_test_target_kfold(self) -> np.ndarray:
        """Loads the one kfold target reserved for testing."""
        targetfile_name = "y_" + self._fname_test + ".npy"
        targetfile_path = os.path.join(self._fpath, targetfile_name)

        return np.load(targetfile_path)

    def _load_data(self, data_type: str) -> np.ndarray:
        """Load data from the data files generated by the pre-processing scripts."""
        if data_type == "train" and self._fnames_train:
            return self._load_train_data_kfolds()
        if data_type == "test" and self._fname_test:
            return self._load_test_data_kfold()
        if self._fname is None:
            raise ValueError("No filename for data provided!")

        if data_type == 'train' and self._only_test:
            return np.empty([0, 0, 0])

        datafile_name = "x_" + self._fname + "_" + data_type + ".npy"
        datafile_path = os.path.join(self._fpath, datafile_name)

        return np.load(datafile_path, mmap_mode="r+")

        # WIP tensorflow record implementation.
        # data = tf.data.TFRecordDataset(datafile_path)
        # data = self._ignore_order(data)
        # data = data.map(self._read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        # data.shuffle(buffer_size=self.buffer_size, seed=self.jet_seed)
        # data.prefetch(buffer_size=tf.data.AUTOTUNE)
        # data.batch(self.batch_size)

    def _load_target(self, data_type: str) -> np.ndarray:
        """Load data from the data files generated by the pre-processing scripts."""
        if data_type == "train" and self._fnames_train:
            return self._load_train_target_kfolds()
        if data_type == "test" and self._fname_test:
            return self._load_test_target_kfold()
        if self._fname is None:
            raise ValueError("No filename for data provided!")

        if data_type == 'train' and self._only_test:
            return np.empty([0, 0, 0])

        datafile_name = "y_" + self._fname + "_" + data_type + ".npy"
        datafile_path = os.path.join(self._fpath, datafile_name)

        return np.load(datafile_path, mmap_mode="r+")

        return np.load(datafile_path, mmap_mode="r+")

        # WIP tensorflow record implementation.
        # data = tf.data.TFRecordDataset(datafile_path)
        # data = self._ignore_order(data)
        # data = data.map(self._read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        # data.shuffle(buffer_size=self.buffer_size, seed=self.jet_seed)
        # data.prefetch(buffer_size=tf.data.AUTOTUNE)
        # data.batch(self.batch_size)

    def _success_message(self):
        # Display success message for loading data when called.
        print("\n----------------")
        #print(tcols.OKGREEN + "Data loading complete:" + tcols.ENDC)
        print("Data loading complete:")
        print(f"File name: {self._fname}")
        print(f"Training data size: {self.ntrain_jets:,}")
        print(f"Test data size: {self.ntest_jets:,}")
        print(f"Number of constituents: {self.ncons:,}")
        print(f"Number of features: {self.nfeat:,}")
        print("----------------\n")

    # WIP tensorflow record implementation.
    # def _ignore_order(self, data: tf.data.TFRecordDataset):
    #     ignore_order = tf.data.Options()
    #     ignore_order.experimental_deterministic = False
    #     data = data.with_options(ignore_order)

    #     return data

    # def _read_tfrecord(self, example):
    #     tf_record_format = self._get_record_format(self.feats)
    #     example = tf.io.parse_single_example(example, tf_record_format)
    #     x_data = example["px"]
    #     y_data = example["label"]
    #     return x_data, y_data

    # def _get_record_format(self, feats: str):
    #     switcher = {
    #         "andre": lambda: self._andre_format(),
    #         "jedinet": lambda: self._jedinet_format(),
    #     }

    #     tfrecord_format = switcher.get(feats, lambda: None)()
    #     if tfrecord_format is None:
    #         raise TypeError("Feature selection name not valid!")

    #     return tfrecord_format

    # def _andre_format(self):
    #     return {
    #         "pt": tf.io.FixedLenFeature([], tf.float32),
    #         "eta": tf.io.FixedLenFeature([], tf.float32),
    #         "phi": tf.io.FixedLenFeature([], tf.float32),
    #         "label": tf.io.FixedLenFeature([], tf.float32),
    #     }

    # def _jedinet_format(self):
    #     return {
    #         "px": tf.io.FixedLenFeature([], tf.float32),
    #         "py": tf.io.FixedLenFeature([], tf.float32),
    #         "pz": tf.io.FixedLenFeature([], tf.float32),
    #         "E": tf.io.FixedLenFeature([], tf.float32),
    #         "Erel": tf.io.FixedLenFeature([], tf.float32),
    #         "pt": tf.io.FixedLenFeature([], tf.float32),
    #         "ptrel": tf.io.FixedLenFeature([], tf.float32),
    #         "eta": tf.io.FixedLenFeature([], tf.float32),
    #         "etarel": tf.io.FixedLenFeature([], tf.float32),
    #         "etarot": tf.io.FixedLenFeature([], tf.float32),
    #         "phi": tf.io.FixedLenFeature([], tf.float32),
    #         "phirel": tf.io.FixedLenFeature([], tf.float32),
    #         "phirot": tf.io.FixedLenFeature([], tf.float32),
    #         "deltaR": tf.io.FixedLenFeature([], tf.float32),
    #         "cos(theta)": tf.io.FixedLenFeature([], tf.float32),
    #         "cos(thetarel)": tf.io.FixedLenFeature([], tf.float32),
    #         "label": tf.io.FixedLenFeature([], tf.float32),
    #     }
