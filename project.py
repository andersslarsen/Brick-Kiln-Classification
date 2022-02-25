# THIS CODE ASSUMES THAT THE DATASET
# BRICK-KILN IS ALREADY DOWNLOADED AND
# UNPACKED/ZIPPED
# PLEASE SPECIFY IN LINE 651 THE INPUT-PATH TO THE DATASET

import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, auc, precision_recall_curve
import seaborn as sns
import numpy as np
from pathlib import Path
import h5py
import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from PIL import Image

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
metrics_model = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
]

class SustainBenchDataset:
    """
    Shared dataset class for all SustainBench datasets.
    Each data point in the dataset is an (x, y, metadata) tuple, where:
    - x is the input features
    - y is the target
    - metadata is a vector of relevant information, e.g., domain.
      For convenience, metadata also contains y.
    """
    DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
    DEFAULT_SPLIT_NAMES = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

    def __init__(self, root_dir, download, split_scheme):
        if len(self._metadata_array.shape) == 1:
            self._metadata_array = self._metadata_array.unsqueeze(1)
        self.check_init()

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the SustainBenchSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        if isinstance(self.y_array[0], Path):  # edited to fit output which are also images
            # dataset has images as an output
            y = self.get_output_image(self.y_array[idx])  # a list of image path name
        else:
            y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y, metadata

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        raise NotImplementedError

    def eval(self, y_pred, y_true, metadata):
        """
        Args:
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        raise NotImplementedError

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (SustainBenchSubset): A (potentially subsampled) subset of the SustainBenchDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = SustainBenchSubset(self, split_idx, transform)
        return subset

    def check_init(self):
        """
        Convenience function to check that the SustainBenchDataset is properly configured.
        """
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'SustainBenchDataset is missing {attr_name}.'

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Check splits
        assert self.split_dict.keys()==self.split_names.keys()
        assert 'train' in self.split_dict
        assert 'val' in self.split_dict

        ## Check that required arrays are Tensors # edited
        # assert isinstance(self.y_array, torch.Tensor), 'y_array must be a torch.Tensor'
        # assert isinstance(self.metadata_array, torch.Tensor), 'metadata_array must be a torch.Tensor'

        # Check that dimensions match
        assert len(self.y_array) == len(self.metadata_array)
        assert len(self.split_array) == len(self.metadata_array)

        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]
        # For convenience, include y in metadata_fields if y_size == 1
        if self.y_size == 1:
            assert 'y' in self.metadata_fields

    @property
    def latest_version(cls):
        def is_later(u, v):
            """Returns true if u is a later version than v."""
            u_major, u_minor = tuple(map(int, u.split('.')))
            v_major, v_minor = tuple(map(int, v.split('.')))
            if (u_major > v_major) or (
                (u_major == v_major) and (u_minor > v_minor)):
                return True
            else:
                return False

        latest_version = '0.0'
        for key in cls.versions_dict.keys():
            if is_later(key, latest_version):
                latest_version = key
        return latest_version

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset, e.g., 'amazon', 'camelyon17'.
        """
        return self._dataset_name

    @property
    def version(self):
        """
        A string that identifies the dataset version, e.g., '1.0'.
        """
        if self._version is None:
            return self.latest_version
        else:
            return self._version

    @property
    def versions_dict(self):
        """
        A dictionary where each key is a version string (e.g., '1.0')
        and each value is a dictionary containing the 'download_url' and
        'compressed_size' keys.
        'download_url' is the URL for downloading the dataset archive.
        If None, the dataset cannot be downloaded automatically
        (e.g., because it first requires accepting a usage agreement).
        'compressed_size' is the approximate size of the compressed dataset in bytes.
        """
        return self._versions_dict

    @property
    def data_dir(self):
        """
        The full path to the folder in which the dataset is stored.
        """
        return self._data_dir

    @property
    def collate(self):
        """
        Torch function to collate items in a batch.
        By default returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)

    @property
    def split_scheme(self):
        """
        A string identifier of how the split is constructed,
        e.g., 'standard', 'in-dist', 'user', etc.
        """
        return self._split_scheme

    @property
    def split_dict(self):
        """
        A dictionary mapping splits to integer identifiers (used in split_array),
        e.g., {'train': 0, 'val': 1, 'test': 2}.
        Keys should match up with split_names.
        """
        return getattr(self, '_split_dict', SustainBenchDataset.DEFAULT_SPLITS)

    @property
    def split_names(self):
        """
        A dictionary mapping splits to their pretty names,
        e.g., {'train': 'Train', 'val': 'Validation', 'test': 'Test'}.
        Keys should match up with split_dict.
        """
        return getattr(self, '_split_names', SustainBenchDataset.DEFAULT_SPLIT_NAMES)

    @property
    def split_array(self):
        """
        An array of integers, with split_array[i] representing what split the i-th data point
        belongs to.
        """
        return self._split_array

    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def y_size(self):
        """
        The number of dimensions/elements in the target, i.e., len(y_array[i]).
        For standard classification/regression tasks, y_size = 1.
        For multi-task or structured prediction settings, y_size > 1.
        Used for logging and to configure models to produce appropriately-sized output.
        """
        return self._y_size

    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)

    @property
    def is_classification(self):
        """
        Boolean. True if the task is classification, and false otherwise.
        Used for logging purposes.
        """
        return (self.n_classes is not None)

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array

    @property
    def metadata_map(self):
        """
        An optional dictionary that, for each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        For example, if we have
            metadata_fields = ['hospital', 'y']
            metadata_map = {'hospital': ['East', 'West']}
        then if metadata_array[i, 0] == 0, the i-th data point belongs to the 'East' hospital
        while if metadata_array[i, 0] == 1, it belongs to the 'West' hospital.
        """
        return getattr(self, '_metadata_map', None)

    @property
    def original_resolution(self):
        """
        Original image resolution for image datasets.
        """
        return getattr(self, '_original_resolution', None)

    def initialize_data_dir(self, root_dir, download):
        """
        Helper function for downloading/updating the dataset if required.
        Note that we only do a version check for datasets where the download_url is set.
        Currently, this includes all datasets except Yelp.
        Datasets for which we don't control the download, like Yelp,
        might not handle versions similarly.
        """
        if self.version not in self.versions_dict:
            raise ValueError(f'Version {self.version} not supported. Must be in {self.versions_dict.keys()}.')

        download_url = self.versions_dict[self.version]['download_url']
        compressed_size = self.versions_dict[self.version]['compressed_size']

        os.makedirs(root_dir, exist_ok=True)

        data_dir = os.path.join(root_dir, f'{self.dataset_name}_v{self.version}')
        version_file = os.path.join(data_dir, f'RELEASE_v{self.version}.txt')
        current_major_version, current_minor_version = tuple(map(int, self.version.split('.')))

        # Check if we specified the latest version. Otherwise, print a warning.
        latest_major_version, latest_minor_version = tuple(map(int, self.latest_version.split('.')))
        if latest_major_version > current_major_version:
            print(
                f'*****************************\n'
                f'{self.dataset_name} has been updated to version {self.latest_version}.\n'
                f'You are currently using version {self.version}.\n'
                f'We highly recommend updating the dataset by not specifying the older version in the command-line argument or dataset constructor.\n'
                f'See https://wilds.stanford.edu/changelog for changes.\n'
                f'*****************************\n')
        elif latest_minor_version > current_minor_version:
            print(
                f'*****************************\n'
                f'{self.dataset_name} has been updated to version {self.latest_version}.\n'
                f'You are currently using version {self.version}.\n'
                f'Please consider updating the dataset.\n'
                f'See https://wilds.stanford.edu/changelog for changes.\n'
                f'*****************************\n')

        # If the data_dir exists and contains the right RELEASE file,
        # we assume the dataset is correctly set up
        if os.path.exists(data_dir) and os.path.exists(version_file):
            return data_dir

        # If the data_dir exists and does not contain the right RELEASE file, but it is not empty and the download_url is not set,
        # we assume the dataset is correctly set up
        if ((os.path.exists(data_dir)) and
            (len(os.listdir(data_dir)) > 0) and
            (download_url is None)):
            return data_dir

        # Otherwise, we assume the dataset needs to be downloaded.
        # If download == False, then return an error.
        if download == False:
            if download_url is None:
                raise FileNotFoundError(f'The {self.dataset_name} dataset could not be found in {data_dir}. {self.dataset_name} cannot be automatically downloaded. Please download it manually.')
            else:
                raise FileNotFoundError(f'The {self.dataset_name} dataset could not be found in {data_dir}. Initialize the dataset with download=True to download the dataset. If you are using the example script, run with --download. This might take some time for large datasets.')

        # Otherwise, proceed with downloading.
        if download_url is None:
            raise ValueError(f'Sorry, {self.dataset_name} cannot be automatically downloaded. Please download it manually.')

        from sustainbench.datasets.download_utils import download_and_extract_archive
        print(f'Downloading dataset to {data_dir}...')
        print(f'You can also download the dataset manually at https://wilds.stanford.edu/downloads.')
        try:
            start_time = time.time()
            download_and_extract_archive(
                url=download_url,
                download_root=data_dir,
                filename='archive.tar.gz',
                remove_finished=True,
                size=compressed_size)

            download_time_in_minutes = (time.time() - start_time) / 60
            print(f"It took {round(download_time_in_minutes, 2)} minutes to download and uncompress the dataset.")
        except Exception as e:
            print(f"\n{os.path.join(data_dir, 'archive.tar.gz')} may be corrupted. Please try deleting it and rerunning this command.\n")
            print(f"Exception: ", e)

        return data_dir

    @staticmethod
    def standard_eval(metric, y_pred, y_true):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results = {
            **metric.compute(y_pred, y_true),
        }
        results_str = (
            f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        )
        return results, results_str

    @staticmethod
    def standard_group_eval(metric, grouper, y_pred, y_true, metadata, aggregate=True):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - grouper (CombinatorialGrouper): Grouper object that converts metadata into groups
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results, results_str = {}, ''
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        g = grouper.metadata_to_group(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups)
        for group_idx in range(grouper.n_groups):
            group_str = grouper.group_field_str(group_idx)
            group_metric = group_results[metric.group_metric_field(group_idx)]
            group_counts = group_results[metric.group_count_field(group_idx)]
            results[f'{metric.name}_{group_str}'] = group_metric
            results[f'count_{group_str}'] = group_counts
            if group_results[metric.group_count_field(group_idx)] == 0:
                continue
            results_str += (
                f'  {grouper.group_str(group_idx)}  '
                f"[n = {group_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {group_results[metric.group_metric_field(group_idx)]:5.3f}\n")
        results[f'{metric.worst_group_metric_field}'] = group_results[f'{metric.worst_group_metric_field}']
        results_str += f"Worst-group {metric.name}: {group_results[metric.worst_group_metric_field]:.3f}\n"
        return results, results_str


class SustainBenchSubset(SustainBenchDataset):
    def __init__(self, dataset, indices, transform):
        """
        This acts like torch.utils.data.Subset, but on SustainBenchDatasets.
        We pass in transform explicitly because it can potentially vary at
        training vs. test time, if we're using data augmentation.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_scheme', '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, metadata

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)


#from sustainbench.datasets.sustainbench_dataset import SustainBenchDataset
#from sustainbench.common.grouper import CombinatorialGrouper
#from sustainbench.common.utils import subsample_idxs, shuffle_arr

from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score

class BrickKilnDataset(SustainBenchDataset):
    """
    Supported `split_scheme`: 'official'
    Input (x):
        64 x 64 x 13 imagery from Sentinel-2. Images are not normalized.
    Output (y):
        y is a binary label representing containing or not containing a brick kiln
    Metadata:
        Metadata contains image lat and long bounds, as well as indices from the original
        tif file the image is from.
    Website: TODO
    Original publication: TODO
    License:
        S2 data is U.S. Public Domain.
    """
    _dataset_name = 'brick_kiln'
    _versions_dict = { # TODO
        '1.0': {
            'download_url': None,
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme not in ['official']:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        self.metadata = pd.read_csv(os.path.join(self.data_dir, 'list_eval_partition.csv'))
        self._split_array = self.metadata['partition'].values

        self._y_array = torch.from_numpy(self.metadata['y'].values)
        self._y_size = 1

        self._metadata_fields = ['y', 'hdf5_file', 'hdf5_idx', 'lon_top_left', 'lat_top_left', 'lon_bottom_right', 'lat_bottom_right', 'indice_x', 'indice_y']
        self._metadata_array = torch.tensor(self.metadata[self.metadata_fields].astype(float).values)

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        hdf5_loc = self.metadata['hdf5_file'].iloc[idx]
        with h5py.File(os.path.join(self.data_dir, f'examples_{hdf5_loc}.hdf5'), 'r') as f:
            img = f['images'][self.metadata['hdf5_idx'].iloc[idx]]

        img = torch.from_numpy(img).float()
        return img


    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - prediction_fn (function): A function that turns y_pred into predicted labels. If none, y_pred is
              expected to be probability score
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        if prediction_fn is None:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)

            results = {'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
            results_str = f'Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}'
        else:
            precision = precision_score(y_true, prediction_fn(y_pred))
            recall = recall_score(y_true, prediction_fn(y_pred))
            accuracy = accuracy_score(y_true, prediction_fn(y_pred))
            auc = roc_auc_score(y_true, y_pred)

            results = {'Precision': precision, 'Recall': recall, 'Accuracy': accuracy, 'AUC': auc}
            results_str = f'Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, AUC: {auc}'

        return results, results_str

def normalizeArr(list):
    arr = np.array(list)
    arr_min = arr.min(axis=(1, 2), keepdims=True)
    arr_max = arr.max(axis=(1, 2), keepdims=True)
    arr = (arr - arr_min)/(arr_max-arr_min)
    return arr

def createCnn():
    img_height = 64
    img_width = 64
    color_channels = 3

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, color_channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))

    return model

def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.pdf")
    plt.show()

    print('non-brickkilns correctly Detected (True Negatives): ', cm[0][0])
    print('non-brickkilns Incorrectly Detected (False Positives): ', cm[0][1])
    print('Total non-brickkilns: ', np.sum(cm[0]))

    print('brickkilns Missed (False Negatives): ', cm[1][0])
    print('brickkilns Detected (True Positives): ', cm[1][1])
    print('Total brickkilns: ', np.sum(cm[1]))

def plot_metrics(history):
    metrics = ['loss', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()
        plt.savefig("loss_prec_rec.pdf")
        plt.show()

def main():
    data = BrickKilnDataset(root_dir="../input/brick-kiln/")

    train = data.get_subset('train', 0.01)
    y_train = train.y_array.numpy()
    val = data.get_subset('val', 0.01)
    y_val = val.y_array.numpy()
    test = data.get_subset('test', 0.01)
    y_test = test.y_array.numpy()

    x_train = []
    x_val = []
    x_test = []

    for i in train.indices:
        img = data.get_input(idx=i)
        img = img[1:4].numpy()
        img = np.moveaxis(img, 0, -1)
        x_train.append(img)

    for i in val.indices:
        img = data.get_input(idx=i)
        img = img[1:4].numpy()
        img = np.moveaxis(img, 0, -1)
        x_val.append(img)

    for i in test.indices:
        img = data.get_input(idx=i)
        img = img[1:4].numpy()
        img = np.moveaxis(img, 0, -1)
        x_test.append(img)

    x_train_norm, x_val_norm, x_test_norm = normalizeArr(x_train), normalizeArr(x_val), normalizeArr(x_test)
    x_train_norm, y_train = shuffle(x_train_norm, y_train)
    x_val_norm, y_val = shuffle(x_val_norm, y_val)
    x_test_norm, y_test = shuffle(x_test_norm, y_test)

    neg, pos = np.bincount(y_train)
    tot = neg + pos

    weight_0 = (1 / neg) * (tot / 2.0)
    weight_1 = (1 / pos) * (tot / 2.0)

    print(weight_0, weight_1)

    class_weight = {0: weight_0, 1: weight_1}

    EPOCHS = 15
    BATCH_SIZE = 256

    modelCnn = createCnn()
    modelCnn.summary()

    modelCnn.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=metrics_model)

    history = modelCnn.fit(x_train_norm, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val_norm, y_val),
                    class_weight=class_weight)

    probability_model = tf.keras.Sequential([modelCnn,
                    tf.keras.layers.Softmax()])


    predictions = probability_model.predict(x_test_norm, batch_size=BATCH_SIZE)

    preds = []
    for i in range(len(predictions)):
        pred = np.argmax(predictions[i])
        preds.append(pred)

    target_names = ['Class 0', 'Class 1']

    print("ROC_AUC-Score: ", roc_auc_score(y_test, preds))

    precision, recall, _ = precision_recall_curve(y_test, preds)
    plotLPR = plot_metrics(history)

    print("AUC-score: ", auc(recall, precision))
    print(classification_report(y_test, preds, target_names=target_names))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(16, 16))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("acc_loss.pdf")
    plt.show()

    results = modelCnn.evaluate(x_test_norm, y_test,
                                           batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(modelCnn.metrics_names, results):
      print(name, ': ', value)
    print()

    conMat = plot_cm(y_test, preds)


if __name__ == '__main__':
    main()
