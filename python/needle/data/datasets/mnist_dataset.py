from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.transforms = transforms
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = self.images[index].reshape(28, 28, -1)
        y = self.labels[index]
        # print(x.shape)
        x = self.apply_transforms(x)
        return x.reshape(-1, 28*28), y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.images.shape[0]
        ### END YOUR SOLUTION




def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename) as f:
        magic_num = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder('>')[0]
        num_images = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder('>')[0]
        rows = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder('>')[0]
        cols = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder('>')[0]
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows*cols)
        images = images.astype(np.float32)
        images /= 255.0

    with gzip.open(label_filename) as f:
        magic_num = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder('>')[0]
        num_labels = np.frombuffer(f.read(4), dtype=np.uint32).newbyteorder('>')[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        labels = labels.reshape(num_labels)

    return images, labels