from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import sys, time
from pympler import asizeof
import torch

import grpc
import data_transfer_pb2
import data_transfer_pb2_grpc
import io
import numpy as np

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class SophonDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            mode: str = "remote_nooff",
            host: str = "localhost",
            epoch: int = 0,
            profiling: bool = False
    ) -> None:
        super(SophonDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.mode = mode
        self.host = host
        self.epoch = epoch

        self.channel = None
        self.stub = None
        
        self.profiling = profiling

        self.sizes = {"-1": -1}
        self.times = {"-1": -1}

        self.decision = {}

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def set_mode(self, mode="remote_nooff"):
        """
        local: read from local storage through pil loader
        remote_nooff: read from remote server through grpc
        remote_alloff: offload all preprocessing
        """
        self.mode == mode
    
    def get_mode(self):
        return self.mode
    
    def set_host(self, host="localhost"):
        self.host = host
    
    def get_host(self):
        return self.host

    def set_epoch(self, epoch=0):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    def set_channel(self, channel):
        self.channel = channel
    
    def set_decision(self, decision):
        self.decision = decision

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if "remote" in self.mode:
            if self.stub == None:
                self.channel = grpc.insecure_channel(f'{self.host}:50051', options=[('grpc.max_receive_message_length', 200 * 1024 * 1024)])
                self.stub = data_transfer_pb2_grpc.DataTransferStub(self.channel)

        if self.mode == "local":
            return self.__getitem_local__(index)
        elif self.mode == "remote_nooff":
            return self.__getitem_remote_nooff__(index)
        elif self.mode == "remote_alloff":
            return self.__getitem_remote_alloff__(index)
        elif self.mode == "remote_piloff":
            return self.__getitem_remote_piloff__(index)
        elif self.mode == "remote_sophon":
            return self.__getitem_remote_sophon__(index)
        else:
            raise NotImplementedError("mode {} not supported".format(mode))

    def __getitem_remote_sophon__(self, index: int) -> Tuple[Any, Any]:
        """
        Read from remote server
        Offload all preprocessing on the PIL image before converted to tensor.
        """
        path, target = self.samples[index]

        # if self.decision[index] == -1:
        # # if os.path.getsize(path) < 150528:
        # # if False:
        #     pproc = -1
        #     # return self.__getitem_remote_nooff__(index)
        # elif self.decision[index] == 1:
        #     pproc = 1
        #     # return self.__getitem_remote_piloff__(index)
        # else:
        #     raise NotImplementedError("Decision {} not supported".format(self.decision[index]))

        pproc = self.decision[index]

        sample = self.loader(path, host=self.host, pproc=pproc)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        start_transform = pproc
        if pproc == -1:
            start_transform = 0
        for i in range(start_transform,4):
            sample = self.transform.transforms[i](sample)

        self.times[index] = None
        self.sizes[index] = None

        return sample, target

    def __getitem_remote_piloff__(self, index: int) -> Tuple[Any, Any]:
        """
        Read from remote server
        Offload all preprocessing on the PIL image before converted to tensor.
        """
        path, target = self.samples[index]
        # sample = remote_piloff_loader(path, host=self.host)
        sample = remote_piloff_loader(path, stub=self.stub)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        for i in range(2,4):
                sample = self.transform.transforms[i](sample)
        # print(sample.shape)
        self.times[index] = None
        self.sizes[index] = None

        return sample, target
    
    def __getitem_remote_alloff__(self, index: int) -> Tuple[Any, Any]:
        """
        Read from remote server
        All preprocessing offloaded
        """
        path, target = self.samples[index]
        sample = self.loader(path, stub=self.stub)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        self.times[index] = None
        self.sizes[index] = None

        return sample, target


    def __getitem_remote_nooff__(self, index: int) -> Tuple[Any, Any]:
        """
        Read from remote server
        No preprocessing offloading
        """
        path, target = self.samples[index]

        loader_start = time.time()
        # sample = remote_nooff_loader(path, host=self.host)
        sample, decode_time = remote_nooff_loader(path, stub=self.stub)
        loader_end = time.time()
        loader_time = loader_end - loader_start
        # print(f"loader time: {loader_time}")

        sizes = {}
        times = {}

        # sizes["index"] = index
        times["loader"] = loader_time
        times["decoding"] = decode_time

        filesize = os.path.getsize(path)
        sizes['raw'] = filesize

        # decode_tobytes_start = time.time()
        sizes['decoding'] = sys.getsizeof(sample.tobytes())
        # decode_tobytes_end = time.time()
        # decode_tobytes_time = decode_tobytes_end - decode_tobytes_start
        # print(f"decode tobytes time: {decode_tobytes_time}")

        sizes['original_shape'] = sample.size

        if self.transform is not None:
            # sample = self.transform(sample)
            compose = self.transform
            for t in compose.transforms:

                transform_start = time.time()
                sample = t(sample)
                transform_end = time.time()
                transform_time = transform_end - transform_start
                times["{}".format(type(t).__name__)] = transform_time

                # print(sample)
                if torch.is_tensor(sample):
                    # sizes['after {}'.format(t)] = sample.nelement() * sample.element_size()
                    # tensor_numpy_tobytes_start = time.time()
                    sizes['{}'.format(type(t).__name__)] = sys.getsizeof(sample.numpy().tobytes())
                    # tensor_numpy_tobytes_end = time.time()
                    # print("tensor_numpy_tobytes_time: {}".format(tensor_numpy_tobytes_end - tensor_numpy_tobytes_start))
                    # print(list(sample.getdata()))
                else:
                    sizes['{}'.format(type(t).__name__)] = sys.getsizeof(sample.tobytes())
                    
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        self.times[index] = times
        self.sizes[index] = sizes

        # print("{}: {}".format(index, times))
        # print(self.times)

        # print(times)
        # print(sizes)

        return sample, target




    def __getitem_local__(self, index: int) -> Tuple[Any, Any]:
        
        path, target = self.samples[index]

        loader_start = time.time()
        sample = self.loader(path)
        loader_end = time.time()
        loader_time = loader_end - loader_start
        # print(f"loader time: {loader_time}")

        sizes = {}
        times = {}

        # sizes["index"] = index
        times["loader"] = loader_time

        filesize = os.path.getsize(path)
        sizes['size raw'] = filesize

        # decode_tobytes_start = time.time()
        sizes['size after decoding'] = sys.getsizeof(sample.tobytes())
        # decode_tobytes_end = time.time()
        # decode_tobytes_time = decode_tobytes_end - decode_tobytes_start
        # print(f"decode tobytes time: {decode_tobytes_time}")

        sizes['original shape'] = sample.size

        if self.transform is not None:
            # sample = self.transform(sample)
            compose = self.transform
            for t in compose.transforms:

                transform_start = time.time()
                sample = t(sample)
                transform_end = time.time()
                transform_time = transform_end - transform_start
                times["{}".format(type(t).__name__)] = transform_time

                # print(sample)
                if torch.is_tensor(sample):
                    # sizes['after {}'.format(t)] = sample.nelement() * sample.element_size()
                    # tensor_numpy_tobytes_start = time.time()
                    sizes['size after {} numpy tobytes'.format(type(t).__name__)] = sys.getsizeof(sample.numpy().tobytes())
                    # tensor_numpy_tobytes_end = time.time()
                    # print("tensor_numpy_tobytes_time: {}".format(tensor_numpy_tobytes_end - tensor_numpy_tobytes_start))
                    # print(list(sample.getdata()))
                else:
                    sizes['size after {}'.format(type(t).__name__)] = sys.getsizeof(sample.tobytes())
                    
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        self.times[index] = times
        self.sizes[index] = sizes

        # print("{}: {}".format(index, times))
        # print(self.times)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        start = time.time()
        img = Image.open(f)
        end = time.time()
        image_open_time = end-start

        filesize = os.path.getsize(path)
        initial_img_size = asizeof.asizeof(img)

        start = time.time()
        img.load()
        end = time.time()
        img_load_time = end-start

        size_after_load = asizeof.asizeof(img)

        start = time.time()
        res = img.convert('RGB')
        end = time.time()
        img_corvert_time = end-start

        size_after_convert = asizeof.asizeof(res)
        # print("image_open_time: {}  img_load_time: {}   img_corvert_time: {}".format(image_open_time, img_load_time, img_corvert_time))
        # print("filesize: {} initial_img_size: {}    size_after_load: {} size_after_convert: {}".format(filesize, initial_img_size, size_after_load, size_after_convert))
        # im = res.getdata()
        # img_list = list(im)
        # print("palette: {}  pyaccess: {}    getdata: {} list_lenth: {}  im_sizea: {}".format(
        #                 res.palette, res.pyaccess, res.getdata(), len(img_list),  asizeof.asizeof(im.tobytes())))
        return res

# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def remote_nooff_loader(path: str, stub = None):
    # with grpc.insecure_channel(f'{host}:50051', options=[('grpc.max_receive_message_length', 200 * 1024 * 1024)]) as channel:
    # stub = data_transfer_pb2_grpc.DataTransferStub(channel)
    response = stub.get_data_nooff(data_transfer_pb2.DataRequest(fname=path, pproc=0, size=100000))
    print("nooff Client received: " + response.fname)
    decode_start = time.time()
    img = Image.open(io.BytesIO(response.image))
    res = img.convert('RGB')
    decode_end = time.time()
    decode_time = decode_end - decode_start
    return res, decode_time


def remote_alloff_loader(path: str, stub = None):
    # with grpc.insecure_channel(f'{host}:50051', options=[('grpc.max_receive_message_length', 200 * 1024 * 1024)]) as channel:
    #     stub = data_transfer_pb2_grpc.DataTransferStub(channel)
        # Replace 'path/to/yourfile.txt' with the actual file path
        response = stub.get_data_alloff(data_transfer_pb2.DataRequest(fname=path, pproc=0, size=100000))
        print("alloff Client received: " + response.fname)
        
            # Convert byte data back to a numpy array
        dtype = np.dtype(response.dtype)
        tensor_data = np.frombuffer(response.image, dtype=dtype).copy()
        tensor_data = tensor_data.reshape(response.shape)
        
        # Convert numpy array back to a PyTorch tensor
        res = torch.from_numpy(tensor_data)
        # res = response.image
        return res

def remote_piloff_loader(path: str, stub = None):
    # with grpc.insecure_channel(f'{host}:50051', options=[('grpc.max_receive_message_length', 200 * 1024 * 1024)]) as channel:
    # stub = data_transfer_pb2_grpc.DataTransferStub(channel)
    # Replace 'path/to/yourfile.txt' with the actual file path
    response = stub.get_data_piloff(data_transfer_pb2.DataRequest(fname=path, pproc=0, size=100000))
    print("piloff Client received: " + response.fname)
    img = Image.frombytes(response.pil_mode, (response.width, response.height), response.image)
    return img

def remote_sophon_loader(path: str, host: str = "localhost", pproc=-1):
    with grpc.insecure_channel(f'{host}:50051', options=[('grpc.max_receive_message_length', 200 * 1024 * 1024)]) as channel:
        stub = data_transfer_pb2_grpc.DataTransferStub(channel)
        # Replace 'path/to/yourfile.txt' with the actual file path
        response = stub.get_data_sophon(data_transfer_pb2.DataRequest(fname=path, pproc=pproc, size=100000))
        print("sophon Client received: " + response.fname)
        if pproc == -1:
            img = Image.open(io.BytesIO(response.image))
            img = img.convert('RGB')
            return img
        elif pproc <= 2:
            img = Image.frombytes(response.pil_mode, (response.width, response.height), response.image)
            return img
        elif pproc <= 4:
            dtype = np.dtype(response.dtype)
            tensor_data = np.frombuffer(response.image, dtype=dtype).copy()
            tensor_data = tensor_data.reshape(response.shape)
            
            # Convert numpy array back to a PyTorch tensor
            res = torch.from_numpy(tensor_data)
            return res
        else:
            raise NotImplementedError("pproc {} not supported by sophon".format(pproc))




def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class SophonImageFolder(SophonDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = remote_nooff_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            mode: str = "remote_nooff",
            host: str = "localhost",
            epoch: int = 0,
            profiling: bool = False
    ):
        if mode == "remote_nooff":
            loader = remote_nooff_loader
        elif mode == "local":
            loader = default_loader
        elif mode == "remote_alloff":
            loader = remote_alloff_loader
        elif mode == "remote_piloff":
            loader = remote_piloff_loader
        elif mode == "remote_sophon":
            loader = remote_sophon_loader
        else:
            raise NotImplementedError("mode {} not supported".format(mode))
        super(SophonImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          mode=mode,
                                          host=host,
                                          epoch = epoch,
                                          profiling = profiling)
        self.imgs = self.samples
