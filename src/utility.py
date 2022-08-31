import os
import glob
import numpy as np
import open3d as o3d
import cv2 as cv
import h5py
import gtsam

# current working directory used to save images and point clouds
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data"))

# in seconds
SYNC_FRAMES_TIMEOUT = 0.5

# in microseconds
MAX_ALLOWED_FRAME_TIME_ERROR = 100

# resolution of depth_mode=NFOV_UNBINNED depth images
NFOV_UNBINNED_DEPTH_MODE_RESOLUTION = (640, 576)
NFOV_UNBINNED_DEPTH_MODE_MAX_POINTS = NFOV_UNBINNED_DEPTH_MODE_RESOLUTION[0] * NFOV_UNBINNED_DEPTH_MODE_RESOLUTION[1]

# flip transform used when visualizing point clouds
FLIP_TRANSFROM = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

# default avatar configuration dictionary, with no verbose, downsampling and filtering
DEFAULT_AVATAR_CONFIG = {"verbose": False,
                         "voxel_down_sample": False,
                         "voxel_size": 0.01,
                         "filter_outliers": False,
                         "nb_neighbors": 20,
                         "std_ratio": 2.0,
                         "radial_filter": False,
                         "nb_points": 50,
                         "radius": 0.05,
                         "save_point_cloud_frames": True,
                         "stride": 1}


class suppress_stdout_stderr(object):
    """Original comment:

    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions

    Comment from 3DAvatar project author:

    The class is used to suppress non-critical error messages originating from Azure Kinect SDK C/C++ sub-function.
    The non-critical errors are "uncatchable", and this class has to be used to avoid filling the console with
    unnecessary messages. The non-critical error messages is due to setting synchronized_images_only=False in
    device configuration. Is not recommended by SDK when doing synchronized capture because color and depth image
    arrive at different times and not simultaneously.
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def read_image_from_file(device_id: int, image_type: str = "ir"):
    image = None
    img_path = os.path.join(CURRENT_DIR, f"captured_images/{image_type}/{image_type}_device_{device_id}.png")

    try:
        with open(img_path, "r") as f:
            image = cv.imread(img_path)
    except FileNotFoundError:
        print(f"Image of type {image_type.upper()} for device {device_id} not found")
        pass

    return image


def read_point_cloud_from_file(device_id: int):
    file_path = os.path.join(DATA_DIR, f"point_clouds/point_cloud_device{device_id}.pcd")
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def remove_previously_recorded_pcd_frames():
    pcd_path = os.path.join(DATA_DIR, f"pcd/*.pcd")
    for f in glob.glob(pcd_path):
        os.remove(f)


def read_hdf5_dataset(dataset_filename: str) -> h5py.File:
    save_path = os.path.join(DATA_DIR, f"dataset/hdf5/{dataset_filename}.h5")

    try:
        f = h5py.File(save_path, "r")
        return f
    except FileNotFoundError:
        print(f"Dataset {dataset_filename}.h5 not found")
        print("Exiting..")
        exit(1)


def X(i):
    """Create key for pose i."""
    return int(gtsam.symbol(str('x'), i))


def L(j):
    """Create key for landmark j."""
    return int(gtsam.symbol(str('l'), j))









