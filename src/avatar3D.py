import os
import glob
import copy
from threading import Thread

import open3d as o3d
import numpy as np
import h5py

from numba import jit
from datetime import datetime

from src.utility import DATA_DIR, NFOV_UNBINNED_DEPTH_MODE_MAX_POINTS
from src.scan3D import Scan3D
from src.calibration import Calibration


class Avatar3D:
    """Class Avatar3D represents the avatar/model of the person.

    It is the 'main' class for creating the avatar, and it uses both Scan3d and Calibration.

    Functionality include:
        - Updating the 3D avatar by capturing and transforming from depth image to point cloud.
          Processing of the point cloud is performed if set in configuration.
        - Writes point cloud frames to file, if set in configuration.
        - Transforming depth image to point cloud
        - Thresholding points to only get point cloud of area of interest
        - Transforming points from one coordinate system to another coordinate system
    """

    def __init__(self, config: dict, scan: Scan3D=None):
        """Avatar3D constructor

        The variable 'combined_pcd' is the PointCloud object containing all points of the avatar.
        With the update function, this can be used to record frames of a person.

        Exits if calibration of extrinsics fails. Retry or check if markers are visible and/or occluded.

        Initializing an object of the Avatar3D class with only config argument will automatically:
            - Initialize Scan3D object, connect, start, and capture with devices.
            - Calibrate intrinsics and extrinsics.
            - Calculate an approximated max_threshold depending on the distance to each camera pole
              in the rig. The max_threshold defines the 'scanning' area, which will be a triangular area.

        :param config: dict
        :param scan: Scan3D=None
        """

        self.config = config

        if scan is None:
            if self.config["verbose"]:
                print("Starting 3D scan..")

            self.scan = Scan3D(verbose=self.config["verbose"])
            self.scan.connect()

            if self.config["verbose"]:
                self.scan.print_device_info()

            self.scan.start()
            self.scan.capture()

        else:
            self.scan = scan

        self.num_devices = self.scan.num_devices

        self.calibration = Calibration(self.scan, self.config["verbose"], False)
        calibration_success = self.calibration.init_calibration()
        if not calibration_success:
            print("Failed to initialize extrinsics")
            print("Exiting..")
            exit(1)

        #self.scan.stop()
        #exit(0)

        # used when thresholding
        self.min_threshold = 0.1
        self.max_threshold = self.calibration.average_distance_to_markers * 0.7

        # used when generating hdf5 dataset
        self.highest_number_of_points_in_frame = 0

        # avatar point cloud object
        self.combined_pcd = o3d.geometry.PointCloud()

        self.frame_id = 0

    def update(self):
        """Updates the point cloud by synchronized capture, transforming from depth image to point cloud
        for each device, fusing them and processing the point cloud if set in configuration.

        Writes current point cloud frame to file if set in configuration.

        Remarks:
            - Some voxel down sampling is necessary if outlier filtration is required.
              If not, the program slows down to the point that fast movement by the person is not captured
              due to the amount of points.

        :return:
        """

        successful_capture = self.scan.capture()
        while not successful_capture:
            successful_capture = self.scan.capture()

        # ctrl-c in the middle of numba function will throw SystemError,
        # raise KeyboardInterrupt to exit safely
        try:
            points = self.depth_to_point_cloud()
        except SystemError:
            raise KeyboardInterrupt

        self.combined_pcd.points = o3d.utility.Vector3dVector(points)

        # voxel down sample
        if self.config["voxel_down_sample"] and self.config["voxel_size"] > 0.0:
            self.combined_pcd = self.combined_pcd.voxel_down_sample(voxel_size=self.config["voxel_size"])

        # write point cloud frame to file
        if self.config["save_point_cloud_frames"]:

            # used when generating hdf5 dataset after recording frames
            if self.highest_number_of_points_in_frame < len(self.combined_pcd.points):
                self.highest_number_of_points_in_frame = len(self.combined_pcd.points)

            # save point cloud with thread
            thread = Thread(target=self.write_point_cloud_to_file,
                            args=(copy.deepcopy(self.combined_pcd), self.frame_id))
            thread.start()
            #self.write_point_cloud_to_file()
            self.frame_id += 1

    @staticmethod
    def write_point_cloud_to_file(pcd: o3d.geometry.PointCloud, frame_id: int):
        save_path = os.path.join(DATA_DIR, f"pcd/{frame_id}.pcd")
        o3d.io.write_point_cloud(save_path, pcd, compressed=True)

    def depth_to_point_cloud(self, captures_list: list=None) -> np.ndarray:
        """Transforms depth images from each device to point clouds. The point clouds are
        thresholded and fused by transforming the point clouds to the same coordinate system.

        Points array is initialized to store max number of points, but only a fraction of this
        is used. The reason is to facilitate storing points from each device by slicing instead
        of append/extend operations, which are slower operations compared to slicing.

        Only the part of the points array with actual points will be returned. The size of the array
        will be the number of points of the avatar.

        :param captures_list: list=None
        :return: points: np.ndarray
        """
        # check if captures_list is provided
        if captures_list is None:
            captures_list = self.scan.captures_list

        # init array size equal to the max number of points for optimization reasons
        points = np.zeros((NFOV_UNBINNED_DEPTH_MODE_MAX_POINTS*self.num_devices, 3))
        points_indices = [0]

        # iterate depth image from every device and convert to point cloud
        for device_id, capture in enumerate(captures_list):

            # transform depth image to point cloud using intrinsics (focal length (fx,fy) and optical centers (cx,cy))
            height, width = capture.depth.data.shape
            params = self.scan.calibration_mode[device_id].depth_cam_cal.intrinsics.parameters.param
            xyz_data = self._fast_depth_image_to_point_cloud_numba(depth_image=capture.depth.data,
                                                                   height=height,
                                                                   width=width,
                                                                   stride=self.config["stride"],
                                                                   focal_length=(params.fx, params.fy),
                                                                   optical_centre=(params.cx, params.cy))

            # thresholding removes outliers and empty points, save number of remaining points
            xyz_data = self.threshold_points(xyz_data)
            points_indices.append(points_indices[device_id] + xyz_data.shape[0])

            # slicing to store points is used instead of append/extend
            # reason is significant speedup for increasing the quality of real-time viewing
            points[points_indices[device_id]:points_indices[device_id+1]] = self.transform_points(device_id, xyz_data)

        return points[:points_indices[-1]]

    @staticmethod
    @jit(nopython=True)
    def _fast_depth_image_to_point_cloud_numba(depth_image: np.ndarray, height: int, width: int,
                                               stride: int, focal_length: tuple, optical_centre: tuple) -> np.ndarray:
        """Internal function for transforming depth image to 3d points.

        Uses focal length (fx, fy) and optical centre (cx, cy) to perform the transformation.
        This function uses numba to speedup transformation.

        :param depth_image: np.ndarray
        :param height: int
        :param width: int
        :param stride: int
        :param focal_length: tuple
        :param optical_centre: tuple
        :return: points: np.ndarray
        """

        # numba-jit version of np.mgrid
        x_values = np.arange(0, height, stride)
        y_values = np.arange(0, width, stride)

        u = np.empty((len(x_values), len(y_values)))
        for j, y in enumerate(y_values):
            for i, x in enumerate(x_values):
                u[i, j] = x

        v = np.empty((len(x_values), len(y_values)))
        for i, x in enumerate(x_values):
            v[i, :] = y_values

        z = depth_image[0:height:stride, 0:width:stride] / 1000.0
        x = (v - optical_centre[0]) * z / focal_length[0]
        y = (u - optical_centre[1]) * z / focal_length[1]

        return np.stack((x, y, z), axis=-1).reshape(-1, 3)

    @staticmethod
    def _fast_depth_image_to_point_cloud(focal_length: float, optical_centre: float,
                                         depth_image: np.ndarray, height: int, width: int,
                                         stride: int=1) -> np.ndarray:
        """Internal function for transforming depth image to 3d points.

        Uses focal length (fx, fy) and optical centre (cx, cy) to perform the transformation.
        This function uses np.mgrid from numpy to perform the transformation.

        :param focal_length: float
        :param optical_centre: float
        :param depth_image: np.ndarray
        :param height: int
        :param width: int
        :param stride: int
        :return: points: np.ndarray
        """

        # pure numpy version
        grid = np.mgrid[0:height:stride, 0:width:stride]
        u, v = grid[0], grid[1]

        z = depth_image[0:height:stride, 0:width:stride] / 1000.0
        x = (v - optical_centre[0]) * z / focal_length[0]
        y = (u - optical_centre[1]) * z / focal_length[1]

        return np.stack((x, y, z), axis=-1).reshape(-1, 3)

    def threshold_points(self, points: np.ndarray) -> np.ndarray:
        """Thresholds points by z-value, i.e. the depth value.

        Only the points with a depth value of more than the min_threshold and less than the max_threshold is kept.

        Remarks:
            - Calculating the distance is an option but the sum-operation is more expensive and slows down the program.

        :param points: np.ndarray
        :return: points: np.ndarray
        """

        # distance threshold
        #distance = np.sqrt(np.sum(points[..., :] ** 2, axis=1))
        #points = points[(distance >= self.min_threshold) & (distance <= self.max_threshold), :]

        # using distance instead of only z-value (depth) will remove more outliers,
        # but significantly reduces the quality during real-time viewing
        z_value = points[:, 2]
        points = points[(z_value > self.min_threshold) & (z_value < self.max_threshold), :]

        return points

    def transform_points(self, device_id: int, points: np.ndarray) -> np.ndarray:
        """Transforms points from one coordinate system to another coordinate system.

        Points captured by the master device is not transformed.

        :param device_id: int
        :param points: np.ndarray
        :return: points: np.ndarray
        """

        if device_id == self.num_devices - 1:
            return points

        T_sub_master = np.linalg.inv(self.calibration.extrinsics[device_id])
        points = (T_sub_master[:3, :3] @ points.T + T_sub_master[:3, 3].reshape((3, 1))).T

        return points

    def generate_hdf5_dataset(self, dataset_id: any):
        """Generates an HDF5 dataset of all point clouds frames recorded.

        Creates a dataset with name 'point_clouds', and attributes 'number_of_frames' and 'number_of_points_each_frame'.

        During recording of frames, the highest number of points in a frame is saved. This number is used afterwards
        when generating the dataset for padding all point clouds with [0,0,0] until all point clouds are of same size.

        The .h5 file is saved with the name format: YYYY_MM_DD_id={dataset_id}.h5, where parameter 'dataset_id'
        is the unique identifier. Dataset uses 'gzip' compression.

        If dataset with name YYYY_MM_DD_id={dataset_id}.h5 already exists, the dataset is overwritten.

        :param dataset_id:
        :return:
        """

        if dataset_id is None:
            print("Failed to generate HDF5 dataset: dataset_id is None")
            return

        print("Generating HDF5 dataset,")
        print("please wait..")

        pcd_path = os.path.join(DATA_DIR, "pcd/*.pcd")
        pcd_frames = sorted(glob.glob(pcd_path), key=os.path.getmtime)

        # highest number of points after post processing
        highest_num_of_points = 0

        dataset_save_path = os.path.join(DATA_DIR,
                                         f"dataset/hdf5/{datetime.today().strftime('%Y_%m_%d')}_id={dataset_id}.h5")

        with h5py.File(dataset_save_path, "w") as dataset_pcd:

            all_padded_point_clouds = [None] * len(pcd_frames)

            for frame_id, pcd_frame in enumerate(pcd_frames):

                # read point cloud frame
                point_cloud = o3d.io.read_point_cloud(pcd_frame)

                # filter outliers
                if self.config["filter_outliers"]:
                    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=self.config["nb_neighbors"],
                                                                     std_ratio=self.config["std_ratio"])
                    point_cloud = point_cloud.select_by_index(ind)

                # radial filter
                if self.config["radial_filter"]:
                    cl, ind = point_cloud.remove_radius_outlier(nb_points=self.config["nb_points"],
                                                                radius=self.config["radius"])
                    point_cloud = point_cloud.select_by_index(ind)

                # convert point cloud to numpy array
                points = np.asarray(point_cloud.points)

                # used later to remove redundant padding due to removal of outliers
                if highest_num_of_points < points.shape[0]:
                    highest_num_of_points = points.shape[0]

                # point clouds are padded with [NaN, NaN, NaN] to make them a uniform size
                pad_amount = self.highest_number_of_points_in_frame - points.shape[0]
                points_padded = np.pad(points, ((0, pad_amount), (0, 0)), 'constant', constant_values=(0, np.nan))

                all_padded_point_clouds[frame_id] = points_padded

            all_point_clouds = [pcd[:highest_num_of_points, :] for pcd in all_padded_point_clouds]

            # create a point clouds dataset in the file
            dataset_pcd.create_dataset(name="point_clouds",
                                       data=np.asarray(all_point_clouds),
                                       compression="gzip")

            # add relevant attributes
            dataset_pcd.attrs["number_of_frames"] = len(pcd_frames)
            dataset_pcd.attrs["number_of_points_each_frame"] = highest_num_of_points

        print("Finished generating HDF5 dataset")