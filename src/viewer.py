import time
import os
import glob

import open3d as o3d
import numpy as np
import imageio
import cv2 as cv

from src.utility import DATA_DIR, FLIP_TRANSFROM, remove_previously_recorded_pcd_frames, read_hdf5_dataset
from src.avatar3D import Avatar3D


class Viewer:
    """Class Viewer is used to view point cloud frames.

    Remarks:
        - 'sleep_per_frame' should be adjusted to get a better viewing speed, because the point clouds
        are not captured at exactly 30 fps.

    Functionality include:
        - Real-time viewing of point cloud frames.
        - Replay already recorded dataset with point cloud frames.
        - Creating gifs for already recorded dataset with point cloud frames
    """

    def __init__(self, avatar: Avatar3D=None):
        """Viewer constructor

        :param avatar: Avatar3D=None
        """
        self.avatar = avatar

    def real_time_render(self, sleep_per_frame: float=0.033):
        """Real-time view of humanoid point cloud.

        Remarks:
            - Depending on computer/gpu specification, voxel down sampling may be needed.
            - 'sleep_per_frame' can be adjusted to slow down frame update frequency

        :param sleep_per_frame: float
        :return:
        """

        if self.avatar is None:
            print("Avatar3D object is None: viewing in real-time requires Avatar3D object")
            print("Exiting..")
            exit(0)

        # remove previously recorded frames
        remove_previously_recorded_pcd_frames()

        print("Starting real-time visualization..")
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        self.avatar.update()
        self.avatar.combined_pcd.transform(FLIP_TRANSFROM)
        vis.add_geometry(self.avatar.combined_pcd)

        i = 0
        try:
            while True:
                vis.remove_geometry(self.avatar.combined_pcd)
                print(f"Updating visualization frame id {i}", end="\r", flush=True)
                self.avatar.update()
                self.avatar.combined_pcd.transform(FLIP_TRANSFROM)
                vis.add_geometry(self.avatar.combined_pcd)

                vis.poll_events()
                vis.update_renderer()

                i += 1
                time.sleep(sleep_per_frame)
        except KeyboardInterrupt:
            print("Exiting visualization..")
            pass

        vis.destroy_window()
        self.avatar.scan.stop()


    def visualize_frames(self, dataset_filename: str=None, sleep_per_frame: float=0.05):
        """Replays the point clouds frames from dataset with specified filename.

        'sleep_per_frame' can be adjusted to slow down frame update frequency.

        Raises exception if dataset with given filename is not found.

        If no dataset is specified, attempts to read all individual .pcd files.
        Raises exception if none are found.

        :param dataset_filename: str
        :param sleep_per_frame: float
        :return:
        """

        # get point cloud frames
        point_clouds, _ = self._get_point_cloud_frames(dataset_filename)

        print("Starting visualization..")
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        frame_id = 0

        # first frame
        pcd_frame = point_clouds[frame_id]

        # if dataset was found and used, convert from numpy array to open3d point cloud
        if type(pcd_frame).__module__ == np.__name__:
            pcd_frame = o3d.geometry.PointCloud()
            pcd_frame.points = o3d.utility.Vector3dVector(point_clouds[frame_id])


        # flip and add point cloud
        pcd_frame.transform(FLIP_TRANSFROM)
        vis.add_geometry(pcd_frame)

        try:
            while True:
                vis.remove_geometry(pcd_frame)

                try:
                    pcd_frame = point_clouds[frame_id]
                    if type(pcd_frame).__module__ == np.__name__:
                        pcd_frame = o3d.geometry.PointCloud()
                        pcd_frame.points = o3d.utility.Vector3dVector(point_clouds[frame_id])
                except IndexError:
                    print("\nNo more frames to visualize..")
                    print("Exiting visualization..")
                    break

                # update point cloud
                print(f"Updating visualization frame id {frame_id}", end="\r", flush=True)
                pcd_frame.transform(FLIP_TRANSFROM)
                vis.add_geometry(pcd_frame)

                vis.poll_events()
                vis.update_renderer()

                frame_id += 1

                # 1/fps, will be slightly slower than 30 fps
                time.sleep(sleep_per_frame)
        except KeyboardInterrupt:
            print("\nExiting visualization..")
            pass

        vis.destroy_window()

    def create_gif_of_point_cloud_frames(self, dataset_filename: str=None):
        """Creates a gif of the point cloud frames.

        Specfify which dataset to create a gif out of.

        :param dataset_filename: str
        :return:
        """

        # get point cloud frames
        point_clouds, _ = self._get_point_cloud_frames(dataset_filename)

        print("Creating gif..")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)

        frame_id = 0

        images = []

        # first frame
        pcd_frame = point_clouds[frame_id]

        # if dataset was found and used, convert from numpy array to open3d point cloud
        if type(pcd_frame).__module__ == np.__name__:
            pcd_frame = o3d.geometry.PointCloud()
            pcd_frame.points = o3d.utility.Vector3dVector(point_clouds[frame_id])

        # flip and add point cloud
        pcd_frame.transform(FLIP_TRANSFROM)
        vis.add_geometry(pcd_frame)

        try:
            while True:
                vis.remove_geometry(pcd_frame)

                try:
                    pcd_frame = point_clouds[frame_id]
                    if type(pcd_frame).__module__ == np.__name__:
                        pcd_frame = o3d.geometry.PointCloud()
                        pcd_frame.points = o3d.utility.Vector3dVector(point_clouds[frame_id])
                except IndexError:
                    print("\nNo more frames to save..")
                    break

                # update point cloud
                print(f"    Updating gif frame id {frame_id}", end="\r", flush=True)
                pcd_frame.transform(FLIP_TRANSFROM)
                vis.add_geometry(pcd_frame)

                vis.poll_events()
                vis.update_renderer()

                image = np.asarray(vis.capture_screen_float_buffer(True))
                image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                images.append(image)

                frame_id += 1
        except KeyboardInterrupt:
            pass

        vis.destroy_window()

        print("Saving gif..")
        save_path = os.path.join(DATA_DIR, f"gif/{dataset_filename}.gif")
        imageio.mimsave(save_path, images, duration=0.05)
        print("Finished creating and saving gif")

    def show_single_frame(self, frame_id: int, dataset_filename: str=None):
        """Displays a single point cloud frame.

        Specify the frame id and from which dataset to get point cloud frame.

        :param frame_id: int
        :param dataset_filename: str
        :return:
        """

        if dataset_filename is None:
            save_path = os.path.join(DATA_DIR, f"pcd/{frame_id}.pcd")
            pcd_frame = o3d.io.read_point_cloud(save_path)
            o3d.visualization.draw_geometries([pcd_frame])
        else:
            dataset = read_hdf5_dataset(dataset_filename)
            point_clouds = dataset["point_clouds"]
            attributes = dataset.attrs
            if frame_id in range(attributes["number_of_frames"]):
                pcd_frame_points = point_clouds[frame_id]
                pcd_frame = o3d.geometry.PointCloud()
                pcd_frame.points = o3d.utility.Vector3dVector(pcd_frame_points)
            else:
                print("Frame_id is not valid")
                return

            o3d.visualization.draw_geometries([pcd_frame])

    @staticmethod
    def _get_point_cloud_frames(dataset_filename: str) -> tuple:
        """Internal function used to get point cloud frames from specified dataset.

        Attempts to read individual .pcd files if no dataset has been specified and they exist.

        :param dataset_filename:
        :return:
        """

        print("Reading point cloud frames from file,")
        print("please wait..")
        if dataset_filename is not None:
            dataset = read_hdf5_dataset(dataset_filename)
            point_cloud_frames = dataset["point_clouds"][:]
            attributes = dataset.attrs
        else:
            pcd_path = os.path.join(DATA_DIR, "pcd/*.pcd")
            pcd_files = sorted(glob.glob(pcd_path), key=os.path.getmtime)
            if len(pcd_files) == 0:
                raise FileNotFoundError

            pcd_frames = []
            for pcd_frame_path in pcd_files:
                pcd_frame = o3d.io.read_point_cloud(pcd_frame_path)
                pcd_frames.append(pcd_frame)

            point_cloud_frames = pcd_frames
            attributes = len(pcd_files)

        print("Finished reading point cloud frames")
        return point_cloud_frames, attributes
