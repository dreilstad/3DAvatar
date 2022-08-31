import sys
sys.path.append("..")

from src.avatar3D import Avatar3D
from src.viewer import Viewer
from src.utility import DEFAULT_AVATAR_CONFIG, remove_previously_recorded_pcd_frames


def real_time_viewer_example(dataset_id: any):

    # recommend voxel down sampling for real time visualization
    avatar_config = {}
    avatar_config["verbose"] = False

    avatar_config["voxel_down_sample"] = False
    avatar_config["voxel_size"] = 0.01

    avatar_config["filter_outliers"] = True
    avatar_config["nb_neighbors"] = 20
    avatar_config["std_ratio"] = 2.0

    avatar_config["radial_filter"] = False
    avatar_config["nb_points"] = 50
    avatar_config["radius"] = 0.05

    avatar_config["save_point_cloud_frames"] = True
    avatar_config["stride"] = 1

    avatar = Avatar3D(avatar_config)
    viewer = Viewer(avatar)
    viewer.real_time_render(sleep_per_frame=0.0)

    # save to dataset
    avatar.generate_hdf5_dataset(dataset_id)
    remove_previously_recorded_pcd_frames()

if __name__ == '__main__':

    dataset_id = None
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]

    real_time_viewer_example(dataset_id)
