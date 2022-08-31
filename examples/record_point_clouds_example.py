import sys
sys.path.append("..")

from src.avatar3D import Avatar3D
from src.utility import DEFAULT_AVATAR_CONFIG, remove_previously_recorded_pcd_frames


def record_point_clouds_example(dataset_id: any):
    print("Starting recording of point clouds..")

    # remove previously recorded frames
    remove_previously_recorded_pcd_frames()

    config = DEFAULT_AVATAR_CONFIG
    config["filter_outliers"] = True

    avatar = Avatar3D(config)
    avatar.update()

    i = 0
    try:
        while True:
            print(f"Updating frame id {i}", end="\r", flush=True)
            avatar.update()
            i += 1
    except KeyboardInterrupt:
        print("\nExiting..")
        pass

    avatar.scan.stop()
    avatar.generate_hdf5_dataset(dataset_id)

    # cleanup temporary .pcd files
    remove_previously_recorded_pcd_frames()

if __name__ == '__main__':

    dataset_id = None
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]

    record_point_clouds_example(dataset_id)
