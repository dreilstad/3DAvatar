import sys
sys.path.append("..")

from src.viewer import Viewer

def replay_point_clouds_example(dataset_filename):

    view = Viewer()
    #view.show_single_frame(100, dataset_filename)
    view.visualize_frames(dataset_filename, sleep_per_frame=0.05)
    #view.create_gif_of_point_cloud_frames(dataset_filename)

if __name__ == '__main__':

    dataset_filename = None
    if len(sys.argv) > 1:
        dataset_filename = sys.argv[1]

    replay_point_clouds_example(dataset_filename)
