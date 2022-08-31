# 3DAvatar: 3D camera rig and 3D humanoid reconstruction

#### Summer project by Didrik Spanne Reilstad at SINTEF - Summer 2022
## Table of content
- [Description](#description)
- [How to use](#how-to-use)
  - [Requirements](#requirements)
  - [Create environment](#create-environment)
  - [Clone repository](#clone-repository)
  - [Use source files](#use-source-files)
  - [3D camera rig setup](#3d-camera-rig-setup)
- [Project structure](#project-structure)
  - [src/](#src)
  - [examples/](#examples)
  - [data/](#data)

## Description <div id="description"></div>
3DAvatar is a tool for 3D humanoid reconstruction using a 3D camera rig.

The 3D camera rig consists of Azure Kinect cameras. Using the synchronously captured depth images from each camera, a point cloud of a human standing in the camera rig area is generated.

Use cases are: 
- recording point cloud frames of certain movements for further analysis.
- replaying recorded point cloud frames
- real-time viewing of point cloud

<img src="data/gif/point_cloud.gif" width="70%" height="70%"></img>

## How to use <div id="how-to-use"></div>
### Requirements <div id="requirements"></div>
- All required python packages are found in *requirements.txt*
- Tested on Ubuntu 20.04.4
- Additional USB hubs may be required depending on how many Azure Kinect devices are used

### Create environment <div id="create-environment"></div>
Create a conda environment with the *requirements.txt* file containing all required packages to use the source files.  
Run:
```bash
conda create --name [env name] --file requirements.txt
```

### Clone repository <div id="clone-repository"></div>
Clone the repository:
```bash
git clone ssh://git@git.code.sintef.no/max/3davatar.git
```

### Use source files <div id="use-source-files"></div>
Check that creating the environment and cloning repository was successful by running one of the examples.  
For example:
```bash
cd 3davatar/examples/
python3 scan3D_example.py
```
Continue using project source files as you see fit.

### 3D camera rig setup <div id="3d-camera-rig-setup"></div>
- Every device needs to be connected to a power source and connected to the computer via USB. 
  - Back of device will blink orange if only USB is connected and no power is connected. No light if only power is connected and no USB is connected.
- Next step is to daisy chain the devices using the AUX cables.
  - The master device is the device with only SYNC OUT connected and no SYNC IN connected.
    - If no master device is found, the source files will throw an exception.
    - The coordinate system of the master device will be the world coordinate system (or the reference frame) of the generated humanoid avatar.
  - Connect SYNC OUT of master device to SYNC IN of the next subordinate device.
  - Connect SYNC OUT of the subordinate device above to SYNC IN of the next subordinate device.
  - Repeat step above until you reach the last device. The last subordinate device should only have SYNC IN connected and no SYNC OUT connected.
- Additional remarks:
  - After setup, running the example program *scan3D_example.py* is recommended to check if the devices can see the markers in the saved IR images. The master device is required to see all markers.
  - Lighting conditions affect marker detection. Darker lighting conditions is preferred.
  - Reflective surfaces may also affect marker detection.

## Project structure <div id="project-structure"></div>
### src/ <div id="src"></div>
Directory contains the project source code. The Avatar3D class is the 'main' class, which itself uses the Scan3D class and Calibration class.  
List of source files:
- **avatar3D.py**
  - Contains the 'main' class Avatar3D. The class contains the complete point cloud, after transforming depth to point clouds, processing, and fusing individual point clouds. Creates Scan3D and Calibration object for further use if non if provided manually.
- **scan3D.py**
  - Contains the class Scan3D. The class is used to communicate with the connected Azure Kinects.
- **calibration.py**
  - Contains the class Calibration. The class is used to extract and store the intrinsics and extriniscs. Intrinsics are used for converting from depth image to point cloud, and also pose estimation between device and marker. Extrinsics are used for transforming the point clouds to the same coordinate system in order to create a coherent and complete point cloud of the human.
- **utility.py**
  - Contains various helper functions and constant variables that do not fit in any of the classes above. 
- **viewer.py**
  - Contains the class Viewer. The class is used for real-time viewing the point cloud, or for replaying the recorded point cloud frames. 

### examples/ <div id="examples"></div>
Directory contains example python programs on how one can use the different parts of the project source code.  
List of examples:
- **scan3D_example.py**
  - Shows how to use the Scan3D class by connecting and starting devices, printing device info, capturing synchronously or with a standalone device, and stopping the devices.
- **record_point_clouds_example.py**
  - Shows how to record point cloud frames of a human standing in the camera rig area until keyboard interrupt is received (CTRL-C), and also saving the frames to a HDF5 dataset.
- **replay_point_clouds_example.py**
  - Shows how to replay the recorded point cloud frames either using the HDF5 dataset or the individual .pcd files in the data/pcd/ directory if they have not been removed.
- **real_time_viewer_example.py**
  - Shows how to start real-time point cloud viewer of the human in the camera rig area until keyboard interrupt is received (CTRL-C). Voxel down sampling is recommended if real-time outlier filtration is enabled, or else the real-time viewer will be slow.

### data/ <div id="data"></div>
Directory contains various folders where project data is stored.  
The folders are:
- **dataset/** - contains the saved HDF5 datasets
- **gif/** - contains the saved gif
- **images/** - contains the saved depth and IR images
- **pcd/** - contains the temporary .pcd files used to generate the HDF5 datasets, and should be cleaned up after every recording. See ***examples/record_point_clouds_example.py*** for reference





