import os
import time

import numpy as np
import cv2 as cv
import k4a

from src.utility import DATA_DIR, SYNC_FRAMES_TIMEOUT, MAX_ALLOWED_FRAME_TIME_ERROR, suppress_stdout_stderr


class Scan3D:
    """Class Scan3D for using one or more Azure Kinect cameras.

    Uses the 'k4a' module for communicating with devices, which is an Azure Kinect SDK python wrapper.

    Functionality include:
        - Connecting and starting device/s.
        - Capturing color, depth and ir images from a single device or synchronized with multiple devices
        - Stopping and closing device/s
        - Get intrinsics of each device
        - Write captured images to file
        - Print info about all connected devices
    """

    def __init__(self, verbose: bool=False):
        """Scan3d constructor

        Set verbose=True to enable printing information to console.

        Raises exception if no devices are found.

        :param verbose: bool=False
        """

        # get number of devices
        self.num_devices = k4a.Device.get_device_count()
        print(f"Number of devices found: {self.num_devices}")
        if self.num_devices == 0:
            raise Exception("No devices found!")

        self.devices_list = [None] * self.num_devices
        self.captures_list = [None] * self.num_devices

        self.device_configs = [None] * self.num_devices
        self.calibration_mode = [None] * self.num_devices

        self.verbose = verbose

    def connect(self):
        """Connects to devices.

        All successfully opened devices are added to a list.
        The master device is found and put last in the list for easy access.
        Function raises exception if opening device through 'k4a' fails.
        :return:
        """

        if self.verbose:
            print("Connecting to devices..")

        # Open all connected devices
        for i in range(self.num_devices):
            self.devices_list[i] = k4a.Device.open(i)
            if self.devices_list[i] is None:
                raise Exception("Failed to open device " + str(i))

        # Find master device and put last in list of devices, only relevant when using multiple devices
        for device in self.devices_list:
            if device.sync_out_connected and (not device.sync_in_connected):
                self.devices_list.remove(device)
                self.devices_list.append(device)
                break

    def start(self, device_config: k4a.DeviceConfiguration=None):
        """Starts all connected devices with device configuration.

        Standard device configuration is used if no specific device config is provided.

        Remarks
            - color_format=COLOR_BGRA32 is NOT recommended with multiple devices because the format is not native,
              may result in error due to low USB bandwidth.

        :param device_config: k4a.DeviceConfiguration=None
        :return:
        """

        if self.verbose:
            print("Starting device cameras..")


        if self.num_devices == 1:
            self._start_standalone_camera(device_config)
        else:
            self._start_multiple_synchronized_cameras(device_config)

    def capture(self) -> bool:
        """Captures image with all connected devices.

        With multiple devices the captures are all synchronized using the timestamp.

        :return:
        """

        if self.num_devices == 1:
            if self.verbose:
                print("Capturing from device")

            self.captures_list[0] = self.devices_list[0].get_capture(-1)
            return True
        else:
            if self.verbose:
                print("Capturing from multiple devices and syncing frames..")

            # due to setting synchronized_images_only=False in device config, the SDK prints non-exiting
            # errors which cannot be caught by try-except. With suppress_stdout_stderr prevents this printing
            with suppress_stdout_stderr():
                return self._sync_captured_frames()


    def stop(self):
        """Stops camera of all connected devices and closes them.

        :return:
        """

        if self.verbose:
            print("Stopping device cameras and closing device..")

        for device in self.devices_list:
            device.stop_cameras()
            device.close()

    def _sync_captured_frames(self) -> bool:
        """Internal function used for synchronized capture with multiple devices.

        Captured frames are synchronized if the timestamp error is larger than 100 ms,
        with master device as the reference. Synchronization will timeout if more than 1 second is spent syncing.

        Returns True when frames are synced, False if the synchronization is timed out.

        :return: True/False
        """

        # first capture
        self.captures_list[-1] = self.devices_list[-1].get_capture(-1)
        for device_id, device in enumerate(self.devices_list[:-1]):
            self.captures_list[device_id] = device.get_capture(-1)

        # while loop for syncing frames until all frames are within 100 ms of each other, or until timeout
        synced_frames = False
        start = time.time()
        while not synced_frames:

            # if frames become too out of sync, timeout occurs and returns False
            duration = time.time() - start
            if duration > SYNC_FRAMES_TIMEOUT:
                return synced_frames

            # retrieve master depth image and timestamp, throws AssertionError if depth image has not arrived
            try:
                master_image = self.captures_list[-1].depth
                master_depth_time = master_image.device_timestamp_usec
            except AssertionError:
                master_image = None


            # iterate subordinate devices
            for device_id, device in enumerate(self.devices_list[:-1]):

                # retrieve subordinate depth image and timestamp, throws AssertionError if depth image has not arrived
                try:
                    sub_image = self.captures_list[device_id].depth
                except AssertionError:
                    sub_image = None

                # check if depth image of master and subordinate device has arrived
                if master_image is not None and sub_image is not None:

                    # calculate expected error, small delay between devices is expected to prevent ir interference
                    sub_depth_time = sub_image.device_timestamp_usec
                    expected_sub_depth_time = master_depth_time \
                                              + self.device_configs[device_id].subordinate_delay_off_master_usec \
                                              + self.device_configs[device_id].depth_delay_off_color_usec

                    sub_depth_time_error = sub_depth_time - expected_sub_depth_time


                    # subordinate is lagging too much behind, need to update subordinate
                    if sub_depth_time_error < -MAX_ALLOWED_FRAME_TIME_ERROR:
                        self.captures_list[device_id] = device.get_capture(-1)
                        break

                    # subordinate is ahead of master, need to update master
                    elif sub_depth_time_error > MAX_ALLOWED_FRAME_TIME_ERROR:
                        self.captures_list[-1] = self.devices_list[-1].get_capture(-1)
                        break

                    # captures are sufficiently synchronized, check if all subordinate frames have been synced
                    else:
                        if device_id == self.num_devices-2:
                            synced_frames = True

                elif master_image is None:
                    self.captures_list[-1] = self.devices_list[-1].get_capture(-1)
                    break
                elif sub_image is None:
                    self.captures_list[device_id] = device.get_capture(-1)
                    break

        return synced_frames


    def _start_standalone_camera(self, device_config: k4a.DeviceConfiguration):
        """Internal function used for starting standalone camera.

        Saves depth_mode and color_resolution used for later transforming from depth
        image to point cloud.

        Function raises exception if unable to start camera.

        :param device_config: k4a.DeviceConfiguration
        :return:
        """

        # default standalone config
        if device_config is None:
            device_config = k4a.DeviceConfiguration(
                color_format=k4a.EImageFormat.COLOR_MJPG,
                color_resolution=k4a.EColorResolution.RES_720P,
                depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
                camera_fps=k4a.EFramesPerSecond.FPS_30,
                synchronized_images_only=True,
                depth_delay_off_color_usec=0,
                wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
                subordinate_delay_off_master_usec=0,
                disable_streaming_indicator=False)


        status = self.devices_list[0].start_cameras(device_config)
        if status != k4a.EStatus.SUCCEEDED:
            raise Exception("Failed to start camera")

        self.device_configs[0] = device_config
        self.calibration_mode[0] = self.devices_list[0].get_calibration(
                                                            depth_mode=device_config.depth_mode,
                                                            color_resolution=device_config.color_resolution)

    def _start_multiple_synchronized_cameras(self, device_config: k4a.DeviceConfiguration):
        """Internal function used for starting multiple cameras.

        Saves depth_mode and color_resolution used for later transforming from depth
        image to point cloud of each device.

        Sets color exposure time manually. Recommended for synchronicity.
        Increase exposure value if synchronized color capture is needed.

        Function raises exception if unable to start cameras.

        :param device_config: k4a.DeviceConfiguration
        :return:
        """

        for i, device in enumerate(self.devices_list):

            # manual color exposure time recommended for synchronicity
            device.set_color_control(k4a.EColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
                                     k4a.EColorControlMode.MANUAL,
                                     0)

            device_i_config = self._device_config_multiple_cameras(device, i, device_config)

            status = device.start_cameras(device_i_config)
            if status != k4a.EStatus.SUCCEEDED:
                raise Exception("Failed to start device " + str(i))

            self.device_configs[i] = device_i_config
            self.calibration_mode[i] = device.get_calibration(
                                                    depth_mode=device_i_config.depth_mode,
                                                    color_resolution=device_i_config.color_resolution)

    def _device_config_multiple_cameras(self, device: k4a.Device, device_id: int,
                                        device_config: k4a.DeviceConfiguration=None) -> k4a.DeviceConfiguration:
        """Internal function used for getting device configuration for multiple devices.

        Delay of 160 us between devices is necessary to avoid IR interference.
        Will force correct 'sync_mode', 'delay_off_master_usec' and 'synchronized_images_only'
        config reardless of specified device configuration

        :param device: k4a.Device
        :param device_id: int
        :param device_config: k4a.DeviceConfiguration=None
        :return: device_config: k4a.DeviceConfiguration
        """

        # subordinate specific config settings
        # delay between devices is necessary to avoid ir interference
        sync_mode = k4a.EWiredSyncMode.SUBORDINATE
        delay_off_master_usec = 160 * (device_id + 1)

        # master specific config settings
        if device.sync_out_connected and (not device.sync_in_connected):
            sync_mode = k4a.EWiredSyncMode.MASTER
            delay_off_master_usec = 0

        # use default if no device config is given
        if device_config is None:
            device_config = k4a.DeviceConfiguration(
                color_format=k4a.EImageFormat.COLOR_MJPG,
                color_resolution=k4a.EColorResolution.RES_720P,
                depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
                camera_fps=k4a.EFramesPerSecond.FPS_30,
                synchronized_images_only=False,
                depth_delay_off_color_usec=0,
                wired_sync_mode=sync_mode,
                subordinate_delay_off_master_usec=delay_off_master_usec,
                disable_streaming_indicator=False)
        else:
            device_config.sync_mode = sync_mode
            device_config.delay_off_master_usec = delay_off_master_usec
            device_config.synchronized_images_only = False

        return device_config

    def get_intrinsics(self, device_id: int) -> tuple:
        """Retrieves the intrinsic parameters of specified device.

        The returned intrinsics are the camera matrix and distortion coefficients.

        :param device_id: int
        :return: camera_matrix: np.ndarray, distortion_coeffs: np.ndarray
        """

        intrinsics_params = self.calibration_mode[device_id].depth_cam_cal.intrinsics.parameters.param

        camera_matrix = np.array([[intrinsics_params.fx, 0, intrinsics_params.cx],
                                  [0, intrinsics_params.fy, intrinsics_params.cy],
                                  [0, 0, 1]])

        distortion_coeffs = np.array([intrinsics_params.k1,
                                      intrinsics_params.k2,
                                      intrinsics_params.p1,
                                      intrinsics_params.p2,
                                      intrinsics_params.k3,
                                      intrinsics_params.k4,
                                      intrinsics_params.k5,
                                      intrinsics_params.k6])

        return camera_matrix, distortion_coeffs

    def write_depth_and_ir_to_file(self):
        """Writes depth and IR images to file.

        Normalizes ir and depth image values from 16-bit into 8-bit,
        necessary for using certain functions from openCV.

        :return:
        """

        for i, capture in enumerate(self.captures_list):

            ir_image = cv.normalize(np.clip(capture.ir.data, a_min=0, a_max=1500), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

            depth_image = np.zeros_like(capture.depth.data)
            depth_image = cv.normalize(capture.depth.data, depth_image, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

            # save images
            save_path_ir = os.path.join(DATA_DIR, f"images/ir/ir_device_{i}.png")
            save_path_depth = os.path.join(DATA_DIR, f"images/depth/depth_device_{i}.png")

            cv.imwrite(save_path_ir, ir_image)
            cv.imwrite(save_path_depth, depth_image)

    def print_device_info(self):
        """Prints device info of every connected device.

        :return:
        """

        print("")
        for i, device in enumerate(self.devices_list):
            print(f"Device {i}:")
            # A bug stores the b'' in the serial number, so remove the b and apostrophes.
            print(device.serial_number[2:-1])
            print(str(device.hardware_version))
            print(f"Sync out: {device.sync_out_connected}")
            print(f"Sync in: {device.sync_in_connected}")
            print("")