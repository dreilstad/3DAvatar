import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gplt

from src.scan3D import Scan3D
from src.utility import X, L


class Calibration:
    """Class Calibration for calibrating intrinsics and extrinsics Azure Kinect cameras.

    Uses openCV to detect markers and estimating pose of device/marker. The transformations between device and marker
    are then used to calculate the transformation between devices, i.e. the extrinsics. Extrinsics are then further
    optimized using GTSAM.

    Master device is used as the world coordinate system.

    Intrinsics are retrieved using each device via the Scan3D object.

    Functionality include:
        - Initializing intrinsics
        - Initializing extrinsics
        - Detecting markers
        - Optimizing extrinsics
    """

    def __init__(self, scan: Scan3D, verbose: bool, draw_detected_markers: bool=False):
        """Calibration constructor

        Set verbose=True to enable printing more information to console.

        :param scan: Scan3D
        :param verbose: bool
        :param draw_detected_markers: bool=False
        """

        self.scan = scan
        self.verbose = verbose
        self.draw_detected_markers = draw_detected_markers

        self.num_devices = self.scan.num_devices
        self.intrinsics = [None] * self.num_devices
        self.extrinsics = [None] * (self.num_devices - 1)

        self.T_master_aruco_dict = {}
        self.T_sub_aruco_dicts = []
        self.average_distance_to_markers = -1.0


    def init_calibration(self) -> bool:
        """Initializes both intrinsics and extrinsics.

        Returns True if initialization was a success.

        Initializing intrinsics will 'never' fail.
        Initializing extrinsics fails if markers are not detected, visible or markers are occluded.

        :return: success: bool
        """

        print("Initializing intrinsics..")
        self.init_intrinsics()

        print("Initializing extrinsics..")
        init_attempts = 1
        success = self.init_extrinsics(self.scan.captures_list)
        while not success:
            print(f"    attempt nr {init_attempts} failed, retrying..")
            if init_attempts == 10:
                break

            self.scan.capture()
            success = self.init_extrinsics(self.scan.captures_list)
            init_attempts += 1

        return success

    def init_intrinsics(self):
        """Initializes intrinsics.

        Iterates every device and retrieves the camera matrix and distortion coefficients.
        Stored as a tuple in 'intrinsics' class variable.

        :return:
        """

        for device_id in range(self.num_devices):
            camera_matrix, distortion_coeffs = self.scan.get_intrinsics(device_id)
            self.intrinsics[device_id] = (camera_matrix, distortion_coeffs)

    def set_intrinsics(self, device_id: int, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray):
        """Set intrinsics manually for a specific device with device id.

        Can be used when manually calibrating intrinsics. Checks if device id is valid.

        Remarks:
            - Manual calibration is not necessary.
            - Manually calibrating intrinsics has been tested, and was found to be virtually identical to the
              intrinsics retrieved from each device via the SDK wrapper.

        :param device_id: int
        :param camera_matrix: np.ndarray
        :param distortion_coeffs: np.ndarray
        :return:
        """

        if device_id in range(0, self.num_devices):
            self.intrinsics[device_id] = (camera_matrix, distortion_coeffs)
        else:
            print("Failed to set intrinsics: device id is not valid")

    def init_extrinsics(self, captures_list: list) -> bool:
        """Initialize extriniscs by detecting markers, estimating pose transformation between device and marker,
        and calculate the transformation between subordinate devices and master device using common markers.

        Clips the captured IR images because potential reflections causes marker detection failure

        Returns False if captures_list is None or if master device only detects fewer than 2 markers.
        Detecting minimum 2 markers is necessary for transforming every point cloud the master's coordinate system.

        If draw_detected_markers=True is set, detected marker coordinate axis will be drawn and displayed.

        :param captures_list: list
        :return:
        """

        if captures_list is None:
            return False

        # clipping the ir image is necessary because of reflections causing marker detection failure
        ir_images = [np.clip(capture.ir.data, a_min=0, a_max=1500) for capture in captures_list]

        # detect aruco markers for master device
        master_id = self.num_devices - 1
        corners, ids, rejected, img = self.detect_markers(master_id, ir_images[master_id])

        if ids is None or len(corners) < 2:
            return False

        # get intrinsics for master
        camera_matrix, distortion_coeffs = self.intrinsics[master_id]

        # get transformation matrices from master to aruco markers
        detected_image = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        for aruco_id, corner in zip(ids, corners):
            rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corner, 0.1, camera_matrix, distortion_coeffs)

            self.T_master_aruco_dict[str(aruco_id[0])] = self._convert_rvec_tvec_to_transformation_matrix(rvec, tvec)

            if self.draw_detected_markers:
                cv.aruco.drawDetectedMarkers(detected_image, corners)
                cv.drawFrameAxes(detected_image, camera_matrix, distortion_coeffs,
                                 self.T_master_aruco_dict[str(aruco_id[0])][:3, :3],
                                 self.T_master_aruco_dict[str(aruco_id[0])][:3, 3],
                                 0.1, thickness=1)

                cv.imshow(f"Detected ArUco: capture {master_id}", detected_image)

        # detect aruco markers for subordinate devices
        for device_id, ir_image in enumerate(ir_images[:-1]):

            corners, ids, rejected, img = self.detect_markers(device_id, ir_image)

            if ids is None:
                return False

            # get intrinsics for master
            camera_matrix, distortion_coeffs = self.intrinsics[device_id]

            # get transformation from subordinate device to aruco markers
            detected_image = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            T_sub_aruco_dict = {}
            for aruco_id, corner in zip(ids, corners):
                rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corner, 0.1, camera_matrix, distortion_coeffs)

                T_sub_aruco_dict[str(aruco_id[0])] = self._convert_rvec_tvec_to_transformation_matrix(rvec, tvec)

                if self.draw_detected_markers:
                    cv.aruco.drawDetectedMarkers(detected_image, corners)
                    cv.drawFrameAxes(detected_image, camera_matrix, distortion_coeffs,
                                     T_sub_aruco_dict[str(aruco_id[0])][:3, :3],
                                     T_sub_aruco_dict[str(aruco_id[0])][:3, 3],
                                     0.1, thickness=1)

                    cv.imshow(f"Detected ArUco: capture {device_id}", detected_image)

            self.T_sub_aruco_dicts.append(T_sub_aruco_dict)

            # get transformation from subordinate to master using transformation
            # to common aruco marker
            for aruco_id, T_master_aruco in self.T_master_aruco_dict.items():
                if aruco_id in T_sub_aruco_dict.keys():
                    T_sub_aruco = T_sub_aruco_dict[aruco_id]
                    T_aruco_master = np.linalg.inv(T_master_aruco)
                    T_sub_master = T_sub_aruco @ T_aruco_master

                    self.extrinsics[device_id] = T_sub_master

        cv.waitKey(0)
        cv.destroyAllWindows()

        #self._optimize_extrinsics()
        return True

    def detect_markers(self, device_id: int, ir_image: np.ndarray) -> tuple:
        """Detects marker/s from IR image.

        IR image is normalized from 16-bit to 8-bit. This is required to use certain openCV functions.

        Dictionary parameters have been tuned somewhat to get more accurate transformations.

        :param device_id: int
        :param ir_image: np.ndarray
        :return: tuple
        """

        # normalize 16-bit image values to 8-bit image values,
        # necessary for use of certain openCV functions
        image = cv.normalize(ir_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        # set aruco dict and parameters, values found to give the best results when doing pose estimation
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        aruco_params = cv.aruco.DetectorParameters_create()
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 15
        aruco_params.adaptiveThreshWinSizeStep = 1
        aruco_params.adaptiveThreshConstant = 2
        aruco_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR

        # detect markers
        corners, ids, rejected = cv.aruco.detectMarkers(image,
                                                        aruco_dict,
                                                        parameters=aruco_params)

        if self.verbose:
            print(f"{len(corners)} aruco markers detected: capture {device_id}")

        print(f"    {len(corners)} aruco markers detected: capture {device_id}")

        return corners, ids, rejected, image

    def _convert_rvec_tvec_to_transformation_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Internal function used to convert rvec and tvec from openCV to a 4x4 transformation matrix.

        :param rvec: np.ndarray
        :param tvec: np.ndarray
        :return: transformation_matrix: np.ndarray
        """

        transformation_matrix = np.zeros((4, 4))
        rotation_matrix = np.array(cv.Rodrigues(rvec)[0])

        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3] = tvec.ravel()
        transformation_matrix[3, 3] = 1

        # calculates average distance to markers, used later when thresholding point clouds
        distance = np.linalg.norm(transformation_matrix[0:3, 3])
        if self.average_distance_to_markers < 0.0:
            self.average_distance_to_markers = distance
        else:
            self.average_distance_to_markers = (self.average_distance_to_markers + distance) / 2

        return transformation_matrix

    """
    def _optimize_extrinsics(self):

        graph = gtsam.NonlinearFactorGraph()

        PRIOR_MODEL = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]))
        graph.add(gtsam.PriorFactorPose3(X(self.num_devices-1), gtsam.Pose3(), PRIOR_MODEL))

        initial_estimate = gtsam.Values()
        initial_estimate.insert(X(self.num_devices-1), gtsam.Pose3())

        NOISE_MODEL = gtsam.noiseModel.Diagonal.Sigmas(np.array([5e-1, 5e-1, 5e-1, 1e-1, 1e-1, 1e-1]))
        Between = gtsam.BetweenFactorPose3

        # add transformations between marker and master device
        for aruco_id, T in self.T_master_aruco_dict.items():
            graph.add(Between(L(int(aruco_id)), X(self.num_devices-1),
                              gtsam.Pose3(T), NOISE_MODEL))

            initial_estimate.insert(L(int(aruco_id)), gtsam.Pose3(T).inverse())

        # add transformations between marker and subordinate devices
        for device_id, T_sub_aruco_dict in enumerate(self.T_sub_aruco_dicts):
            for aruco_id, T in T_sub_aruco_dict.items():
                graph.add(Between(L(int(aruco_id)), X(device_id),
                                  gtsam.Pose3(T), NOISE_MODEL))

                initial_estimate.update(L(int(aruco_id)), gtsam.Pose3(T).inverse())

        # add transformation between devices
        for device_id, T in enumerate(self.extrinsics):
            graph.add(Between(X(self.num_devices-1), X(device_id), gtsam.Pose3(T), NOISE_MODEL))
            initial_estimate.insert(X(device_id), gtsam.Pose3(T))

        graph.print("\nFactor Graph:\n")

        print("INITIAL_ESTIMATE:")
        print(initial_estimate)

        #marginals = gtsam.Marginals(graph, initial_estimate)
        self._plot_poses(initial_estimate)

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(5000)
        params.setRelativeErrorTol(1e-9)
        params.setAbsoluteErrorTol(1e-9)

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()

        print("RESULT:")
        print(result)

        print("ERROR INITIAL:")
        print(graph.error(initial_estimate))
        print("ERROR RESULT:")
        print(graph.error(result))

        self._plot_poses(result)

        opt_extrinsics = [None] * len(self.extrinsics)
        for i in range(len(self.extrinsics)):
            opt_extrinsics[i] = result.atPose3(X(i)).matrix()
        print(self.extrinsics[0] - opt_extrinsics[0])

        self.extrinsics = opt_extrinsics

    def _plot_poses(self, result):
        fig = plt.figure(0, figsize=(10,15))
        axes = fig.gca(projection="3d")
        plt.cla()

        pose_ids = [i for i in range(self.num_devices)]
        pose_ids.append(10)
        pose_ids.append(8)

        for pose_id in pose_ids:
            if pose_id < self.num_devices-1:
                gplt.plot_pose3(0, result.atPose3(X(pose_id)), 0.1)
                self.plot_text(axes, result.atPose3(X(pose_id)), f"x{pose_id}")
            elif pose_id == self.num_devices-1:
                gplt.plot_pose3(0, result.atPose3(X(pose_id)), 0.1)
                self.plot_text(axes, result.atPose3(X(pose_id)), f"x{pose_id}")
            elif pose_id > self.num_devices-1:
                try:
                    pose_landmark = result.atPose3(L(pose_id))
                    gplt.plot_point3(0, gtsam.Point3(pose_landmark.x(),
                                                     pose_landmark.y(),
                                                     pose_landmark.z()), "rx")
                    self.plot_text(axes, result.atPose3(L(pose_id)), f"l{pose_id}")
                except RuntimeError:
                    pass

        gplt.set_axes_equal(0)
        plt.show()

    def plot_text(self, axes, pose, text):
        t = pose.translation()

        axes.text(t[0], t[1], t[2], text)
    """
