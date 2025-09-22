import argparse
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from hardware.camera import RealSenseCamera, get_realsense_devices
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp

logging.basicConfig(level=logging.INFO)


# Franka robot
from franky import *
ROBOT_IP = "192.168.1.1"
# -----------------------------
# Eye-in-hand
# -----------------------------
R_cam2gripper_avg = np.array([
    [ 0.0250142, -0.99918834, -0.03157457],
    [ 0.99874154,  0.02635147, -0.04267234],
    [ 0.04346974, -0.03046742,  0.99859006]
])
t_cam2gripper_avg = np.array([[0.05895725], [-0.02991709], [-0.03509327]])

# EE 坐标系下的 Tool 偏移
TOOL_IN_EE = np.array([-0.010, -0.000, 0.105])
# Home
JOINT_POSITION_START = np.array([0. , -0.78539816,  0. , -2.35619449,  0. , 1.57079633,  0.78539816])
# Box
JOINT_POSITION_BOX = np.array([1.01604994, -0.39622053,  0.11822519, -2.34395197, 0.08965438, 2.01721451, 0.98380203])

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=5,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    device_serials = get_realsense_devices()
    logging.info("Selected device serial numbers: %s", device_serials[0])
    cam = RealSenseCamera(device_id=device_serials[0])
    cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)
    fx, fy = cam.intrinsics.fx, cam.intrinsics.fy
    cx, cy = cam.intrinsics.ppx, cam.intrinsics.ppy
    print("Camera Intrinsic: ", fx, fy, cx, cy)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network, weights_only=False)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    # Robot Config
    print("Init franka robot...")
    fr3_robot = Robot(ROBOT_IP)
    fr3_robot.relative_dynamics_factor = 0.05
    fr3_gripper = Gripper(ROBOT_IP)
    start_joint_pose = JointMotion(JOINT_POSITION_START)
    box_joint_pose = JointMotion(JOINT_POSITION_BOX)
    print("Move robot to start position...")
    fr3_robot.move(start_joint_pose)

    if fr3_gripper.width < 0.05:
        print("Open gripper...")
        fr3_gripper.open(speed=0.02)

    try:
        fig = plt.figure(figsize=(10, 10))
        while True:
           

            image_bundle = cam.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            depth[depth >1.2] = 0 # distance > 1.2m ,remove it
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
            
            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
            grasps = detect_grasps(q_img, ang_img, width_img)
           
            if len(grasps) > 0:
                plot_grasp(fig=fig, rgb_img=cam_data.get_rgb(rgb, False), grasps=grasps, save=False)
                fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.001)

                target_grasp = grasps[0]

                # Get grasp position from model output
                pos_z = depth[target_grasp.center[1] + cam_data.top_left[1],
                            target_grasp.center[0] + cam_data.top_left[0]]
                pos_x = np.multiply(target_grasp.center[1] + cam_data.top_left[1] - cx, pos_z / fx)
                pos_y = np.multiply(target_grasp.center[0] + cam_data.top_left[0] - cy, pos_z / fy)
                if pos_z == 0:
                    print("Invalid depth value, skip...")
                    continue
                point_cam = np.array([pos_x, pos_y, pos_z])
                print("Position in camera frame: ", point_cam)

                # Camera -> EE
                point_ee = R_cam2gripper_avg @ point_cam + t_cam2gripper_avg
                point_ee = point_ee.flatten()  # EE 坐标系下的目标点

                # 当前 EE 位姿（Base 下）
                cartesian_state = fr3_robot.current_cartesian_state
                ee_pose_base = cartesian_state.pose.end_effector_pose  # Affine

                # 目标点转到 Base
                point_ee_in_base = ee_pose_base * Affine(point_ee, np.array([0.0, 0.0, 0.0, 1.0]))
                point_ee_in_base_pos = point_ee_in_base.translation

                # Tool 偏移在 Base 下
                R_ee_base = ee_pose_base.matrix[:3, :3]
                tool_in_base = R_ee_base @ TOOL_IN_EE

                # EE 末端目标位置（考虑 tool）
                ee_target_in_base = point_ee_in_base_pos - tool_in_base

                # -----------------------------
                # 处理角度：图像 -> 相机 -> EE -> Base
                # -----------------------------
                # === 角度归一化到 (-pi, pi] ===
                pred_angle = (target_grasp.angle - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi

                # === 转到相机坐标系下 ===
                Rz_cam = R.from_euler("z", pred_angle).as_matrix()
                # === 相机 -> EE -> Base ===
                R_target_ee = R_cam2gripper_avg @ Rz_cam
                R_target_base = R_ee_base @ R_target_ee

                # === 保证 z 轴朝上，消除180度二义性 ===
                if R_target_base[2, 2] < 0:
                    R_target_base = R_target_base @ R.from_euler("z", 180, degrees=True).as_matrix()

                q_target_base = R.from_matrix(R_target_base).as_quat()

                # 构造 Base 下目标位姿
                target_pose_base = Affine(ee_target_in_base, q_target_base)

                # 移动
                motion = CartesianMotion(target_pose_base, ReferenceType.Absolute)
                fr3_robot.move(motion)

                time.sleep(1.0)  # 等待末端到位
                print(f"Arrived position: {point_ee}")

                print("Grasping ...........")
                success = fr3_gripper.grasp(0.0, speed=0.04, force=20, epsilon_outer=1.0)

                print("Grasping success, move to home...")
                fr3_robot.move(box_joint_pose)
                fr3_gripper.open(speed=0.02)
                time.sleep(1.0)
                fr3_robot.move(start_joint_pose)

    finally:
        print("Done...")
