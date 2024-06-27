"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.inspire_gripper import inspire_gripper

# new 
def transform_point(point, transformation_matrix):
    """
    Apply a transformation matrix to a 3D point.
    
    Parameters:
    point (tuple): A point in 3D space (x, y, z).
    transformation_matrix (np.ndarray): A 4x4 transformation matrix.
    
    Returns:
    np.ndarray: Transformed point as a 1x3 array.
    """
    x, y, z = point
    point_homogeneous = np.array([x, y, z, 1])
    transformed_point_homogeneous = transformation_matrix @ point_homogeneous
    return transformed_point_homogeneous[:3]

#    default='data/demo_pusht_real', 
@click.command()
@click.option('--output', '-o', default='data/pour_water', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default='192.168.56.2', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1/frequency
    # 新加
    # transformation_matrix=np.array(
    #     [[1.,    0.,             0.,             0],
    #     [0.,    -0.70710678,    0.70710678,    0],
    #     [0.,    -0.70710678,     -0.70710678,    0],
    #     [0.,    0.,             0.,             1]])

    # inverse_transformation_matrix=np.array(
    #     [[1.,    0.           ,0.,           0],
    #     [0.,    -0.70710678  ,-0.70710678,   0],
    #     [0.,    -0.70710678  ,-0.70710678,  0],
    #     [0.,    0.          ,0.,            1]])
    
    # transformation_matrix=np.array(
    #     [[1.,    0.,             0.],
    #     [0.,    -0.70710678,    -0.70710678],
    #     [0.,    0.70710678,     -0.70710678]])

    transformation_matrix=np.array(
        [[1.,    0.           ,0.,           0],
        [0.,    -0.70710678  ,-0.70710678,   0],
        [0.,    0.70710678  ,-0.70710678,  0],
        [0.,    0.          ,0.,            1]])

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=False,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                # use_gripper=True # 是否使用家爪
                # ) as env :
            ) as env ,\
            inspire_gripper(
                shm_manager=shm_manager, 
                get_max_k=128,
                frequency=125, 
                launch_timeout=3,
                com_port='/dev/ttyUSB0',
                baudrate=115200
            ) as gripper:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']

            # 增加夹爪控制维度
            target_pose = np.append(target_pose, 1)
            
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            # gripper状态
            gripper_open = 1

            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency) # [0.0, 0.0, 0.0]
                # print("sm.is_button_pressed(0):",sm.is_button_pressed(0))
                # print("sm.is_button_pressed(1):",sm.is_button_pressed(1))
                
                # if not sm.is_button_pressed(0): # 面对标志 左边按钮 不按下是false
                #     # translation mode
                #     drot_xyz[:] = 0 # 不按左边按钮 姿态不变
                # else:
                #     dpos[:] = 0 # 按下位置不变
                # if not sm.is_button_pressed(1):
                #     # 2D translation mode
                #     dpos[2] = 0 # 平面运动

                # pour water
                drot_xyz[2] = 0
                drot_xyz[0] = 0
                if not sm.is_button_pressed(0): # 面对标志 左边按钮 不按下是false
                    # translation mode
                    drot_xyz[:]=0  # 不按左边按钮 姿态不变
                else:
                    dpos[:]=0

                # drot_xyz[0]=0
                

                # if not sm.is_button_pressed(0): # 面对标志 左边按钮 不按下是false
                #     # translation mode
                #     # drot_xyz[:]=0  # 不按左边按钮 姿态不变
                #     drot_xyz[1]=0
                # else:
                #     pass

                # if sm.is_button_pressed(1): # 面对标志 左边按钮 不按下是false
                #     # translation mode
                #     dpos[:]=0  # 不按左边按钮 姿态不变
                # else:
                #     pass


                dpos=transform_point(dpos, transformation_matrix)
                drot_xyz=transform_point(drot_xyz, transformation_matrix)
                # print("dpos:",dpos/ (env.max_pos_speed / frequency))
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                # print(dpos)
                # print("target_pose_1",target_pose)

                # target_pose_temp = transform_point(target_pose[:3], transformation_matrix)
                # target_pose_temp += dpos
                # target_pose_temp = transform_point(target_pose_temp, inverse_transformation_matrix)
                # target_pose[:3] = target_pose_temp
                # print("target_pose_2",target_pose)
                # 原始代码
                target_pose[:3] += dpos # [0.0, 0.0, 0.0]
                # 原始
                # target_pose[3:] = (drot * st.Rotation.from_rotvec(
                #     target_pose[3:])).as_rotvec()
                target_pose[3:6] = (drot * st.Rotation.from_rotvec(
                target_pose[3:6])).as_rotvec()
                # print(target_pose)

                # 夹爪控制
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='o'):
                        gripper_open = 1
                        print("打开夹爪")
                        gripper.moveMax_cmd(500) # 发送夹爪打开指令
                    elif key_stroke == KeyCode(char='p'):
                        gripper_open = 0
                        print("关闭夹爪")
                        gripper.moveMinHold_cmd(500, 100) # 发送夹爪关闭指令
                target_pose[6]=gripper_open

                # execute teleop command
                env.exec_actions(
                    actions=[target_pose], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()