# # from rtde_control import RTDEControlInterface
# # from rtde_receive import RTDEReceiveInterface


# # rtde_r = RTDEReceiveInterface(hostname="192.168.1.8")
# # rtde_c = RTDEControlInterface(hostname="192.168.1.8")
# # import numpy as np

# # import rtde_control
# # rtde_c = rtde_control.RTDEControlInterface("192.168.1.8")
# # rtde_c.moveL(np.array([172,-60,-138,-171,45,-138]) / 180 * np.pi, 0.5, 0.3)


# # from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
# # from multiprocessing.managers import SharedMemoryManager

# # shm_manager = SharedMemoryManager()
# # sm=Spacemouse(shm_manager=shm_manager)
# # while True:
# #     sm_state = sm.get_motion_state_transformed()
# #     print(sm_state)

# import numpy as np
# import scipy.spatial.transform as st

# def create_transformation_matrix(translation, rotation_angles):
#     """
#     Create a 4x4 transformation matrix.
    
#     Parameters:
#     translation (tuple): Translation along x, y, z axes (tx, ty, tz).
#     rotation_angles (tuple): Rotation angles around x, y, z axes in radians (rx, ry, rz).
    
#     Returns:
#     np.ndarray: A 4x4 transformation matrix.
#     """
#     tx, ty, tz = translation
#     rx, ry, rz = rotation_angles
    
#     # Rotation matrices around x, y, z axes
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(rx), -np.sin(rx)],
#         [0, np.sin(rx), np.cos(rx)]
#     ])
    
#     Ry = np.array([
#         [np.cos(ry), 0, np.sin(ry)],
#         [0, 1, 0],
#         [-np.sin(ry), 0, np.cos(ry)]
#     ])
    
#     Rz = np.array([
#         [np.cos(rz), -np.sin(rz)],
#         [np.sin(rz), np.cos(rz)],
#         [0, 0, 1]
#     ])
    
#     # Combined rotation matrix
#     # R = Rz @ Ry @ Rx
#     R = Rx
    
#     # Transformation matrix
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = [tx, ty, tz]
    
#     return T

# def transform_point(point, transformation_matrix):
#     """
#     Apply a transformation matrix to a 3D point.
    
#     Parameters:
#     point (tuple): A point in 3D space (x, y, z).
#     transformation_matrix (np.ndarray): A 4x4 transformation matrix.
    
#     Returns:
#     np.ndarray: Transformed point as a 1x3 array.
#     """
#     x, y, z = point
#     point_homogeneous = np.array([x, y, z, 1])
#     transformed_point_homogeneous = transformation_matrix @ point_homogeneous
#     return transformed_point_homogeneous[:3]

# def inverse_transformation_matrix(transformation_matrix):

#     # Compute the inverse of a 4x4 homogeneous transformation matrix.
#     # Parameters:
#     # transformation_matrix (np.ndarray): A 4x4 homogeneous transformation matrix.
#     # Returns:
#     # np.ndarray: The inverse 4x4 transformation matrix.
#     R = transformation_matrix[:3, :3]
#     t = transformation_matrix[:3, 3]
#     # Compute the inverse rotation matrix (transpose of the rotation matrix)
#     R_inv = R.T
#     # Compute the inverse translation
#     t_inv = -R_inv @ t
#     # Form the inverse transformation matrix
#     inverse_T = np.eye(4)
#     inverse_T[:3, :3] = R_inv
#     inverse_T[:3, 3] = t_inv
#     return inverse_T

# # Example usage
# # translation = (0, 0, 400)
# # translation = (0, 0, 400)
# # rotation_angles = (-3*np.pi/4, 0, 0)  # 45 degrees around each axis
# # point = (464.41, -338.76, 77.8)
# # # point = (500, 80, 0)

# # transformation_matrix = create_transformation_matrix(translation, rotation_angles)
# # print("transformation_matrix:", transformation_matrix)
# # transformed_point = transform_point(point, transformation_matrix)

# # print("Original point:", point)
# # print("Transformed point:", transformed_point)


# # inverse_transformation_matrix = inverse_transformation_matrix(transformation_matrix)
# # transformed_point_2 = transform_point(transformed_point, inverse_transformation_matrix)
# # print("Original point_2:", transformed_point_2)
# # # print(transformation_matrix)
# # print("inverse_transformation_matrix:", inverse_transformation_matrix)

# # # import numpy as np

# # # Example usage
# # translation = (0, 0, 400)
# # rotation_angle_x = np.pi / 4 # 45 degrees
# # # Create the original transformation matrix
# # def create_transformation_matrix(translation, rotation_angle_x):
# #     tx, ty, tz = translation
# #     Rx = np.array([
# #     [1, 0, 0],
# #     [0, np.cos(rotation_angle_x), -np.sin(rotation_angle_x)],
# #     [0, np.sin(rotation_angle_x), np.cos(rotation_angle_x)]
# #     ])
# #     T = np.eye(4)
# #     T[:3, :3] = Rx
# #     T[:3, 3] = [tx, ty, tz]
# #     return T
# # transformation_matrix = create_transformation_matrix(translation, rotation_angle_x)
# # # Compute the inverse transformation matrix
# # inverse_T = inverse_transformation_matrix(transformation_matrix)
# # print("Original transformation matrix:\n", transformation_matrix)
# # print("Inverse transformation matrix:\n", inverse_T)
# # # Verify the inverse (should result in identity matrix)
# # identity_check = transformation_matrix @ inverse_T
# # print("Identity check (original * inverse):\n", identity_check)

# # transformation_matrix=np.array(
# #     [[1.,    0.,             0.,             0.],
# #     [0.,    -0.70710678,    -0.70710678,    0.],
# #     [0.,    0.70710678,     -0.70710678,    400.],
# #     [0.,    0.,             0.,             1.]])

# # inverse_transformation_matrix=np.array(
# #     [[1.,    0.           ,0.,           0.              ],
# #     [0.,    -0.70710678  ,0.70710678,   -282.84271247   ],
# #     [0.,    -0.70710678  ,-0.70710678,  282.84271247    ],
# #     [0.,    0.          ,0.,            1.              ]])


# # point = (464.41, -338.76, 77.8)
# # transformed_point = transform_point(point, transformation_matrix)

# # print("Original point:", point)
# # print("Transformed point:", transformed_point)

# # transformed_point_2 = transform_point(transformed_point, inverse_transformation_matrix)
# # print("Original point_2:", transformed_point_2)

# # # rot_xyz=[3.14, 0., 0.]
# # # drot = st.Rotation.from_rotvec(rot_xyz)
# # # R = transformation_matrix[:3, :3].tolist()
# # # print((drot@R).as_rotvec())


# transformation_matrix_bound=np.array(
#     [[1.,    0.           ,0.,           0],
#     [0.,    -0.70710678  ,-0.70710678,   0],
#     [0.,    0.70710678  ,-0.70710678,  0],
#     [0.,    0.          ,0.,            1]])

# inverse_transformation_matrix_bound=np.array(
#     [[1.,    0.           ,0.,           0],
#     [0.,    -0.70710678  ,0.70710678,   0],
#     [0.,    -0.70710678  ,-0.70710678,  0],
#     [0.,    0.          ,0.,            1]])

# # inverse_transformation_matrix = inverse_transformation_matrix(transformation_matrix)
# # print("inverse_transformation_matrix:", inverse_transformation_matrix)
# point = (464.41, -338.76, 77.8)
# transformed_point = transform_point(point, transformation_matrix)
# print("Transformed point:", transformed_point)
# transformed_point = transform_point(transformed_point, inverse_transformation_matrix)
# print("Original point:", point)


# import time
# print("time.monotonic()", time.monotonic())
# print(" time.time()", time.time())

import zarr
import os

# 加载.zarr存储的数据
mode='r'
# zarr_group = zarr.load('/home/yuezk/yzk/diffusion_policy/data/demo_pusht_real/replay_buffer.zarr')
group = zarr.open(os.path.expanduser('/home/yuezk/yzk/diffusion_policy/data/demo_pusht_real/replay_buffer.zarr'), mode)
pass    
