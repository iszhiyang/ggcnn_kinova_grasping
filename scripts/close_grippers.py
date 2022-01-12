#!/usr/bin/env python

# Author: Isaac Ng

# References
# ----------
# http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html


import sys
import copy
import rospy
import moveit_commander
from movo_action_clients.gripper_action_client import GripperActionClient

from moveit_msgs.msg import PlaceLocation, MoveItErrorCodes
from moveit_python import MoveGroupInterface, PlanningSceneInterface

import geometry_msgs.msg

_upper_body_joints = ["right_shoulder_pan_joint",
                      "right_shoulder_lift_joint",
                      "right_arm_half_joint",
                      "right_elbow_joint",
                      "right_wrist_spherical_1_joint",
                      "right_wrist_spherical_2_joint",
                      "right_wrist_3_joint",
                      "left_shoulder_pan_joint",
                      "left_shoulder_lift_joint",
                      "left_arm_half_joint",
                      "left_elbow_joint",
                      "left_wrist_spherical_1_joint",
                      "left_wrist_spherical_2_joint",
                      "left_wrist_3_joint",
                      "linear_joint",
                      "pan_joint",
                      "tilt_joint"]
                      
# Head looking straight
default_pose_tucked = [-1.595, -1.5, 0.1, -2.612, 0.0, 0.496, -1.69,
                       1.595, 1.5, -0.1, 2.612, 0.0, -0.496, 1.69,
                       0.14, 0, 0]

# default_pose_tucked = [-1.595, -1.5, 0.40, -2.612, 0.0, 0.496, -1.69,
#                        1.595, 1.5, -0.4, 2.612, 0.0, -0.496, 1.69,
#                        0.14, 0, 0]
# default_pose_tucked = [-1.595, -1.5, 0.40, -2.612, 0.0, 0.496, -1.69,
#                                     1.595, 1.5, -0.4, 2.612, 0.0, -0.496, 1.69,
#                                     0.14, 0, -0.6]
if __name__=="__main__":
    rospy.init_node('zl_close_gripper',
                    anonymous=False)

    moveit_commander.roscpp_initialize(sys.argv)

    scene = moveit_commander.PlanningSceneInterface()

    lgripper = GripperActionClient('left')
    rgripper = GripperActionClient('right')
    gripper_closed = 0.00
    gripper_open = 0.165
    # gripper_open = 0.07
    #gripper_open = 0.08
    
    # larm_group = moveit_commander.MoveGroupCommander("left_arm")
    # rarm_group = moveit_commander.MoveGroupCommander("right_arm")
    # upper_body = moveit_commander.MoveGroupCommander("upper_body")

    # move_group = MoveGroupInterface("upper_body", "base_link")
    # lmove_group = MoveGroupInterface("left_arm", "base_link")
    # rmove_group = MoveGroupInterface("right_arm", "base_link")

    print("Done spinning up MoveIt!")
    # upper_body.go(joints=[])
    # larm_group.
    

    # while not rospy.is_shutdown():
    #     print("doing L0_goto_upper_body_joints")
    #     result = move_group.moveToJointPosition(_upper_body_joints, default_pose_tucked, 0.005, wait=True)


    #     print("error code: ", result.error_code.val)
    #     if result.error_code.val == MoveItErrorCodes.SUCCESS:
    #         break

    print("Waiting 3 seconds...")
    rospy.sleep(3)
    print("Done waiting!")
    #lgripper.command(gripper_open,block=False)
    #rgripper.command(gripper_open, block=False)
    #raw_input()
    #lgripper.command(gripper_closed,block=False)
    rgripper.command(gripper_closed, block=False)


    print("successful")
