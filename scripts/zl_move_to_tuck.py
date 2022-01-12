#!/usr/bin/env python

# References
# ----------
# http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html


import sys
import copy
import time
import rospy
import moveit_commander
from movo_action_clients.gripper_action_client import GripperActionClient

from moveit_msgs.msg import PlaceLocation, MoveItErrorCodes
from moveit_python import MoveGroupInterface, PlanningSceneInterface

import geometry_msgs.msg


if __name__=="__main__":
    rospy.init_node('move_test',
                    anonymous=False)
    robot = moveit_commander.RobotCommander()
    moveit_commander.roscpp_initialize(sys.argv)

    scene = moveit_commander.PlanningSceneInterface()

    lgripper = GripperActionClient('left')
    rgripper = GripperActionClient('right')
    gripper_closed = 0.00
    gripper_open = 0.165
    # gripper_open = 0.07
    rgripper.command(gripper_open,block=False)
    print(rgripper.result())
    print('time',rospy.get_time())
    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print("============ Available Planning Groups:", robot.get_group_names())
    # time.sleep(10)
    # # rgripper.command(gripper_closed,block=False)
    
    rarm_group = moveit_commander.MoveGroupCommander("right_arm")
    rmove_group = MoveGroupInterface("right_arm", "base_link")
    # rarm_group.clear_pose_targets()
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x = -0.527
    pose_goal.orientation.y = 0.493
    pose_goal.orientation.z = 0.475
    pose_goal.orientation.w = 0.504
    pose_goal.position.x = 0.8
    pose_goal.position.y = -0.152
    pose_goal.position.z = 1.073
    rarm_group.set_pose_target(pose_goal)
    plan = rarm_group.go(wait=True)
    rarm_group.stop()
    rarm_group.clear_pose_targets()

    # rgripper.command(gripper_closed,block=False)