from cv2 import FAST_FEATURE_DETECTOR_FAST_N
import rospy
# from movo_msgs.msg import JacoCartesianVelocityCmd
import numpy as np

# import kinova_msgs.srv
from rospy.client import DEBUG
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import tf.transformations as tft

# from helpers.transforms import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform
from helpers.transforms import publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform
from helpers.covariance import generate_cartesian_covariance

# from helpers.gripper_action_client import set_finger_positions
# from helpers.position_action_client import move_to_position

import sys
import copy
import time
# import moveit_commander
# from movo_action_clients.gripper_action_client import GripperActionClient

# from moveit_msgs.msg import PlaceLocation, MoveItErrorCodes
# from moveit_python import MoveGroupInterface, PlanningSceneInterface
V_WIDTH = 0
def gripper_callback(msg):
    global V_WIDTH
    position = msg.position
    width = position[0]
    ratio = 0.5
    goal_width = 0.85 - 0.75 * ratio
    d_width = goal_width - width
    print(goal_width)
    print(width)
    if d_width > 0.04:
        V_WIDTH = max(min(d_width * 2000,5000),-5000)
    else:
        V_WIDTH = 0
    print(V_WIDTH)

if __name__ == '__main__':
    rospy.init_node('gripper_velocity_control')
    # single_open()
    # command_sub = rospy.Subscriber('/ggcnn/out/command', std_msgs.msg.Float32MultiArray, command_callback, queue_size=1)
    position_sub = rospy.Subscriber('/movo/right_gripper/joint_states', sensor_msgs.msg.JointState, gripper_callback, queue_size=1)

    # Publish velocity at 100Hz.
    gripper_velo_pub = rospy.Publisher('/movo/right_gripper/vel_cmd', std_msgs.msg.Float32, queue_size=1)

    r = rospy.Rate(100)
        default_z = 0.67

    while not rospy.is_shutdown():
        gripper_vel.data = V_WIDTH
        gripper_velo_pub.publish(gripper_vel)
        r.sleep()
        # rospy.spin()




