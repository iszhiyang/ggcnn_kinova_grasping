#! /usr/bin/env python

from cv2 import FAST_FEATURE_DETECTOR_FAST_N
import rospy
from movo_msgs.msg import JacoCartesianVelocityCmd
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

rospy.init_node('kinova_velocity_control')
# moveit_commander.roscpp_initialize(sys.argv)
# scene = moveit_commander.PlanningSceneIntserface()
# GRIPPER = GripperActionClient('right')

MAX_VELO_X = 0.25
MAX_VELO_Y = 0.15
MAX_VELO_Z = 0.085
MAX_ROTATION = 2
CURRENT_VELOCITY = [0, 0, 0, 0, 0, 0]
CURRENT_FINGER_VELOCITY = [0, 0, 0]
# GOAL_Z = 0.0
COUNT = 0
VELO_COV = generate_cartesian_covariance(0)
V_WIDTH = 0
GRIP_WIDTH_MM = 70
CURR_DEPTH = 350  # Depth measured from camera.

SERVO = False
DEBUGMODEL = True
# Safe workspace
X_MIN = 0.53
Y_MIN = -0.48
# Z_MIN = 0.85
Z_MIN = 0.95
X_MAX = 0.86
Y_MAX = -0.02
Z_MAX = 1.15
DESK_DEPTH = 0.53
GP_BASE_X = 0
GP_BASE_Y = 0
GP_BASE_Z = 0
GP_BASE_Y_UP = 0
P_GRIPPER_X = 0
P_GRIPPER_Y = 0
P_GRIPPER_Z = 0
P_GRIPPER_THETA_Z = 0
OUT_OF_RANGE = False
OPENLOOP = False
MOVE_DOWN = False
MOVE_UP = False
GRIPPER_OPEN = False
GRIPPER_CLOSE = False
GRASP_DONE = False
FINGER_POS_DIFF = [0]
START_BUFFER = False
FINGER_POS = 0

GRIPPER_VEL_PUB = rospy.Publisher('/movo/right_gripper/vel_cmd', std_msgs.msg.Float32, queue_size=1)

class Averager():
    def __init__(self, inputs, time_steps):
        self.buffer = np.zeros((time_steps, inputs)) # zl: 3 * 4 array
        self.steps = time_steps
        self.curr = 0
        self.been_reset = True

    def update(self, v):
        if self.steps == 1:
            self.buffer = v
            return v
        self.buffer[self.curr, :] = v # zl: store inputs data in the dim 1
        self.curr += 1
        if self.been_reset:
            self.been_reset = False
            while self.curr != 0:
                self.update(v)
        if self.curr >= self.steps:
            self.curr = 0
        return self.buffer.mean(axis=0) # return average value of recent 3 frames

    def evaluate(self):
        if self.steps == 1:
            return self.buffer
        return self.buffer.mean(axis=0)

    def reset(self):
        self.buffer *= 0
        self.curr = 0
        self.been_reset = True


pose_averager = Averager(4, 3)



def command_callback(msg):
    global SERVO
    global CURR_Z
    global CURR_DEPTH
    global pose_averager
    # global GOAL_Z
    global GRIP_WIDTH_MM
    global VELO_COV
    global DEBUGMODEL
    global X_MIN
    global Y_MIN
    global Z_MIN
    global X_MAX
    global Y_MAX
    global Z_MAX
    CURR_DEPTH = msg.data[5]
    global CURR_X, CURR_Y, CURR_Z
    global GP_BASE_Y, GP_BASE_Y_UP
    global MOVE_DOWN, OPENLOOP
    global GRIPPER_OPEN
    global P_GRIPPER_X, P_GRIPPER_Y, P_GRIPPER_Z, P_GRIPPER_THETA_Z
    global GP_BASE_X, GP_BASE_Z
    global DESK_DEPTH

    if SERVO:
        d = list(msg.data)

        # PBVS Method.
        # print('real depth is',d[2])
        if d[2] > 0.10:  # Min effective range of the realsense.

            # Convert width in pixels to mm.
            # 0.07 is distance from end effector (CURR_Z) to camera.
            # 0.1 is approx degrees per pixel for the realsense.
            if d[2] > 0.15:
                GRIP_WIDTH_PX = msg.data[4] + 5 # offset: 5
                GRIP_WIDTH_MM = max(min(GRIP_WIDTH_PX,150),0) / 150 # ratio 0-1
            
            # Construct the Pose in the frame of the camera.
            gp = geometry_msgs.msg.Pose()
            gp.position.x = d[0]
            gp.position.y = d[1]
            gp.position.z = d[2]
            # print(d[3])
            # q = tft.quaternion_from_euler(0, 0, -1 * d[3])
            q = tft.quaternion_from_euler(-1 * d[3], 0, 0)
            gp.orientation.x = q[0]
            gp.orientation.y = q[1]
            gp.orientation.z = q[2]
            gp.orientation.w = q[3]

            # Calculate Pose of Grasp in Robot Base Link Frame
            # Average over a few predicted poses to help combat noise.
            gp_base = convert_pose(gp, 'realsense_depth_optical_frame', 'right_base_link') #  transfer grasping point from camera frame to base link frame
            gpbo = gp_base.orientation
            e = tft.euler_from_quaternion([gpbo.x, gpbo.y, gpbo.z, gpbo.w])
            # Only really care about rotation about z (e[2]).
            av = pose_averager.update(np.array([gp_base.position.x, gp_base.position.y, gp_base.position.z, e[2]]))

        else:
            gp_base = geometry_msgs.msg.Pose()
            av = pose_averager.evaluate()

        # Average pose in base frame.
        gp_base.position.x = av[0]
        gp_base.position.y = av[1]
        gp_base.position.z = av[2]
        # GOAL_Z = av[2]
        ang = av[3] - np.pi/2  # We don't want to align, we want to grip.
        # print('Ang is: ', [d[3],e[2] - np.pi/2 ,ang])
        ###############################################
        q = tft.quaternion_from_euler(np.pi, 0, ang) # zl: Is RPY the same as ours?
        ###############################################
        gp_base.orientation.x = q[0]
        gp_base.orientation.y = q[1]
        gp_base.orientation.z = q[2]
        gp_base.orientation.w = q[3]

        # Get the Position of the End Effector in the frame fo the Robot base Link
        g_pose = geometry_msgs.msg.Pose()
        g_pose.position.z = 0.03  # Offset from the end_effector frame to the actual position of the fingers.
        # zl: there is no offset in x or y direction?
        g_pose.orientation.w = 1
        p_gripper = convert_pose(g_pose, 'right_ee_link', 'right_base_link')
        p_gripper_o=p_gripper.orientation
        p_gripper_o_euler = tft.euler_from_quaternion([p_gripper_o.x, p_gripper_o.y, p_gripper_o.z, p_gripper_o.w])
        P_GRIPPER_X = p_gripper.position.x
        P_GRIPPER_Y = p_gripper.position.y
        P_GRIPPER_Z = p_gripper.position.z
        P_GRIPPER_THETA_Z = p_gripper_o_euler[0]
        # print('Current Theta_z is:',P_GRIPPER_THETA_Z)
        publish_pose_as_transform(gp_base, 'right_base_link', 'G', 0.0) #zl: only for visualization?
        
        # Calculate Position Error.
        # gp_base: grasp point in base_link frame
        # p_gripper: fingier tip in base_link frame
        # Position Error is calculated in base_link frame
        dx = (gp_base.position.x - p_gripper.position.x)
        dy = (gp_base.position.y - p_gripper.position.y)
        dz = (gp_base.position.z - p_gripper.position.z)
        # print('gp base y, gripper y',[gp_base.position.y,p_gripper.position.y])
        # Orientation velocity control is done in the frame of the gripper,
        #  so figure out the rotation offset in the end effector frame.
        gp_gripper = convert_pose(gp_base, 'right_base_link', 'right_ee_link')
        pgo = gp_gripper.orientation
        q1 = [pgo.x, pgo.y, pgo.z, pgo.w]
        e = tft.euler_from_quaternion(q1)
        dr = 1 * e[0]
        dp = 1 * e[1]
        dyaw = 1 * e[2]
        # print('grasp in ee link:',[dr,dp,dyaw])

        # vx = max(min(dx * 2.5, MAX_VELO_X), -1.0*MAX_VELO_X)
        # vy = max(min(dy * 2.5, MAX_VELO_Y), -1.0*MAX_VELO_Y)
        # vz = max(min(dz - 0.04, MAX_VELO_Z), -1.0*MAX_VELO_Z)

        vx = max(min(dx * 3, MAX_VELO_X), -1.0*MAX_VELO_X)
        vy = max(min(dy * 0.4, MAX_VELO_Y), -1.0*MAX_VELO_Y)
        vz = max(min(dz * 3, MAX_VELO_Z), -1.0*MAX_VELO_Z)
        # Apply a nonlinearity to the velocity
        v = np.array([vx, vy, vz])
        # vc = np.dot(v, VELO_COV)
        vc = v
        
        if OPENLOOP == False:
            print('Stored Y:',GP_BASE_Y,' Current Y:',gp_base.position.y)
            GP_BASE_X = gp_base.position.x
            if gp_base.position.y >= DESK_DEPTH:
                GP_BASE_Y = DESK_DEPTH
            else:
                GP_BASE_Y = gp_base.position.y
            GP_BASE_Z = gp_base.position.z
            # print(GP_BASE_Y)
            GP_BASE_Y_UP = GP_BASE_Y - 0.25
            CURRENT_VELOCITY[0] = vc[0] 
            CURRENT_VELOCITY[1] = vc[1]
            CURRENT_VELOCITY[2] = vc[2]
            CURRENT_VELOCITY[3] = 0
            CURRENT_VELOCITY[4] = 0
            CURRENT_VELOCITY[5] = max(min(1.5 * dyaw, MAX_ROTATION), -1 * MAX_ROTATION)*5
            # CURRENT_VELOCITY[5] = max(min(1 * dr, MAX_ROTATION), -1 * MAX_ROTATION)*5
            GRIPPER_OPEN = True
            # print(GRIPPER_OPEN)
        else:
            pass
########################################################################

            # gripper_open()
            # dy = GP_BASE_Y - P_GRIPPER_Y
            # vy = max(min(dy, MAX_VELO_Y), -1.0*MAX_VELO_Y)
            # CURRENT_VELOCITY[0] = 0
            # CURRENT_VELOCITY[1] = vy *0.5
            # CURRENT_VELOCITY[2] = 0
            # CURRENT_VELOCITY[3] = 0
            # CURRENT_VELOCITY[4] = 0
            # CURRENT_VELOCITY[5] = 0
            # if abs(dy) <= 0.005:
            #     MOVE_DOWN = True
            #     gripper_close()
            # CURRENT_VELOCITY[0] = vc[0]
            # CURRENT_VELOCITY[1] = vc[1]
            # CURRENT_VELOCITY[2] = vc[2]        vy = max(min(dy, MAX_VELO_Y), -1.0*MAX_VELO_Y)
            # CURRENT_VELOCITY[3] = -1 * dp
            # CURRENT_VELOCITY[4] = 1 * dr
            # CURRENT_VELOCITY[5] = max(min(1 * dyaw, MAX_ROTATION), -1 * MAX_ROTATION)
        
        # print(CURRENT_VELOCITY)
def grasp_exec():
    global GRASP_DONE
    global MOVE_DOWN
    global GP_BASE_Y, GP_BASE_Y_UP, P_GRIPPER_Y
    global GRIPPER_OPEN
    global MOVE_UP
    global GP_BASE_X, GP_BASE_Z, P_GRIPPER_X, P_GRIPPER_Z

    if GRASP_DONE == False:
        # print(')
        print('going DOWN! Target depth:', GP_BASE_Y)
        # print('Grasp Done:',GRASP_DONE)
        # print('GP_BASE and P_GRIPPER_Y: ',[GP_BASE_Y,P_GRIPPER_Y])
        # print('GRIPPER OPEN IS:',GRIPPER_OPEN)
        # gripper_open()
        dx = GP_BASE_X - P_GRIPPER_X
        dy = GP_BASE_Y - P_GRIPPER_Y -0.055 # gripper to finger tip offset
        dz = GP_BASE_Z - P_GRIPPER_Z
        # print('Difference in open loop:',[dx, dy, dz])
        # dy = dy*0.6
        # print('new dy is:',dy)
        vx = max(min(dx*2 , MAX_VELO_X), -1.0*MAX_VELO_X)
        vy = max(min(dy*2, MAX_VELO_Y), -1.0*MAX_VELO_Y)
        vz = max(min(dz*2 , MAX_VELO_Z), -1.0*MAX_VELO_Z)
        CURRENT_VELOCITY[0] = vx
        CURRENT_VELOCITY[1] = vy 
        CURRENT_VELOCITY[2] = vz
        CURRENT_VELOCITY[3] = 0
        CURRENT_VELOCITY[4] = 0
        CURRENT_VELOCITY[5] = 0
        message = JacoCartesianVelocityCmd()
        message.x = CURRENT_VELOCITY[0]
        message.y = CURRENT_VELOCITY[1]
        message.z = CURRENT_VELOCITY[2]
        message.theta_x = CURRENT_VELOCITY[3]
        message.theta_y = CURRENT_VELOCITY[4]
        message.theta_z = CURRENT_VELOCITY[5]
        if abs(dy) <= 0.008:
            MOVE_DOWN = True # Reach target object successfully, gripper close
            gripper_close()
        else:
            velo_pub.publish(message) #Continue to move down
    else:
        print("going UP!")
        # print('GP_BASE_UP and P_GRIPPER_Y: ',[GP_BASE_Y_UP,P_GRIPPER_Y])
        dy = GP_BASE_Y_UP - P_GRIPPER_Y
        vy = max(min(dy, MAX_VELO_Y), -1.0*MAX_VELO_Y)
        CURRENT_VELOCITY[0] = 0
        CURRENT_VELOCITY[1] = vy
        CURRENT_VELOCITY[2] = 0
        CURRENT_VELOCITY[3] = 0
        CURRENT_VELOCITY[4] = 0
        CURRENT_VELOCITY[5] = 0
        message = JacoCartesianVelocityCmd()
        message.x = CURRENT_VELOCITY[0]
        message.y = CURRENT_VELOCITY[1]
        message.z = CURRENT_VELOCITY[2]
        message.theta_x = CURRENT_VELOCITY[3]
        message.theta_y = CURRENT_VELOCITY[4]
        message.theta_z = CURRENT_VELOCITY[5]
        if abs(dy) <=0.035:
            gripper_open()
            MOVE_UP = True # Lift up successfully, gripper open
        else:
            velo_pub.publish(message) #Continue to move down
            GRIPPER_OPEN = True
        velo_pub.publish(message)



def gripper_open():
    print('in gripper open')
    global GRIPPER_CLOSE
    global GRASP_DONE
    global V_WIDTH
    global START_BUFFER
    START_BUFFER = True
    global FINGER_POS_DIFF
    gripper_velo_pub_2 = rospy.Publisher('/movo/right_gripper/vel_cmd', std_msgs.msg.Float32, queue_size=1)
    gripper_vel = std_msgs.msg.Float32()
    stop_signal = False
    while not stop_signal:
        gripper_vel.data = -3000
        gripper_velo_pub_2.publish(gripper_vel)
        # print('FINGER_POS_DIFF',FINGER_POS_DIFF)
        if len(FINGER_POS_DIFF) >10:
            if sum(FINGER_POS_DIFF[:-6])/len(FINGER_POS_DIFF[:-6]) <0.01:
                stop_signal = True
        # rospy.sleep(0.01)
    print('After grasp open')
    FINGER_POS_DIFF = [0]
    START_BUFFER = False

def gripper_close():
    print('in gripper close')
    global GRIPPER_CLOSE
    global GRASP_DONE
    global V_WIDTH
    global START_BUFFER
    START_BUFFER = True
    # time.sleep(2)
    global FINGER_POS_DIFF
    gripper_velo_pub_2 = rospy.Publisher('/movo/right_gripper/vel_cmd', std_msgs.msg.Float32, queue_size=1)
    gripper_vel = std_msgs.msg.Float32()
    stop_signal = False
    while not stop_signal:
        gripper_vel.data = 3000
        gripper_velo_pub_2.publish(gripper_vel)
        # print('FINGER_POS_DIFF',FINGER_POS_DIFF)
        if len(FINGER_POS_DIFF) >10:
            if sum(FINGER_POS_DIFF[:-6])/len(FINGER_POS_DIFF[:-6]) <0.01:
                stop_signal = True
        # rospy.sleep(0.01)
    print('After grasp close')
    FINGER_POS_DIFF = [0]
    START_BUFFER = False
    GRASP_DONE = True



def gripper_callback(msg):
    global V_WIDTH
    global GRIP_WIDTH_MM
    global OPENLOOP
    global FINGER_POS
    global FINGER_POS_DIFF
    global START_BUFFER
    global COUNT
    global GRIPPER_VEL_PUB
    position = msg.position
    COUNT += 1
    # print('Finger pos',FINGER_POS)
    # print(position)
    diff = abs(FINGER_POS - position[0]) # get difference between this frame and last frame
    # print
    if START_BUFFER == True:
        # print('here!')
        if COUNT % 20 ==0:
            FINGER_POS_DIFF.append(diff)
            # print('RECORD: ',FINGER_POS_DIFF)
    FINGER_POS = position[0]
    width = position[0]
    ratio = GRIP_WIDTH_MM
    goal_width = 0.85 - 0.75 * ratio
    d_width = goal_width - width
    # print('Goal width',goal_width,' Current width',width)
    if OPENLOOP == False:
        if abs(d_width) > 0.05:
            V_WIDTH = max(min(d_width * 1000,5000),-5000)
            gripper_vel = std_msgs.msg.Float32()
            gripper_vel.data = V_WIDTH
            # print('Speed is:',V_WIDTH)
            # print('command sent:',gripper_vel.data)
            GRIPPER_VEL_PUB.publish(gripper_vel)
        else:
            V_WIDTH = 0
        # print('V_WIDTH',V_WIDTH)
        # print(V_WIDTH)
    #0.1 open
    #0.85 close
    # vel negtive open, positive close

# def finger_position_callback(msg):
#     global SERVO
#     global CURRENT_FINGER_VELOCITY
#     global CURR_DEPTH
#     global CURR_Z        if debug == True:
#     global GRIP_WIDTH_MM

#     # Only move the fingers when we're 200mm from the table and servoing.
#     if CURR_Z < 0.200 and CURR_DEPTH > 80 and SERVO:
#         # 4000 ~= 70mm
#         g = min((1 - (min(GRIP_WIDTH_MM, 70)/70)) * (6800-4000) + 4000, 5500)

#         # Move fast from fully open.
#         gain = 2
#         if CURR_Z > 0.12:
#             gain = 5

#         err = gain * (g - msg.finger1)
#         CURRENT_FINGER_VELOCITY = [err, err, 0]

#     else:
#         CURRENT_FINGER_VELOCITY = [0, 0, 0]

def reset():
    global OUT_OF_RANGE
    global OPENLOOP
    global MOVE_DOWN
    global MOVE_UP
    global GRIPPER_CLOSE
    global GRIPPER_OPEN
    global GRASP_DONE
    global MAX_ROTATION
    global P_GRIPPER_X, P_GRIPPER_Y, P_GRIPPER_Z, P_GRIPPER_THETA_Z
    # print('In reset')
    # print('OPENLOOP IS:',OPENLOOP)
    # print('MOVE_UP IS:',MOVE_UP)
    # OUT_OF_RANGE = False
    # OPENLOOP = False
    MOVE_DOWN = False
    # MOVE_UP = False
    GRIPPER_OPEN = False
    GRIPPER_CLOSE = False
    GRASP_DONE = False
    default_x = 0.047
    default_y = 0.126
    default_z = 0.57
    default_theta_z = -1.577
    dx = default_x - P_GRIPPER_X
    dy = default_y - P_GRIPPER_Y
    dz = default_z - P_GRIPPER_Z
    dyaw = default_theta_z - P_GRIPPER_THETA_Z
    vx = max(min(dx * 2.5, MAX_VELO_X), -1.0*MAX_VELO_X)
    vy = max(min(dy * 2.5, MAX_VELO_Y), -1.0*MAX_VELO_Y)
    vz = max(min(dz, MAX_VELO_Z), -1.0*MAX_VELO_Z)
    CURRENT_VELOCITY[0] = vx 
    CURRENT_VELOCITY[1] = vy 
    CURRENT_VELOCITY[2] = vz
    CURRENT_VELOCITY[3] = 0
    CURRENT_VELOCITY[4] = 0
    CURRENT_VELOCITY[5] = max(min(1 * dyaw, MAX_ROTATION), -1 * MAX_ROTATION)*5
    # print('Current gripper pose',[P_GRIPPER_X,P_GRIPPER_Y,P_GRIPPER_Z,P_GRIPPER_THETA_Z])
    if abs(dx)<0.05 and abs(dy)<0.05 and abs(dz)<0.05 and abs(dyaw)<0.15:
        MOVE_UP = False
        OPENLOOP = False
        # raw_input("Back to initial status, new round started")
        # print("Back to initial status, new round started")




def position_callback(msg):
    global CURRENT_VELOCITY
    global X_MIN
    global Y_MIN
    global Z_MIN
    global X_MAX
    global Y_MAX
    global Z_MAX
    global OUT_OF_RANGE
    global OPENLOOP
    CURR_X = msg.translation.x
    CURR_Y = msg.translation.y
    CURR_Z = msg.translation.z
    if CURR_X <= X_MIN or CURR_X >= X_MAX or CURR_Y <= Y_MIN or CURR_Y >= Y_MAX or CURR_Z >= Z_MAX:
        OUT_OF_RANGE = True
        print("Out of work space",[CURR_X, CURR_Y, CURR_Z])
        CURRENT_VELOCITY[0] = 0
        CURRENT_VELOCITY[1] = 0
        CURRENT_VELOCITY[2] = 0
        CURRENT_VELOCITY[3] = 0
        CURRENT_VELOCITY[4] = 0
        CURRENT_VELOCITY[5] = 0
    if CURR_Z<= Z_MIN:
        OPENLOOP = True

    


if __name__ == '__main__':
    # single_open()
    command_sub = rospy.Subscriber('/ggcnn/out/command', std_msgs.msg.Float32MultiArray, command_callback, queue_size=1)
    position_sub = rospy.Subscriber('/right_ee_state', geometry_msgs.msg.Transform, position_callback, queue_size=1)
    gripper_position_sub = rospy.Subscriber('/movo/right_gripper/joint_states', sensor_msgs.msg.JointState, gripper_callback, queue_size=1)
    # wrench_sub = rospy.Publisher('/movo/right_arm/cartesianforce', JacoCartesianVelocityCmd, wrench_callback, queue_size=1)
    # Publish velocity at 100Hz.
    velo_pub = rospy.Publisher('/movo/right_arm/cartesian_vel_cmd', JacoCartesianVelocityCmd, queue_size=1)


    # Publish velocity at 100Hz.
    # gripper_velo_pub = rospy.Publisher('/movo/right_gripper/vel_cmd', std_msgs.msg.Float32, queue_size=1)
    # gripper_vel = std_msgs.msg.Float32()
    r = rospy.Rate(100)
    # r = rospy.Rate(50)
    SERVO = True
    
    while not rospy.is_shutdown():
        if SERVO:
            message = JacoCartesianVelocityCmd()
            if OUT_OF_RANGE == False:
                if OPENLOOP == False:
                    message.x = CURRENT_VELOCITY[0]
                    message.y = CURRENT_VELOCITY[1]
                    message.z = CURRENT_VELOCITY[2] 
                    message.theta_x = CURRENT_VELOCITY[3]
                    message.theta_y = CURRENT_VELOCITY[4]
                    message.theta_z = CURRENT_VELOCITY[5]
                    velo_pub.publish(message)
                    # gripper_vel.data = V_WIDTH
                    # print('Speed is:',V_WIDTH)
                    # print('command sent:',gripper_vel.data)
                    # gripper_velo_pub.publish(gripper_vel)
                    # print(CURRENT_VELOCITY)
                else:
                
                    if MOVE_UP == True:
                        # print("Grasp finished, press enter for a new round")
                        # raw_input("Grasp finished, press enter for a new round")
                        reset()
                        message.x = CURRENT_VELOCITY[0]
                        message.y = CURRENT_VELOCITY[1]
                        message.z = CURRENT_VELOCITY[2]
                        message.theta_x = CURRENT_VELOCITY[3]
                        message.theta_y = CURRENT_VELOCITY[4]
                        message.theta_z = CURRENT_VELOCITY[5]
                        velo_pub.publish(message)
                    else:
                        grasp_exec()
                        


                        
            else:
                message.x = 0
                message.y = 0
                message.z = 0
                message.theta_x = 0
                message.theta_y = 0
                message.theta_z = 0
                velo_pub.publish(message)
            # finger_pub.publish(kinova_msgs.msg.FingerPosition(*CURRENT_FINGER_VELOCITY))
        r.sleep()
