#! /usr/bin/env python

from threading import ThreadError
from cv2 import FAST_FEATURE_DETECTOR_FAST_N
import rospy
from movo_msgs.msg import JacoCartesianVelocityCmd
import numpy as np
from threading import Thread
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
# import times
'''
safe work space of right gripper wrt. base_link frame
[x_min, x_max, y_min, y_max, z_min, z_max]
'''
safe_work_space = [0.53, 0.86, -0.48, 0.1, 0.95, 1.3]
desktop_depth = 0.525



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





class movo_ggcnn():

    def __init__(self):
        self.max_velo = [0.25, 0.25, 0.25] # for cartesian movement
        self.max_velo_rot = 20 # for rotation
        self.current_velocity = [0, 0, 0, 0, 0, 0]
        self.safe_work_space = safe_work_space
        self.desktop_depth = desktop_depth
        self.current_pos_base = [0, 0, 0] # gripper position in base_link frame 
        self.current_pos_right_base = [0, 0, 0, 0] # gripper position and theta in right_base_link frame
        self.gripper_width_goal = 1
        self.gp_goal = [0, 0, 0, 0] # infered grasping point in right_base_link
        self.pose_averager = Averager(4, 3) # Average function, smooth the control
        self.finger_pos = 0 # store finger pos, [0, 1], 0 open, 1 close
        self.gripper_count = 0
        self.finger_pos_diff = [0] # store finger pose difference between two step
        ## signal
        self.openloop = False
        self.out_of_range = False
        self.grasping_done = False
        self.object_lifted = False
        self.start_buffer = False
        self.gripper_velo_pub = rospy.Publisher('/movo/right_gripper/vel_cmd', std_msgs.msg.Float32, queue_size=1)
        self.velo_pub = rospy.Publisher('/movo/right_arm/cartesian_vel_cmd', JacoCartesianVelocityCmd, queue_size=1)
        self.thread1 = Thread(target = self.bringup_ee_state_sub)
        self.thread2 = Thread(target = self.bringup_finger_state_sub)
        self.thread3 = Thread(target = self.bringup_command_sub)
        self.thread1.setDaemon(True)
        self.thread2.setDaemon(True)
        self.thread3.setDaemon(True)
        # self.thread4 = Thread(target = self.bringup_vel_publish)
        self.thread1.start()
        self.thread2.start()
        self.thread3.start()
        # self.thread4.start()
        # self.vel_publish()


    def bringup_command_sub(self):
        # print('command ')
        command_sub = rospy.Subscriber('/ggcnn/out/command', std_msgs.msg.Float32MultiArray, self.command_callback, queue_size=1)
    
    def bringup_ee_state_sub(self):
        # print('ee-state')
        position_sub = rospy.Subscriber('/right_ee_state', geometry_msgs.msg.Transform, self.position_callback, queue_size=1)

    def bringup_finger_state_sub(self):
        # print('finger_state')
        gripper_position_sub = rospy.Subscriber('/movo/right_gripper/joint_states', sensor_msgs.msg.JointState, self.gripper_callback, queue_size=1)
    def bringup_vel_publish(self):
        self.vel_publish()

    def command_callback(self,msg):
        # print('in command callback')
        d = list(msg.data)
        if d[2] > 0.10:  # Min effective range of the realsense.

            # Convert width in pixels to mm.
            # 0.07 is distance from end effector (CURR_Z) to camera.
            # 0.1 is approx degrees per pixel for the realsense.
            if self.openloop == False:
                gripper_width = msg.data[4] + 5 # offset: 5
                self.gripper_width_goal = max(min(gripper_width,150),0) / 150 # ratio 0-1
                print('Infered width:',self.gripper_width_goal)
            
            # Construct the Pose in the frame of the camera.
            gp = geometry_msgs.msg.Pose()
            gp.position.x = d[0]
            gp.position.y = d[1]
            gp.position.z = d[2]
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
            av = self.pose_averager.update(np.array([gp_base.position.x, gp_base.position.y, gp_base.position.z, e[2]]))

        else:
            gp_base = geometry_msgs.msg.Pose()
            av = self.pose_averager.evaluate()

        # Average pose in base frame.
        gp_base.position.x = av[0]
        gp_base.position.y = av[1]
        gp_base.position.z = av[2]

        ang = av[3] - np.pi/2  # We don't want to align, we want to grip.

        q = tft.quaternion_from_euler(np.pi, 0, ang) # zl: Is RPY the same as ours?
       
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
        self.current_pos_right_base[0] = p_gripper.position.x
        self.current_pos_right_base[1] = p_gripper.position.y
        self.current_pos_right_base[2] = p_gripper.position.z
        self.current_pos_right_base[3] = p_gripper_o_euler[0]
        # print('Current Theta_z is:',P_GRIPPER_THETA_Z)
        publish_pose_as_transform(gp_base, 'right_base_link', 'G', 0.0) #zl: only for visualization?
        
        # Calculate Position Error.
        # gp_base: grasp point in base_link frame
        # p_gripper: fingier tip in base_link frame
        # Position Error is calculated in base_link frame
        dx = (gp_base.position.x - p_gripper.position.x)
        dy = (gp_base.position.y - p_gripper.position.y)
        dz = (gp_base.position.z - p_gripper.position.z)
        gp_gripper = convert_pose(gp_base, 'right_base_link', 'right_ee_link')
        pgo = gp_gripper.orientation
        q1 = [pgo.x, pgo.y, pgo.z, pgo.w]
        e = tft.euler_from_quaternion(q1)
        dr = 1 * e[0]
        dp = 1 * e[1]
        dyaw = 1 * e[2]
        vx = max(min(dx * 3, self.max_velo[0]), -1.0* self.max_velo[0])
        vy = max(min(dy * 0.2, self.max_velo[1]), -1.0* self.max_velo[1])
        vz = max(min(dz * 3, self.max_velo[2]), -1.0* self.max_velo[2])
        # Apply a nonlinearity to the velocity
        v = np.array([vx, vy, vz])
        # vc = np.dot(v, VELO_COV)
        vc = v
        
        if self.openloop == False:
            print('Stored Y:',self.gp_goal[1],' Current Y:',gp_base.position.y)
            self.gp_goal[0] = gp_base.position.x
            if gp_base.position.y >= self.desktop_depth:
                self.gp_goal[1] = self.desktop_depth
            else:
                self.gp_goal[1] = gp_base.position.y
            self.gp_goal[2] = gp_base.position.z
            # print(GP_BASE_Y)
            self.gp_goal[3] = self.gp_goal[1] - 0.25
            self.current_velocity[0] = vc[0] 
            self.current_velocity[1] = vc[1]
            self.current_velocity[2] = vc[2]
            self.current_velocity[3] = 0
            self.current_velocity[4] = 0
            self.current_velocity[5] = max(min(1.5 * dyaw, self.max_velo_rot), -1 * self.max_velo_rot)*5
            # CURRENT_VELOCITY[5] = max(min(1 * dr, MAX_ROTATION), -1 * MAX_ROTATION)*5

        else:
            pass

    def gripper_callback(self,msg):
        # print('in gripper callback')
        '''
        pos: 0.1 open          0.85 close
        vel: negtive = open    positive = close
        '''
        position = msg.position
        self.gripper_count += 1
        # print('Finger pos',FINGER_POS)
        # print(position)
        diff = abs(self.finger_pos - position[0]) # get difference between this frame and last frame
        # print
        if self.start_buffer == True:
            # print('here!')
            if self.gripper_count % 20 ==0:
                self.finger_pos_diff.append(diff)
                # print('RECORD: ',FINGER_POS_DIFF)
        self.finger_pos = position[0]
        width = position[0]
        ratio = self.gripper_width_goal
        goal_width = 0.85 - 0.75 * ratio
        d_width = goal_width - width
        if self.openloop == False:
            if abs(d_width) > 0.05:
                v = max(min(d_width * 1000,5000),-5000)
                gripper_vel = std_msgs.msg.Float32()
                gripper_vel.data = v
                # print('Speed is:',V_WIDTH)
                # print('command sent:',gripper_vel.data)
                self.gripper_velo_pub.publish(gripper_vel)
            else:
                pass




    def position_callback(self,msg):
        # print('in position callback')
        '''
        Get position of right gripper in base_link frame.
        If the gripper is out of safe work space, the current velocity will become 0.
        If the gripper is below a threshold, closed-loop turns into open-loop
        '''
        x = msg.translation.x
        y = msg.translation.y
        z = msg.translation.z
        self.current_pos_base[0] = x
        self.current_pos_base[1] = y
        self.current_pos_base[2] = z
        if x <= self.safe_work_space[0] or x >= self.safe_work_space[1] or y <= self.safe_work_space[2] or y >= self.safe_work_space[3] or z >= self.safe_work_space[5]:
            self.out_of_range = True
            print("Out of work space")
            self.current_velocity[0] = 0
            self.current_velocity[1] = 0
            self.current_velocity[2] = 0
            self.current_velocity[3] = 0
            self.current_velocity[4] = 0
            self.current_velocity[5] = 0

        if z<= self.safe_work_space[4]:
            self.openloop = True

    def grasp_exec(self):
        if self.grasping_done == False:
            # print('going DOWN! Target depth:', self.gp_goal[1])
            dx = self.gp_goal[0] - self.current_pos_right_base[0]
            dy = self.gp_goal[1] - self.current_pos_right_base[1] -0.045 # gripper to finger tip offset previously -0.055
            dz = self.gp_goal[2] - self.current_pos_right_base[2]
    
            vx = max(min(dx*2 , self.max_velo[0]), -1.0*self.max_velo[0])
            vy = max(min(dy*3, self.max_velo[1]), -1.0*self.max_velo[1])
            vz = max(min(dz*2 , self.max_velo[2]), -1.0*self.max_velo[2])
            self.current_velocity[0] = vx 
            self.current_velocity[1] = vy
            self.current_velocity[2] = vz
            self.current_velocity[3] = 0
            self.current_velocity[4] = 0
            self.current_velocity[5] = 0
            message = JacoCartesianVelocityCmd()
            message.x = self.current_velocity[0]
            message.y = self.current_velocity[1]
            message.z = self.current_velocity[2]
            message.theta_x = self.current_velocity[3]
            message.theta_y = self.current_velocity[4]
            message.theta_z = self.current_velocity[5]
            if abs(dy) <= 0.01:
                self.gripper_control('close')
            else:
                self.velo_pub.publish(message) #Continue to move down
        else:
            print("going UP!")
            # print('GP_BASE_UP and P_GRIPPER_Y: ',[GP_BASE_Y_UP,P_GRIPPER_Y])
            dy = self.gp_goal[3] - self.current_pos_right_base[1]
            vy = max(min(dy, self.max_velo[1]), -1.0*self.max_velo[1])
            self.current_velocity = [0, vy, 0, 0, 0, 0]
            message = JacoCartesianVelocityCmd()
            message.x = 0
            message.y = vy
            message.z = 0
            message.theta_x = 0
            message.theta_y = 0
            message.theta_z = 0
            if abs(dy) <=0.035:
                self.gripper_control('open')
                 # Lift up successfully, gripper open
            else:
                self.velo_pub.publish(message) #Continue to move down
    
    def gripper_control(self,direction):
        self.start_buffer = True
        gripper_vel = std_msgs.msg.Float32()
        stop_signal = False

        if direction == 'open':
            gripper_vel.data = -3000
            print('in gripper open')
        elif direction == 'close':
            print('in gripper close')
            gripper_vel.data = 3000
        else:
            raise Exception("Wrong gripper setting")

        while not stop_signal:
            self.gripper_velo_pub.publish(gripper_vel)
            if len(self.finger_pos_diff) >10:
                if sum(self.finger_pos_diff[:-6])/len(self.finger_pos_diff[:-6]) <0.01:
                    stop_signal = True

        self.start_buffer = False
        self.finger_pos_diff = [0]
        if direction == 'close':
            self.grasping_done = True
        elif direction == 'open':
            self.object_lifted = True
            print(self.object_lifted)


    def vel_publish(self):
        r = rospy.Rate(100)
        message = JacoCartesianVelocityCmd()
        while not rospy.is_shutdown():
            # print('printing')
            if self.out_of_range == False:
                if self.openloop == False:
                    message.x = self.current_velocity[0]
                    message.y = self.current_velocity[1]
                    message.z = self.current_velocity[2]
                    message.theta_x = self.current_velocity[3]
                    message.theta_y = self.current_velocity[4]
                    message.theta_z = self.current_velocity[5]
                    self.velo_pub.publish(message)
                else:
                    if self.object_lifted == True:
                        self.reset()
                        message.x = self.current_velocity[0]
                        message.y = self.current_velocity[1]
                        message.z = self.current_velocity[2]
                        message.theta_x = self.current_velocity[3]
                        message.theta_y = self.current_velocity[4]
                        message.theta_z = self.current_velocity[5]
                        self.velo_pub.publish(message)
                    else:
                        self.grasp_exec()
            else:
                message.x = 0
                message.y = 0
                message.z = 0
                message.theta_x = 0
                message.theta_y = 0
                message.theta_z = 0
                self.velo_pub.publish(message)
            r.sleep()

    def reset(self):
        default_x = 0.047
        default_y = 0.07
        default_z = 0.57
        default_theta_z = -1.577
        dx = default_x - self.current_pos_right_base[0]
        dy = default_y - self.current_pos_right_base[1]
        dz = default_z - self.current_pos_right_base[2]
        dyaw = default_theta_z - self.current_pos_right_base[3]
        vx = max(min(dx * 2.5, self.max_velo[0]), -1.0* self.max_velo[0])
        vy = max(min(dy * 2.5, self.max_velo[1]), -1.0* self.max_velo[1])
        vz = max(min(dz, self.max_velo[2]), -1.0* self.max_velo[2])
        self.current_velocity[0] = vx
        self.current_velocity[1] = vy 
        self.current_velocity[2] = vz
        self.current_velocity[3] = 0
        self.current_velocity[4] = 0
        self.current_velocity[5] = max(min(1 * dyaw, self.max_velo_rot), -1 * self.max_velo_rot)*5
        if abs(dx)<0.05 and abs(dy)<0.05 and abs(dz)<0.05 and abs(dyaw)<0.15:
            self.object_lifted = False
            self.openloop = False
            self.grasping_done = False

if __name__=="__main__":
    rospy.init_node('ggcnn_movo_control')
    print('before ggcnn')
    ggcnn = movo_ggcnn()
    ggcnn.vel_publish()
    print('after ggcnn')
