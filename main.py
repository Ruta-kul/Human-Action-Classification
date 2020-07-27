"""
authors : Sourabh Hanamsheth, Ruta Kulkarni
Code to get the skeleton data from the camera and display the recognized activity

"""

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import ctypes
import _ctypes
import pygame
import sys

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                  pygame.color.THECOLORS["blue"], 
                  pygame.color.THECOLORS["green"], 
                  pygame.color.THECOLORS["orange"], 
                  pygame.color.THECOLORS["purple"], 
                  pygame.color.THECOLORS["yellow"], 
                  pygame.color.THECOLORS["violet"]]



class Classifier(nn.Module):
    def __init__(self, ip, H1, H2, H3, H4, H5, H6, op):
        super().__init__()
        self.linear1 = nn.Linear(ip, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, H4)
        self.linear5 = nn.Linear(H4, H5)
        self.linear6 = nn.Linear(H5, H6)
        self.linear7 = nn.Linear(H6, op)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        return self.linear7(x)

class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()
        
        self.labels = {0:"Horizontal Arm Wave",
                       1:"High Arm Wave",
                       2:"Two Hand Wave",
                       3:"Catch Up",
                       4:"High Throw",
                       5:"Draw X",
                       6:"Draw Tick",
                       7:"Toss Paper",
                       8:"Forward Kick",
                       9:"Side Kick",
                       10:"Take Umbrella",
                       11:"Bend",
                       12:"Hand Clap",
                       13:"Walk",
                       14:"Phone Call",
                       15:"Drink",
                       16:"Sit Down",
                       17:"Stand Up"}
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Classifier(1800, 1224, 812, 550, 206, 128, 64, 18).to(self.device)
        
        self.model.load_state_dict(torch.load("./kinect_activity_4.pth"))
        self.model.eval()        
        

        self.H = np.load("homography_cam2depth.npy")
        self.activity_input = [0]*1800
 
        self._clock = pygame.time.Clock()

          self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect Activity Recognition")


        self._done = False

        self._clock = pygame.time.Clock()

        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)

        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)


        self._bodies = None
        
        
    def norm(self,data):
        if abs(min(data)) > max(data):
            max_val = abs(min(data))
            min_val = min(data)
        else:
            max_val = max(data)
            min_val = -max(data)
        norm_val = (2*data)/(max_val - min_val)
        return norm_val
        
    
    def normalize(self,array):        
        array[:,0] = self.norm(array[:,0])
        array[:,1] = self.norm(array[:,1])
        array[:,2] = self.norm(array[:,2])
        return array
            
        
    def homogeneous(self,matrix,axis=0):           
        if not axis:            
            one = np.ones((matrix.shape[axis],1))
        else:
            one = np.ones((1,matrix.shape[axis]))
        return np.array(np.append(matrix,one,axis=abs(1-axis)))

    def euclidian(self,matrix,dim="2d"):  
        if dim == "2d":
            matrix[:,0] = matrix[:,0]/matrix[:,2]
            matrix[:,1] = matrix[:,1]/matrix[:,2]
            return np.array([matrix[:,0], matrix[:,1]]).T
        elif dim == "3d":
            matrix[:,0] = matrix[:,0]/matrix[:,3]
            matrix[:,1] = matrix[:,1]/matrix[:,3]
            matrix[:,2] = matrix[:,2]/matrix[:,3]
            return np.array([matrix[:,0], matrix[:,1], matrix[:,2]]).T
    
     


    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;


        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return


        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: 
            pass

    def draw_body(self, joints, jointPoints, color):

        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    

        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);


        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);


        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);


        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()
        
    def getDepthCoord(self, name):
        try:
            x = (int(self.joint_points_depth[name].x))
            y = (int(self.joint_points_depth[name].y))
            z = (int(self._depth[ y * 512 + x ] )) 

            return [x,y,z]
        except:
            return [0,0,0]
        
    
    def cameraSpace(self,joints):
        self.joint_points_depth = self._kinect.body_joints_to_depth_space(joints)
        depth_point0 = self.getDepthCoord(PyKinectV2.JointType_Head)
        depth_point1 = self.getDepthCoord(PyKinectV2.JointType_Neck)        
        depth_point2 = self.getDepthCoord(PyKinectV2.JointType_ShoulderRight)
        depth_point3 = self.getDepthCoord(PyKinectV2.JointType_ElbowRight)
        depth_point4 = self.getDepthCoord(PyKinectV2.JointType_HandRight)
        depth_point5 = self.getDepthCoord(PyKinectV2.JointType_ShoulderLeft)
        depth_point6 = self.getDepthCoord(PyKinectV2.JointType_ElbowLeft)
        depth_point7 = self.getDepthCoord(PyKinectV2.JointType_HandLeft)
        depth_point8 = self.getDepthCoord(PyKinectV2.JointType_SpineMid)
        depth_point9 = self.getDepthCoord(PyKinectV2.JointType_HipRight)
        depth_point10 = self.getDepthCoord(PyKinectV2.JointType_KneeRight)
        depth_point11 = self.getDepthCoord(PyKinectV2.JointType_FootRight)
        depth_point12 = self.getDepthCoord(PyKinectV2.JointType_HipLeft)
        depth_point13 = self.getDepthCoord(PyKinectV2.JointType_KneeLeft)
        depth_point14 = self.getDepthCoord(PyKinectV2.JointType_FootLeft)
        
        body = np.array([depth_point0,depth_point1,depth_point2,depth_point3,depth_point4,depth_point5,depth_point6,depth_point7,depth_point8, depth_point9,depth_point10,depth_point11,depth_point12,depth_point13,depth_point14]).reshape(-1,3)
        body_H = self.homogeneous(body[:,0:2])
        body_cam_H = (np.matmul(np.linalg.inv(self.H),body_H.T)).T
        body_cam = self.euclidian(body_cam_H)

        self.body_cam = np.hstack([body_cam,body[:,2].reshape(-1,1)])
        self.body_cam = self.normalize(self.body_cam)




    def activityRecognition(self):
        body = self.body_cam.ravel().tolist()
        self.activity_input+=body       
        del self.activity_input[:15*3]
        
        self.activity = torch.tensor(self.activity_input).view(1,-1).to(self.device)
        
        output = self.model.forward(self.activity.float())
        _, self.pred = torch.max(output, 1)
        self.prediction = int(self.pred.data[0])
        label = self.labels[self.prediction]
        print(label)

   
        
        


    def run(self):

        while not self._done:

            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: 
                    self._done = True 

                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    

            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = None
            if self._kinect.has_new_depth_frame():
                self._depth = self._kinect.get_last_depth_frame()


            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()

            if self._bodies is not None: 
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints 
                    
                    self.cameraSpace(joints)
                    self.activityRecognition()



                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])

            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            pygame.display.update()

            pygame.display.flip()

            self._clock.tick(60)


        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();
