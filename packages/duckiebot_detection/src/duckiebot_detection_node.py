#!/usr/bin/env python
from copy import deepcopy
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import BoolStamped, VehicleCorners, ObstacleImageDetection, ObstacleImageDetectionList, Vector2D
from geometry_msgs.msg import Point32, Point
from mutex import mutex
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32
from ground_projection.srv import GetGroundCoord
import cv2
import numpy as np
import os
import rospkg
import rospy
import threading
import time
import yaml


class DuckiebotDetectionNode(object):

    def __init__(self):
        self.node_name = rospy.get_name()
        self.bridge = CvBridge()
        self.active = True

        self.publish_freq = self.setupParam("~publish_freq", 2.0)
        # self.blobdetector_min_area = self.setupParam('~blobdetector_min_area', 10)
        # self.blobdetector_min_dist_between_blobs = self.setupParam('~blobdetector_min_dist_between_blobs', 2)
        self.duckiebot_area_thresh = self.setupParam('~duckiebot_area_thresh', 1500)
        erosion_kernel_size = self.setupParam('~erosion_kernel_size', 5)
        dilation_kernel_size = self.setupParam('~dilation_kernel_size', 25)
        self.verbose = self.setupParam('~verbose', True)

        self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        self.publish_duration = rospy.Duration.from_sec(1.0/self.publish_freq)
        self.last_stamp = rospy.Time.now()

        self.low_range1 = np.array([0,140,100])
        self.high_range1 = np.array([3,255,255])
        self.low_range2 = np.array([165,140,100])
        self.high_range2 = np.array([180,255,255])
        # self.top_cutoff = rospy.get_param('~top_cutoff')
        self.detection_top_cutoff = 180

        self.lock = mutex()
        self.sub_image = rospy.Subscriber("~image", CompressedImage,
                                          self.processImage, buff_size=921600, 
                                          queue_size=1)
        self.sub_switch = rospy.Subscriber("~switch", BoolStamped,
                                           self.cbSwitch, queue_size=1)
        self.pub_detection = rospy.Publisher("~detection",
                                             BoolStamped, queue_size=1)
        # NOTE: Using message type "ObstacleImageDetectionList" instead of "VehicleCorners" as
        # "VehicleCorners" assumes only one vehicle in scene.
        self.pub_detection_boxes = rospy.Publisher("~detection_boxes",
                                           ObstacleImageDetectionList, queue_size=1)

        self.pub_detections_image = rospy.Publisher("~detections",
                                                       Image, queue_size=1)
        self.pub_detection_masks = rospy.Publisher("~detection_masks",
                                                       Image, queue_size=1)
        self.pub_time_elapsed = rospy.Publisher("~detection_time",
                                                Float32, queue_size=1)

        rospy.loginfo("[%s] Initialization completed" % (self.node_name))

    def setupParam(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)
        rospy.loginfo("[%s] %s = %s " % (self.node_name, param_name, value))
        return value

    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data
    
    def loginfo(self, s):
		rospy.loginfo('[%s] %s' % (self.node_name, s))

    def processImage(self, image_msg):
        if not self.active:
            return
        
        now = rospy.Time.now()
        
        duckiebot_detected_msg_out = BoolStamped()
        detected_duckiebots_msg_out = ObstacleImageDetectionList()

        try:
            image_cv = self.bridge.compressed_imgmsg_to_cv2(
                image_msg, "bgr8")
        except CvBridgeError as e:
            print e

        start = rospy.Time.now()
        hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        bw1 = cv2.inRange(hsv_image, self.low_range1, self.high_range1)
        bw2 = cv2.inRange(hsv_image, self.low_range2, self.high_range2)
        duckiebot_bw = cv2.bitwise_or(bw1, bw2)
        
        duckie_bw = cv2.erode(duckiebot_bw, self.erosion_kernel)
        duckiebot_bw = cv2.dilate(duckiebot_bw, self.dilation_kernel)

        _, contours, _ = cv2.findContours(duckiebot_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        detection = False
        duckiebot_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # self.loginfo('duckiebot area: %s' % str(area))
            if area > self.duckiebot_area_thresh:
                x,y,w,h = cv2.boundingRect(cnt)
                # duckiebot_boxes.append([x,y,w,h])
                y0 = np.maximum(y, self.detection_top_cutoff)
                y1 = np.maximum(y + h, self.detection_top_cutoff)
                h_fixed = y1 - y0
                y_fixed = y0
                if h_fixed != 0:
                    duckiebot_boxes.append([x,y_fixed,w,h_fixed])
                    detection = True
        
        # publish whether we've had any duckiebot detections
        duckiebot_detected_msg_out.data = detection
        self.pub_detection.publish(duckiebot_detected_msg_out)
        
        # publish the bounding boxes of detected duckiebots
        imheight, imwidth, _ = image_cv.shape
        detection_list = []
        for (x, y, w, h) in duckiebot_boxes:
            detection = ObstacleImageDetection()
            detection.bounding_box.x = x
            detection.bounding_box.y = y
            detection.bounding_box.w = w
            detection.bounding_box.h = h
            detection.type.type = 0
            detection_list.append(detection)
        detected_duckiebots_msg_out.header.stamp = rospy.Time.now()
        detected_duckiebots_msg_out.list = detection_list
        detected_duckiebots_msg_out.imwidth = imwidth
        detected_duckiebots_msg_out.imheight = imheight
        self.pub_detection_boxes.publish(detected_duckiebots_msg_out)

        # publish time it took to process the image
        elapsed_time = (rospy.Time.now() - start).to_sec()
        self.pub_time_elapsed.publish(elapsed_time)

        # publish visualization of bounding boxes and hsv mask
        if self.verbose:
            object_masks_msg_out = self.bridge.cv2_to_imgmsg(duckiebot_bw, "mono8")
            self.pub_detection_masks.publish(object_masks_msg_out)
            
            detections_bgr = deepcopy(image_cv)
            for (x, y, w, h) in duckiebot_boxes:
                cv2.rectangle(detections_bgr,(x,y),(x+w,y+h),(0,0,255),2)

            detections_msg_out = self.bridge.cv2_to_imgmsg(detections_bgr, "bgr8")
            self.pub_detections_image.publish(detections_msg_out)

    def getgroundcoord(self, x, y):
        rospy.wait_for_service('/default/ground_projection/get_ground_coordinate')
        try:
            getGroundCoord = rospy.ServiceProxy('/default/ground_projection/get_ground_coordinate', GetGroundCoord)
            vector = Vector2D()
            vector.x = x
            vector.y = y
            gp = getGroundCoord(vector)
            return gp
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def loginfo(self, s):
		rospy.loginfo('[%s] %s' % (self.node_name, s))


if __name__ == '__main__':
    rospy.init_node('duckiebot_detection', anonymous=False)
    duckiebot_detection_node = DuckiebotDetectionNode()
    rospy.spin()
