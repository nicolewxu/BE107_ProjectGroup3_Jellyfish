#!/usr/bin/env python
# image_processing.py is bare-bones subscriber, in Object Oriented form. 
# If something is publishing to /camera/image_mono, it receives that 
# published image and writes "image received". 
# To run, use roslaunch on camera.launch or <bagfile>.launch and then, 
# in another terminal, type "python image_processing.py"
# Used in Lab 5 of BE 107 at Caltech
# By Melissa Tanner, mmtanner@caltech.edu, April 2015

import rospy
import cv2, cv 
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Image_Processor:
		def __init__(self):
			self.image_source = "/camera/image_mono"
			self.cvbridge = CvBridge()
			self.counter = 0

			# Raw Image Subscriber
			self.image_sub = rospy.Subscriber(self.image_source,Image,self.image_callback)

		def image_callback(self, rosimg):
			#print "image recieved"
			self.counter +=1
			if self.counter%1 is 0:
				# Convert the image.
				try:
						 # might need to change to bgr for color cameras
						img = self.cvbridge.imgmsg_to_cv2(rosimg, 'passthrough')
				except CvBridgeError, e:
						rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
						img = np.zeros((320,240))

				#Invert so we're looking for light objects
				img = (255-img)

				#Threshold with Otsu thresholding
				blurred = cv2.GaussianBlur(img, (9,9), 0)			
				ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)		

				#Find and draw contours and contour centroids in thresholded image            
				contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				fly_contours = []				
				for fly in contours: 
					area = cv2.contourArea(fly)
					hull = cv2.convexHull(fly)
					hull_area = cv2.contourArea(hull)
					if hull_area > 4000 or hull_area < 2000:
						#continue
						pass
					fly_contours.append(fly)
					max_coordinates = np.amax(fly, axis=0)
					min_coordinates = np.amin(fly, axis=0)
					centroid = (max_coordinates + min_coordinates)/2
					print("Centroid coord: " + str(centroid))
					cv2.circle(img, (centroid[0,0], centroid[0,1]), 2, (0, 255, 0), -1)

				cv2.drawContours(img, fly_contours, -1, (0,255,0	), 1)
	
				#Display
				cv2.namedWindow('image', cv2.WINDOW_NORMAL)
				cv2.imshow('image', img)
				cv2.waitKey(100)
				#cv2.destroyAllWindows()



################################################################################
def main():
	image_processor = Image_Processor()
	try:
		# spin() simply keeps python from exiting until this node is stopped
		rospy.spin()     
	except KeyboardInterrupt:
		print "Shutting down"
	cv.DestroyAllWindows()

################################################################################
if __name__ == '__main__':
	rospy.init_node('image_processor', anonymous=True)
	main()
