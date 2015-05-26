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
import csv
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

INVERT          = False
BLUR            = True
BLUR_SIZE       = 9
MAX_SIZE        = 10000#7000
MIN_SIZE        = 500#4500
DETECTOR        = 'SOBEL'
FILE_NAME			  = 'data/debug'
USE_BACKGROUND  = True
BACKGROUND_FILE = 'data/background.png'

class Image_Processor:
		def __init__(self, csvfile):
			self.image_source = "/camera/image_mono"
			self.cvbridge = CvBridge()
			self.counter = 0
			self.background_taken = False
			self.waiting = False

			# Raw Image Subscriber
			self.image_sub = rospy.Subscriber(self.image_source,Image,self.image_callback)
			
			#Open csv file to export data
			
			self.datawrite = csv.writer(csvfile)
			self.datawrite.writerow(['Frame Number', 'Centroid X-coord', 'Centroid Y-coord', 'Contour Area'])
			

		def image_callback(self, rosimg):
			# Convert the image.
			try:
				# might need to change to bgr for color cameras
				img = self.cvbridge.imgmsg_to_cv2(rosimg, 'passthrough')
			except CvBridgeError, e:
				rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
				img = np.zeros((320,240))

			# Before capturing jellyfish, need to take a background snapshot.
			# Wait until the user presses a key, then take that as the 
			# background snapshot 
			if USE_BACKGROUND and not self.background_taken:
				# Prompt for the user
				display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
				cv2.putText(display_img, "Press any key to take a background snapshot",
									 (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), thickness=1)
				cv2.imshow("Background...", display_img)
				pressed_key = cv2.waitKey(1)
				if pressed_key == -1:
					return
				self.background_img = img
				self.background_taken = True
				self.waiting = True
				cv2.imwrite(BACKGROUND_FILE, self.background_img)
				cv2.destroyWindow('Background...')
	
			if self.waiting:
				display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
				cv2.putText(display_img, "Press any key when ready",
									 (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), thickness=1)
				cv2.imshow("Waiting for user...", display_img)
				pressed_key = cv2.waitKey(1)
				if pressed_key == -1:
					return
				self.waiting = False
				cv2.destroyWindow('Waiting for user...')

			# MAIN IMAGE PROCESSING
			self.counter +=1
		
			#Subtract background
			if USE_BACKGROUND:
				img = cv2.absdiff(self.background_img, img)

			#Invert so we're looking for light objects
			if INVERT:				
				img = (255-img)

			#Threshold with Otsu thresholding
			if BLUR:
				img = cv2.GaussianBlur(img, (BLUR_SIZE, BLUR_SIZE), 0)			

			#Find and draw contours and contour centroids in thresholded image            
			if DETECTOR == 'THRESHOLDING':
				ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)		
				contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				jelly_contours = []
				img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)					
				for jelly in contours: 
					area = cv2.contourArea(jelly)
					hull = cv2.convexHull(jelly)
					hull_area = cv2.contourArea(hull)
					if hull_area > MAX_SIZE or hull_area < MIN_SIZE:
						continue
						pass
					jelly_contours.append(jelly)
					max_coordinates = np.amax(jelly, axis=0)
					min_coordinates = np.amin(jelly, axis=0)
					centroid = (max_coordinates + min_coordinates)/2
					print("Centroid coord: " + str(centroid))
					cv2.circle(img, (centroid[0,0], centroid[0,1]), 2, (0, 255, 0), -1)
				cv2.drawContours(img, jelly_contours, -1, (0,255,0	), 1)
			elif DETECTOR == 'SOBEL':
				img_sobel_y = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1)
				img_sobel_x = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0)
				img_sobel = np.absolute(img_sobel_y) + np.absolute(img_sobel_x)
				img = np.uint8(img_sobel)
				ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
				contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				jelly_contours = []		
				img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)		
				for jelly in contours: 
					area = cv2.contourArea(jelly)
					hull = cv2.convexHull(jelly)
					hull_area = cv2.contourArea(hull)
					if hull_area > MAX_SIZE or hull_area < MIN_SIZE:
						continue
						pass
					jelly_contours.append(jelly)
					max_coordinates = np.amax(jelly, axis=0)
					min_coordinates = np.amin(jelly, axis=0)
					centroid = (max_coordinates + min_coordinates)/2
					print("Centroid coord: " + str(centroid))
					self.datawrite.writerow([self.counter, centroid[0][0], centroid[0][1], hull_area])
					cv2.circle(img, (centroid[0,0], centroid[0,1]), 2, (0, 255, 0), -1)
				cv2.drawContours(img, jelly_contours, -1, (0,255,0	), 1)
			elif DETECTOR == 'NONE':
				pass

			# Add a timestamp (frame number)
			cv2.putText(img, str(self.counter), (10,50), cv2.FONT_HERSHEY_PLAIN,
									3, (0,0,255), thickness=3)

			#Display
			cv2.namedWindow('image', cv2.WINDOW_NORMAL)
			cv2.imshow('image', img)
			cv2.waitKey(1)
			#cv2.destroyAllWindows()



################################################################################
def main():
	FILE_NAME = sys.argv[1]

	csvfile = open(FILE_NAME + '.csv', 'wb')
	image_processor = Image_Processor(csvfile)
	try:
		# spin() simply keeps python from exiting until this node is stopped
		rospy.spin()     
	except KeyboardInterrupt:
		print "Shutting down"
		csvfile.close()
	cv.DestroyAllWindows()

################################################################################
if __name__ == '__main__':
	rospy.init_node('image_processor', anonymous=True)
	main()
