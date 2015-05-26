import cv2 
import os
import numpy as np
import cPickle

def main():
	#move_pics()
	#make_avg_background()
	#find_contours_with_background()
	find_contours_with_adaptive_thresholding()
	find_contours_with_otsu_thresholding()
	#find_contours_vanilla()

def move_pics():
	pics = dict()
	for folder in ['fly_with_food_stills', 'larvae_stills']:
		pics[folder] = []
		img_folder = 'videos_for_tracking/' + folder
		for root, dirs, filenames in os.walk(img_folder):
			for f in filenames:
				filename = os.path.join('videos_for_tracking', folder, f)
				larva_img = cv2.imread(filename, 0)
				pics[folder].append(larva_img)
				output_filename = os.path.join('output', folder, f)
				cv2.imwrite(output_filename, larva_img)
	cPickle.dump(pics, open('output/all_imgs.pickle', 'wb'))
	return pics

def make_avg_background():                	                    
	pics = cPickle.load(open('output/all_imgs.pickle', 'r'))
	for scene in pics.keys():
    # initialize accumulator destination
		cumulative_img = np.float64(pics[scene][0])
		cumulative_img /= 255.0
		cv2.imshow('init cumulative', cumulative_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		for n in range(100):
			for i in range(len(pics[scene])):
				#pics[scene][i] = 255 - pics[scene][i]
				#cv2.imshow('img', pics[scene][i])
				#cv2.waitKey(0)
				#pics[scene][i] = cv2.cvtColor(pics[scene][i], cv2.COLOR_BGR2GRAY)
				cv2.accumulateWeighted(np.float64(pics[scene][i])/255.0, cumulative_img, .01)
				
			
		cv2.imshow('accumulating?', cumulative_img)
		cv2.waitKey(0)
		output_filename = os.path.join('output', scene, 'background.jpg')
		cv2.imwrite(output_filename, cumulative_img*255.0)
		
def find_centroids(centroid_filename, contours, img):
	with open(centroid_filename, 'w') as centroid_file:
		centroid_file.write("X\tY\n")
		centroids = []
		for fly in contours:
			max_coordinates = np.amax(fly, axis=0)
			min_coordinates = np.amin(fly, axis=0)
			centroid = (max_coordinates + min_coordinates)/2
			centroid_file.write(str(centroid[0,0]) + "\t" + str(centroid[0,1]) + "\n")
			cv2.circle(img, (centroid[0,0], centroid[0,1]), 2, (0, 255, 0), -1)
			centroids.append(centroid)
		return centroids, img


def find_contours_with_background():
	pics = cPickle.load(open('output/all_imgs.pickle', 'r'))
	for scene in pics.keys():
		background_filename = os.path.join('output', scene, 'background.jpg')		
		background_img = cv2.imread(background_filename, 0)
		x = 0
		for larva_grey in pics[scene]:
			x += 1

			no_background_img = cv2.absdiff(larva_grey, background_img)
			#cv2.imshow('background-corrected', no_background_img)
			#cv2.waitKey(0)

			blurred = cv2.GaussianBlur(no_background_img, (5,5), 0)			
			ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)			
			#cv2.imshow('thresholded?', thresh)
			#cv2.waitKey(0)
			cv2.imwrite(os.path.join('output', scene, 'thresholded', 'thresholded_img_' + str(x) + '.jpg'), thresh)

			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			
			centroid_filename = os.path.join('output', scene, 'contours', 'background_subtraction', 'centroids_' + str(x) + '.txt')
			centroids, larva_grey = find_centroids(centroid_filename, contours, larva_grey)			 
					
			cv2.drawContours(larva_grey, contours, -1, (0,255,0), 1)
			cv2.imshow('contours', larva_grey)
			cv2.waitKey(0)
			cv2.imwrite(os.path.join('output', scene, 'contours', 'background_subtraction', 'contour_of_img_' + str(x) + ".jpg"), larva_grey) 


def find_contours_with_adaptive_thresholding():
	pics = cPickle.load(open('output/all_imgs.pickle', 'r'))
	for scene in pics.keys():
		background_filename = os.path.join('output', scene, 'background.jpg')		
		background_img = cv2.imread(background_filename, 0)
		x = 0
		for larva_grey in pics[scene]:
			x += 1

			blurred = cv2.GaussianBlur(larva_grey, (5,5), 0)
			thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2) 			
			#ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)		
			cv2.imshow('thresholded?', thresh)
		 	#cv2.waitKey(0)
			cv2.imwrite(os.path.join('output', scene, 'thresholded', 'thresholded_img_' + str(x) + '.jpg'), thresh)

			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			centroid_filename = os.path.join('output', scene, 'contours', 'adaptive_threshold', 'centroids_' + str(x) + '.txt')
			centroids, larva_grey = find_centroids(centroid_filename, contours, larva_grey)	

			cv2.drawContours(larva_grey, contours, -1, (0,255,0), 1)
			cv2.imshow('contours', larva_grey)
			#cv2.waitKey(0)
			cv2.imwrite(os.path.join('output', scene, 'contours', 'adaptive_threshold', 'contour_of_img_' + str(x) + ".jpg"), larva_grey) 


def find_contours_with_otsu_thresholding():
	pics = cPickle.load(open('output/all_imgs.pickle', 'r'))
	for scene in pics.keys():
		background_filename = os.path.join('output', scene, 'background.jpg')		
		background_img = cv2.imread(background_filename, 0)
		x = 0
		for larva_grey in pics[scene]:
			x += 1

			#thresh = cv2.adaptiveThreshold(larva_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2) 
			blurred = cv2.GaussianBlur(larva_grey, (5,5), 0)			
			ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)		
			cv2.imshow('thresholded?', thresh)
		 	#cv2.waitKey(0)
			cv2.imwrite(os.path.join('output', scene, 'thresholded', 'thresholded_img_' + str(x) + '.jpg'), thresh)

			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			centroid_filename = os.path.join('output', scene, 'contours', 'otsu_threshold', 'centroids_' + str(x) + '.txt')
			centroids, larva_grey = find_centroids(centroid_filename, contours, larva_grey)	

			cv2.drawContours(larva_grey, contours, -1, (0,255,0), 1)
			cv2.imshow('contours', larva_grey)
			#cv2.waitKey(0)
			cv2.imwrite(os.path.join('output', scene, 'contours', 'otsu_threshold', 'contour_of_img_' + str(x) + ".jpg"), larva_grey) 


def find_contours_vanilla():
	pics = cPickle.load(open('output/all_imgs.pickle', 'r'))
	for scene in pics.keys():
		x = 0
		for larva_grey in pics[scene]:
			x += 1	
			larva_grey = 255-larva_grey
			ret, thresh = cv2.threshold(larva_grey, 127, 255, cv2.THRESH_BINARY) 
			cv2.imshow('original', larva_grey)
			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			cv2.imshow('original 2', larva_grey)			

			centroid_filename = os.path.join('output', scene, 'contours', 'vanilla', 'centroids_' + str(x) + '.txt')
			centroids, larva_grey = find_centroids(centroid_filename, contours, larva_grey)	
			
			cv2.drawContours(larva_grey, contours, -1, (0,255,0), 1)			
			cv2.imshow('contours', larva_grey)
			cv2.waitKey(0)
			cv2.imwrite(os.path.join('output', scene, 'contours', 'vanilla', 'contour_of_img_' + str(x) + ".jpg"), larva_grey) 


main()

