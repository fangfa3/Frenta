import os
import cv2 
import numpy as np
import glob
import json

DIR = 'F:/AI/Frenta'
scale_ratio	= 0.25 #缩放比例

def read_one_img(file = "00037900_all_videos_robot2_03_mp4.jpg"):
	pic=cv2.imread(file)  
	pic = cv2.resize(pic, (int(pic.shape[1] * scale_ratio), int(pic.shape[0] * scale_ratio)))
	return pic

def read_all_img(file_pos='F:/AI/Frenta/images_12k'):
	imagefiles = glob.glob(file_pos+'/*robot1*')
	return imagefiles

def read_roi(file_posi='F:\\AI\\Frenta\\labels_12k', file='00037900_all_videos_robot2_03_mp4.jpg.json'):
	'''
	从json文件中获取足球位置坐标
	'''
	with open(os.path.join(file_posi, file)) as f:
		json_data = json.load(f)
		roi_pos = []
		if 'Rects' in json_data:
			for obj in json_data['Rects']:
				if obj['properties']['world_cup'][0] == "football":
					roi_pos.append(int(int(obj['x'])*scale_ratio))
					roi_pos.append(int(int(obj['y'])*scale_ratio))
					roi_pos.append(int(int(obj['x'] + obj['w'])*scale_ratio))
					roi_pos.append(int(int(obj['y'] + obj['h'])*scale_ratio))
	return roi_pos

def color_select(img, color_space='BGR', color_channel='R', thresh=(2, 255)):
	'''
	获取某通道颜色阈值内的图片
	'''
	if color_space == 'RGB': 
		channel = img[:,:,color_space.index(color_channel)]
	else:
		space = eval('cv2.cvtColor(img, cv2.COLOR_RGB2%s)'%color_space)
		channel = space[:,:,color_space.index(color_channel)]
	img_color_thresh = np.zeros_like(channel)
	img_color_thresh[(channel >= thresh[0]) & (channel <= thresh[1])] = 255
	return img_color_thresh

def HSV_select(img, hsv_low=np.array([0, 0, 0]), hsv_high=np.array([78, 255, 110])):
	'''
	获取HSV阈值内的图片
	'''
	img_Gauss = cv2.GaussianBlur(img, (9, 9), 0)
	img_HSV = cv2.cvtColor(img_Gauss, cv2.COLOR_BGR2HSV)
	img_HSV_thresh = cv2.inRange(img_HSV, hsv_low, hsv_high)
	return img_HSV_thresh

def get_HSV():
	"""
	功能：读取一张图片，显示出来，转化为HSV色彩空间
	     并通过滑块调节HSV阈值，实时显示
	"""

	pic = cv2.imread('F:/AI/Frenta/images_12k/00002130_all_videos_robot2_03_mp4.jpg') # 根据路径读取一张图片
	image = cv2.resize(pic, (pic.shape[1] // 4, pic.shape[0] // 4))
	cv2.imshow("BGR", image) # 显示图片

	hsv_low = np.array([0, 0, 0])
	hsv_high = np.array([0, 0, 0])

	# 下面几个函数，写得有点冗余

	def h_low(value):
	    hsv_low[0] = value

	def h_high(value):
	    hsv_high[0] = value

	def s_low(value):
	    hsv_low[1] = value

	def s_high(value):
	    hsv_high[1] = value

	def v_low(value):
	    hsv_low[2] = value

	def v_high(value):
	    hsv_high[2] = value

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.createTrackbar('H low', 'image', 0, 255, h_low) 
	cv2.createTrackbar('H high', 'image', 0, 255, h_high)
	cv2.createTrackbar('S low', 'image', 0, 255, s_low)
	cv2.createTrackbar('S high', 'image', 0, 255, s_high)
	cv2.createTrackbar('V low', 'image', 0, 255, v_low)
	cv2.createTrackbar('V high', 'image', 0, 255, v_high)

	while True:
	    dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR转HSV
	    dst = cv2.inRange(dst, hsv_low, hsv_high) # 通过HSV的高低阈值，提取图像部分区域
	    cv2.imshow('dst', dst)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	cv2.destroyAllWindows()

def find_contours(img):
	'''
	边缘检测，返回边缘
	'''
	img_Gauss = cv2.GaussianBlur(img, (3, 3), 0)
	img_Canny = cv2.Canny(img_Gauss, 80, 120)
	#img_HSV_thresh = HSV_select(img_Canny)
	ret , binary = cv2.threshold(img_Canny, 200, 255, cv2.THRESH_BINARY)
	binary, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	return contours

def show_constours(img):
	'''
	show pictures with constours
	'''
	img_HSV_thresh = HSV_select(img=img)
	contours = find_contours(img = img_HSV_thresh)
	cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
	cv2.imshow('img_contours', img)
	cv2.waitKey()

def find_circle(img):
	img_HSV_thresh = HSV_select(img=img)
	contours = find_contours(img_HSV_thresh)
	for i in range(len(contours)):
		if len(contours[i])>80:
			rrt = cv2.fitEllipse(contours[i])
			cv2.ellipse(img, rrt, (0, 0, 255), 2, cv2.LINE_AA)
			x, y = rrt[0]
			cv2.circle(img,(np.int(x), np.int(y)), 4, (255, 0, 0), -1, 8, 0)
	img_circle = img
	return img_circle

def get_roi_all():
### Get ROI of all images 
	images = read_all_img()

	for img in images:
		label = img[-37:] + '.json'
		football = 'F:/AI/Frenta/football/'+img[-35:-30]+'.jpg'
		print(football)
		img = read_one_img(img)
		roi_pos = read_roi(file = label)
		if roi_pos:
			#cv2.rectangle(img, (roi_pos[0], roi_pos[1]), (roi_pos[2], roi_pos[3]), (0,0,255), 2)
			roi_img = img[roi_pos[1]:roi_pos[3], roi_pos[0]:roi_pos[2]]
			cv2.imwrite(football, roi_img)
			#cv2.imshow('ROI_img', roi_img)
		cv2.imshow('img', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()

if __name__ == '__main__':	

####调整图像 begin 

	# scale_ratio = 1
	images = read_all_img(file_pos='F:/AI/Frenta/images_12k')
	for img in images:
		print(img)
		img = read_one_img(img)
		img_HSV_thresh = HSV_select(img)

		# img_GRY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# ret , img_binary = cv2.threshold(img_GRY, 110, 255, cv2.THRESH_BINARY)
		# img_binary = cv2.adaptiveThreshold(img_GRY,255,cv2.ADAPTIVE_THRESH_MEAN_C,  
  #                   cv2.THRESH_BINARY,3,5)
		kernel = np.ones((5,5),np.uint8)
		img_dilation = cv2.dilate(img_HSV_thresh,kernel,iterations = 1) #图片膨胀
		img_erosion = cv2.erode(img_dilation,kernel,iterations =1) #图片腐蚀

		cv2.imshow('2', img)
		cv2.imshow('1', img_erosion)

		if cv2.waitKey(1) & 0xFF == ord('n'):
			continue

		if cv2.waitKey(1) & 0xFF == ord('q'):
	 		break
	cv2.destroyAllWindows()
####调整图像 end 

	# get_HSV()



####图片腐蚀与图片膨胀 begin

		# kernel = np.ones((5,5),np.uint8)  
		# img_binary_erosion = cv2.erode(img_binary,kernel,iterations = 1) #图片腐蚀

		# # img_dilation = cv2.dilate(img_erosion,kernel,iterations = 1) #图片膨胀
		# # img_erosion = cv2.erode(img_dilation,kernel,iterations =1) #图片腐蚀

		# img_binary_Canny = cv2.Canny(img_binary_erosion, 80, 120)

		# # contours = find_contours(img = img)
		# # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
		# # cv2.imshow('img_contours', img)

		# img_HSV_thresh = HSV_select(img)
		# img_HSV_erosion = cv2.erode(img_HSV_thresh,kernel,iterations = 1) #图片腐蚀
		# img_HSV_Canny = cv2.Canny(img_HSV_erosion, 80, 120)
####图片腐蚀与图片膨胀 end

##### Hough Circle begin
		# circles = cv2.HoughCircles(img_binary_Canny,cv2.HOUGH_GRADIENT,1,2,
  #                           param1=50,param2=30,minRadius=0,maxRadius=100)
		# print(circles)
		# if circles == None:
		# 	continue
		# circles = np.uint16(np.around(circles))


		# for i in circles[0,:]:
		#     # draw the outer circle
		#     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
		#     # draw the center of the circle
		#     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
		# print(len(circles[0,:]))

##### Hough Circle end











