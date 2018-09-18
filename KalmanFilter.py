import random
import matplotlib.pyplot as plt

n = 0
def KalmanFileter(pos1, pos2, P):
	#Initialization

	Q = 0.001
	R = 1
	#Predict 
	pos1_ = pos1
	P_ = P + Q

	#Update
	K = P_ / (P_ + R)
	pos_new = pos1_ + K * (pos2 - pos1_)
	P_new = (1 - K) * P_

	return pos_new, P_new

def data():
	N = []
	pos2 = []
	for i in range(200):
		pos2.append(10 + random.gauss(0,1))
		#pos2.append(10)
		N.append(i)
	return pos2, N

if __name__ == '__main__':

	pos2, N = data()
	pos_predict = []
	P = []
	pos_new = 10
	P_new = 1
	for i in range(200):
		pos_new, P_new = KalmanFileter(pos1=pos_new, pos2=pos2[i], P=P_new)
		pos_predict.append(pos_new)
		P.append(P_new)
	plt.plot(N, pos_predict, color='red')
	plt.plot(N, pos2, color='gray')
	plt.show()


	import numpy as np
	import matplotlib.pyplot as plt
	 
####### 生成观测值z_mat  #############	 
	# 创建一个0-99的一维矩阵
	z = [i for i in range(100)]
	z_watch = np.mat(z)

	# 创建一个方差为1的高斯噪声，精确到小数点后两位
	noise = np.round(np.random.normal(0, 5, 100), 2)
	noise_mat = np.mat(noise)
	 
	# 将z的观测值和噪声相加
	z_mat = z_watch + noise_mat
	#print(z_watch)

####### 初始化各个矩阵  ##################	 
	# 定义x的初始状态
	x_mat = np.mat([[10,], [10,]])
	# 定义初始状态协方差矩阵
	p_mat = np.mat([[1, 0], [0, 1]])
	# 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
	f_mat = np.mat([[1, 1], [0, 1]])
	# 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
	q_mat = np.mat([[0.0001, 0], [0, 0.0001]])
	# 定义观测矩阵
	h_mat = np.mat([1, 0])
	# 定义观测噪声协方差
	r_mat = np.mat([1])
	 
	for i in range(100):
	    x_predict = f_mat * x_mat
	    p_predict = f_mat * p_mat * f_mat.T + q_mat
	    kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
	    x_mat = x_predict + kalman *(z_mat[0, i] - h_mat * x_predict)
	    p_mat = (np.eye(2) - kalman * h_mat) * p_predict

	    # plt.plot(x_mat[0, 0], x_mat[1, 0], 'ro', markersize = 1)
	    plt.plot(i, x_mat[0, 0], 'go', markersize = 1)
	plt.plot(z, z_mat[0].tolist()[0], color='r')
	# plt.plot(z, x_mat[0].tolist()[0], color='g', label='predict')
	    
	plt.show()

	# print('z:', z)

	# print('z_mat[0, 1]:', z_mat[0, 1])
	# print('z_mat[0, 0:]:', z_mat[0].tolist()[0])






