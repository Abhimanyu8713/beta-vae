import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

class Data(object):

	def __init__(self):
		#image resolution
		self.width = 128
		self.height = 128
		self.radius_range = [5,30]	#range of radius in px
		self.shape_obj = "square"
		self.PLOT = False
		self.image_list = []
		self.FILENAME = 'shape_dataset_valid.h5'
	#randomly generates radius and centre of the object
	def random_gen(self):
		self.radius = np.random.randint(self.radius_range[0],self.radius_range[1])
		self.center = np.random.randint(self.width,size=2)
		if self.center[0]+self.radius > self.width  or self.center[0]-self.radius < 0 or self.center[1]+self.radius > self.height  or self.center[1]-self.radius < 0:
			# print("Again",self.radius,self.center)
			self.random_gen()
		# else:
		# 	print(self.radius,self.center)

	#generate square of given radius and center
	def gen_shape_obj(self,radius,center):
		# print("inside plotting",radius,center)
		img = np.zeros([self.width,self.height])
		for x in range(center[0]-radius,center[0]+radius):
			for y in range(center[1]-radius,center[1]+radius):
				# print(x,y)
				img[x,y] = 1
		if self.PLOT:
			plt.imshow(img)
			plt.show()
		img = img.reshape(1,self.width,self.height)

		return img

	def run(self):
		for i in range(2000):
			self.random_gen()
			self.image_list.append(self.gen_shape_obj(self.radius,self.center))

		self.image_array = np.concatenate(self.image_list)
		print(self.image_array.shape)
		self.data_file = h5py.File(self.FILENAME,'w')
		self.data_file.create_dataset('shapes',data = self.image_array)

	def read_file(self):
		self.file = h5py.File(self.FILENAME,'r')
		imgs = self.file['shapes']
		plt.imshow(imgs[0,:])
		plt.show()

if __name__ == "__main__":

	data = Data()
	data.run()
	data.read_file()