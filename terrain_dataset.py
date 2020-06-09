# import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import glob
import os
import cv2
import h5py
import torch
h_resize = 128
w_resize = 128
# print(files)

class DataLoader:
	def __init__(self):
		# self.path_train = os.getcwd()+"/depth_image_files/ht_map/depth_image_all_aug_train.h5"
		# # self.path_test  = os.getcwd()+"/depth_image_files/ht_map/depth_image_all_aug_test.h5"
		# self.path_test  = os.getcwd()+"/depth_image_files/ht_map_real/height_image_all.h5"

		self.path_train = os.getcwd()+"/beta-vae/data/shape_dataset_train.h5"
		# self.path_test  = os.getcwd()+"/depth_image_files/ht_map/depth_image_all_aug_test.h5"
		self.path_test  = os.getcwd()+"/beta-vae/data/shape_dataset_test.h5"

		# self.path_train = os.getcwd()+"/shape_dataset_train.h5"
		# self.path_test  = os.getcwd()+"/shape_dataset_test.h5"

		# self.path_test = os.getcwd()+"/images_test"
		# self.files_test = [f for f in glob.glob(self.path_test + "/*.jpg", recursive=True)]

		depth_image_all_train = h5py.File(self.path_train,'r')
		depth_image_all_test = h5py.File(self.path_test,'r')
		self.files_train = depth_image_all_train['shapes'][:]
		self.files_test = depth_image_all_test['shapes'][:]
		self.dataset_len_train = self.files_train.shape[0]
		self.dataset_len_test = len(self.files_test)
		# print(len(self.files_train))

	def data_shuffle(self,option):
		if option == 0:
			np.random.shuffle(self.files_train)
		else:
			np.random.shuffle(self.files_test)

	def image_scaling_pixels(self,img,scaling_factor):
		return (img**scaling_factor)

	def train_data(self, batch_size = 20, scaling_factor = 1):
		data_total = []
		self.batches = int(self.dataset_len_train/batch_size)
		self.data_shuffle(0)       
		for b in range(self.batches):
			data = []
			for i in range(batch_size):
				image = self.image_scaling_pixels(self.files_train[i+(b*batch_size)], scaling_factor)
				res = cv2.resize(image, dsize=(h_resize, w_resize), interpolation=cv2.INTER_CUBIC)
				# res = torch.from_numpy(res)
				# print("Images ",image.shape)
				data.append(res.reshape(1,h_resize,w_resize,1))
			data = np.concatenate(data,axis = 0)
			# print((data.shape))
			# print(type(data_total))
			data_total.append(data.reshape(1,batch_size,h_resize,w_resize,1))
		data_total = np.concatenate(data_total,axis = 0)
		# data_total = torch.from_numpy(data_total)
		# print(data_total.shape)
		return data_total


	def test_data(self,batch_size = 20,scaling_factor = 1):
		data_total = []
		self.batches = int(len(self.files_test)/batch_size)
		self.data_shuffle(1)
		for b in range(self.batches):
			data = []
			for i in range(batch_size):

				# image = plt.imread(self.files_test[i])
				# image = plt.imread(self.files_test[i+(b*batch_size)])
				image = self.image_scaling_pixels(self.files_test[i+(b*batch_size)], 1)
				# image = (0.21 * image[:,:,0] + 0.72 * image[:,:,1] + 0.07 * image[:,:,2])/255.0
				# print("shape of image coming", image.shape)
				# plt.imshow(image*image,cmap='gray')
				# plt.show()
				res = cv2.resize(image, dsize=(h_resize, w_resize), interpolation=cv2.INTER_CUBIC)
				data.append(res.reshape(1,h_resize,w_resize,1))
			data = np.concatenate(data,axis = 0)
			# print((data.shape))
			# print(type(data_total))
			data_total.append(data.reshape(1,batch_size,h_resize,w_resize,1))
		data_total = np.concatenate(data_total,axis = 0)
		# data_total = torch.from_numpy(data_total).permute
		# print("Resoulution ",data.shape)
		return data_total


if __name__ == "__main__":
	dl = DataLoader()
	data = dl.train_data(10)
	print((data).shape)
# # # # # data_reshape = [np.reshape(b, [h_resize,w_resize]) for b in data]
# # # # # print(np.array(data_reshape).shape) 
# # # # # for i in range(data.shape[0]):
# plt.imshow(data[0,0,0,:,:].reshape(h_resize,w_resize))
# plt.show()
# dl = DataLoader()
# print(dl.train_data(10).shape)