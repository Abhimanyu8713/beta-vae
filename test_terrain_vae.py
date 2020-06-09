import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import terrain_dataset as td
import cv2
# %matplotlib inline

data_loader = td.DataLoader()

batch_size = 1
HEIGHT = 128
WIDTH = 128
EPOCHS = 1000
BATCHES = int(data_loader.dataset_len_test/batch_size)
channels = 1

X_in = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH,channels], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH,channels], name='Y')

Y_flat = tf.reshape(Y, shape=[-1, channels*HEIGHT * WIDTH])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels =1
n_latent = 8

reshaped_dim = [-1, 1, 1, 9216]
def lrelu(x, alpha=0.3):
	return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob):
	activation = lrelu
	with tf.variable_scope("encoder", reuse=None):
		X = tf.reshape(X_in, shape=[-1, HEIGHT, WIDTH, channels])
		x = tf.layers.conv2d(X, filters = 32, kernel_size=4, strides=2,  activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		x = tf.layers.conv2d(x, filters = 64, kernel_size=4, strides=2,  activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		x = tf.layers.conv2d(x, filters = 128, kernel_size=4, strides=2,  activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		x = tf.layers.conv2d(x, filters = 256, kernel_size=4, strides=2,  activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))


		x = tf.contrib.layers.flatten(x)
		print("######shape of flattened x:", x.shape)

		mn = tf.layers.dense(x, units=n_latent)
		sd = 0.5 * tf.layers.dense(x, units=n_latent)  
		epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
		z  = mn + tf.multiply(epsilon, tf.exp(sd))
		
		return z, mn, sd

def decoder(sampled_z, keep_prob):
	with tf.variable_scope("decoder", reuse=None):
		x = tf.layers.dense(sampled_z, units=9216, activation=tf.nn.relu)
		x = tf.reshape(x, reshaped_dim)
		x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=2, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
		x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=6, strides=2, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) ##for size 128*128
		x = tf.layers.conv2d_transpose(x, filters=dec_in_channels, kernel_size=2, strides=2, activation=tf.nn.sigmoid,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

		img = tf.reshape(x, shape=[-1, HEIGHT, WIDTH,1])
		return img

with tf.device('/device:gpu:0'):
	sampled, mn, sd = encoder(X_in, keep_prob)
	dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, HEIGHT*WIDTH])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
print("################",img_loss)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, './logfile/temp_models_2020_06_07_01_48_53/95.0_model.ckpt')
# train_writer = tf.summary.FileWriter("/home/abhimanyu8713/abhimanyu_research/RL_Hexapod/RL_CPG_gait/xMonsterCPG/summary_train", sess.graph)
# test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')




total_batch = data_loader.test_data(batch_size = batch_size, scaling_factor = 1)
print(total_batch.shape)
loss_epoch = 0
for j in range(BATCHES):
# batch = [np.reshape(b, [HEIGHT, WIDTH]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
	batch = [np.reshape(b, [HEIGHT,WIDTH,1]) for b in total_batch[j,:,:,:,0]]
	# img_ = plt.imread(batch[0])
	# img_ = batch[0].reshape(128,128)
	# plt.imshow(np.reshape(batch[0], [HEIGHT, WIDTH]))
	# plt.show()
	# print("##############", (batch[0].shape))
	z,ls, d, i_ls, d_ls, mu, sigm = sess.run([sampled, loss, dec, img_loss, latent_loss, mn, sd] ,feed_dict = {X_in: batch, Y: batch, keep_prob: 1})
	original_img = np.reshape(batch[0], [HEIGHT, WIDTH])*255
	reconstructed_image = d[0,:,:,0]*255
	print("Decoded image is ",d.shape)
	img_conc = np.concatenate((original_img,reconstructed_image),axis = 1)
	# img_conc = original_img
	cv2.imwrite("./results/img_htmap_{}.jpg".format(j),img_conc)
	plt.imshow(np.reshape(batch[0], [HEIGHT, WIDTH]),cmap='gray')
	plt.show()
	plt.imshow(d[0,:,:,0],cmap='gray')
	plt.show()
	print(ls, np.mean(i_ls), np.mean(d_ls))
'''
# tf.summary.scalar('img_loss',img_loss)
# tf.summary.scalar('latent_loss',latent_loss)

		# print(summary)
	# summary = tf.Summary(value=[tf.Summary.value(tag='loss_epoch',simple_value=loss_epoch)])
		# print(i, loss_print)
	# train_writer.add_summary(summary, i)
	# if not i % 40:
	# 	# save_model = saver.save(sess,"/home/abhimanyu8713/abhimanyu_research/RL_Hexapod/RL_CPG_gait/xMonsterCPG/temp_models/{}_model.ckpt".format(i%40))
	# 	print("Model saved in path: %s" % save_model)
	# 	ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})


# randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
# imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
# imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

# for img in imgs:
# 	plt.figure(figsize=(1,1))
# 	plt.axis('off')
# 	plt.imshow(img, cmap='gray')
'''