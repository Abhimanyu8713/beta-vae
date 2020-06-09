import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import terrain_dataset as td
import yaml
# %matplotlib inline
import time
import os

data_loader = td.DataLoader()

batch_size = 32
HEIGHT = 128
WIDTH = 128
EPOCHS = 1000
BATCHES = int(data_loader.dataset_len_train/batch_size)
TEST_BATCHES = int(data_loader.dataset_len_test/batch_size)
SAVE_MODEL = True
channels = 1
SCALING_FACTOR = 1
# image_dimension = '{}*WIDTH*channels'.
lr = 0.0005

n_latent = 8
timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")


#LOG FILE
config_file = os.getcwd()+'/logfile/config/{}.yaml'.format(timestamp)
users = {'image_dimension':'{}*{}*{}'.format(HEIGHT,WIDTH,channels),'epochs':EPOCHS,'scaling_factor':SCALING_FACTOR,'batch_size':batch_size,'lr_starting': lr,'lr_constant':True,'weights_init':'xavier','latent_variable':n_latent}
with open(config_file, 'w') as f:      
	data = yaml.dump(users, f) 

X_in = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH,channels], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH,channels], name='Y')

Y_flat = tf.reshape(Y, shape=[-1, channels*HEIGHT * WIDTH])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels =1

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


with tf.device('/device:gpu:1'):
	sampled, mn, sd = encoder(X_in, keep_prob)
	dec = decoder(sampled, keep_prob)
beta = 10
unreshaped = tf.reshape(dec, [-1,1*HEIGHT*WIDTH])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = (-0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1))
# loss = tf.reduce_mean(img_loss + latent_loss)
loss = tf.reduce_mean(img_loss) + beta*tf.reduce_mean(latent_loss)

# global_step = tf.Variable(0, trainable=False)
starter_learning_rate = lr
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,10000, 0.96, staircase=True)

optimizer = tf.train.AdamOptimizer(starter_learning_rate).minimize(loss)
tf.summary.scalar('loss',loss)
merged = tf.summary.merge_all()
# print("######",dir(optimizer))
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter("/home/cobra/abhimanyu_research/terrain_classifier/logfile/logdir_2_{}/summary_train".format(timestamp), sess.graph)
test_writer = tf.summary.FileWriter("/home/cobra/abhimanyu_research/terrain_classifier/logfile/logdir_2_{}/summary_test".format(timestamp), sess.graph)
'''
import time
print("hello")
time.sleep(10)
# test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
'''
count = 0

for i in range(EPOCHS):

	summary_test_total = 0
	summary_train_total = 0

	total_batch = data_loader.train_data(batch_size = batch_size, scaling_factor = SCALING_FACTOR) #[batch x batch_size x 128 x 128 x 2]
	# print("Batch shape",total_batch.shape)
	test_batch = data_loader.test_data(batch_size = batch_size, scaling_factor = SCALING_FACTOR)
	# print(total_batch.shape)
	# loss_epoch = 0
	for j in range(BATCHES):
		batch = [np.reshape(b, [HEIGHT,WIDTH,1]) for b in total_batch[j,:,:,:,0]]
		# print("total_batch",np.array(batch).shape)
		# plt.imshow(batch[0][:][:][0])
		# plt.show()

	# 	# print("######Shape of array, ",np.array(batch).shape)
		_,loss_train,summary_train = sess.run([optimizer, loss,merged] ,feed_dict = {X_in: batch, Y: batch})
		# print("Batches", j)

		# count += 1
		# print("Type",int(summary_train))
		# summary_train_total += summary_train
		# count+ = 1
	train_writer.add_summary(summary_train, i)

	for k in range(TEST_BATCHES):
		batch_test = [np.reshape(b, [HEIGHT,WIDTH,1]) for b in test_batch[k,:,:,:,0]]
		# print("Shape of array, ",np.array(batch_test).shape)
		_,loss_test,summary_test = sess.run([optimizer, loss,merged] ,feed_dict = {X_in: batch_test, Y: batch_test})
		# print("Batche/s", k)
		# summary_test_total += summary_test
		# count+ = 1

	test_writer.add_summary(summary_test,i)
	print("After epoch {}, the train loss and test loss is {} , {}".format(i,loss_train,loss_test))
	if not i % 10:
		if SAVE_MODEL:
			save_model = saver.save(sess,"/home/cobra/abhimanyu_research/terrain_classifier/logfile/temp_models_{}/{}_model.ckpt".format(timestamp,i/10))
			print("Model saved in path: %s" % save_model)
		ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch_test, Y: batch_test})
		# print("Decoded image is ",np.sum(d[0]))
		# plt.imshow(np.reshape(batch[0], [HEIGHT, WIDTH]))
		# plt.show()
		# plt.imshow(d[0,:,:,0])
		# plt.show()
		print(i, ls, np.mean(i_ls), np.mean(d_ls))


