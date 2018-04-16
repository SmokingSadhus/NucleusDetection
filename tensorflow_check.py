import tensorflow as tf

#a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
#print(sess.run(c))


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


##################################### Basic Tensorflow operations and checks ######################################
a = tf.Variable(np.random.rand(2,2,2))
b = tf.Variable(np.random.rand(2,2,2))
init = tf.variables_initializer([a, b])
m_check = tf.reduce_sum(tf.multiply((a[:,:,0]-b[:,:,0]),(a[:,:,0]-b[:,:,0])))
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(a[:,:,0]))
	print(sess.run(b[:,:,0]))
	print(sess.run(m_check))
############################################################################
