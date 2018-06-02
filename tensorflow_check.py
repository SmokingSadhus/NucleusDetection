import tensorflow as tf
import numpy as np

#a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
#print(sess.run(c))


from keras import backend as K
K.tensorflow_backend._get_available_gpus()

######################################################################
a = tf.Variable(np.random.rand(4,3,3,3))
init = tf.variables_initializer([a])
m_check = a[:,:,:,1]
sum_val = tf.reduce_sum(m_check) 
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(m_check.get_shape())
    print(sess.run(m_check))
    print(sess.run(sum_val))

exit()
##########################################################################################################################

a = tf.Variable(np.zeros((4,3,3,3)))
init = tf.variables_initializer([a])
m_check = tf.concat([tf.sigmoid(a[:,:,:,0:2]) , tf.nn.relu(a[:,:,:,2:3])],axis = 3)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(m_check.get_shape())
    print(sess.run(m_check))

exit()


################################ Check for custom_activation function ####################################################

#a = tf.Variable(np.random.rand(19,19,5))
a = tf.Variable(np.random.rand(2,2,2))
init = tf.variables_initializer([a])
tf.sigmoid(a[:,:,0:1])
tf.nn.relu(a[:,:,1:2])
#m_check = tf.concat([tf.sigmoid(a[:,:,0:1]) , tf.nn.relu(a[:,:,1:5])],axis = 2)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(a.get_shape())
    print(sess.run(a))

exit()
#############################################################


################################ Check for custom_activation function ####################################################

a = tf.Variable(np.random.rand(19,19,5))
#b = tf.Variable(np.random.rand(2,2,2))
init = tf.variables_initializer([a])
m_check = tf.concat([tf.sigmoid(a[:,:,0:1]) , tf.nn.relu(a[:,:,1:5])],axis = 2)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(m_check.get_shape())
    print(sess.run(m_check))

exit()
#############################################################


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
