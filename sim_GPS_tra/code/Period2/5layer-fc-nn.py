import dataWashing
import math
import tensorflow as tf


# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, one person's one week feature vector is a trainX, length is 54, )       X [batch, n]
# n = 54, decided by the number of POI in POI.txt 
# m = 2, decided by the number of target classes 
# \x/x\x/x\x/x\x/x\x/ ✞     -- fully connected layer (relu+dropout) W1 [n, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/ ✞       -- fully connected layer (relu+dropout) W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/ ✞         -- fully connected layer (relu+dropout) W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/ ✞           -- fully connected layer (relu+dropout) W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[m]
#          ·                                                        Y5 [batch, m]

#get feature_vector as train_X, Y is corresponding label
def load_data():
	train_X, train_Y = dataWashing.data_to_feature()
	return train_X, train_Y

def random_batch(train_X, train_Y, batch_size):
	#this function create batch_size number train batch, each train time input a batch into training, instead of inputing all train data at the same time.
	rnd_indices = np.random.randint(0,len(X_train), batch_size)
	X_batch = train_X[rnd_indices]
	Y_batch = train_Y[rnd_indices]
	return X_batch, Y_batch

def fc_layers(train_X, train_Y):

	#feature_length = the total number of POI
	feature_length = len(train_X[0]) 
	label_length = len(set(train_Y))
	
	#five layers and their number of neurons (the last layer has label_length softmax neurons)
	L1,L2,L3,L4,L5 = 200,100,60,30,label_length

	#weights initialized with small random values between -0.2 and 0.2
	#When using RELUs, make sure biases are initialized with small *positive* values for example 0.1 = tf.ones([K])/10
	W1 = tf.Variable(tf.truncated_normal([feature_length,L1],stddev = 0.1))
	B1 = tf.Variable(tf.ones([L1])/10)
	W2 = tf.Variable(tf.truncated_normal([L1,L2],stddev = 0.1))
	B2 = tf.Variable(tf.ones([L2])/10)
	W3 = tf.Variable(tf.truncated_normal([L2,L3],stddev = 0.1))
	B3 = tf.Variable(tf.ones([L3])/10)
	W4 = tf.Variable(tf.truncated_normal([L3,L4],stddev = 0.1))
	B4 = tf.Variable(tf.ones([L4])/10)
	W5 = tf.Variable(tf.truncated_normal([L4,L5],stddev = 0.1))
	B5 = tf.Variable(tf.zeros([L5]))

	#the model, with dropout at each layer
	XX = X
	Y1 = tf.nn.relu(tf.matmul(XX,W1)+B1)
	Y1d = tf.nn.dropout(Y1,pkeep)

	Y2 = tf.nn.relu(tf.matmul(Y1d,W2)+B2)
	Y2d = tf.nn.dropout(Y2,pkeep)

	Y3 = tf.nn.relu(tf.matmul(Y2d,W3)+B3)
	Y3d = tf.nn.dropout(Y3,pkeep)

	Y4 = tf.nn.relu(tf.matmul(Y3d,W4)+B4)
	Y4d = tf.nn.dropout(Y4,pkeep)

	Ylogits = tf.matmul(Y4d,W5) + B5
	Y = tf.nn.softmax(Ylogits)
	return Ylogits
	
def train(data, label, learning_rate, pkeep_in, n_epochs, batch_size):
	test_ratio = 0.2 
	test_size = int(len(data)* test_ratio)
	train_X = data[:-test_size]
	train_Y = label[:-test_size]
	X_test = data[-test_size:]
	Y_test = label[-test_size:]

	n_inputs = train_X.shape[1]
	n_ouputs = len(set(train_Y))
	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, shape = (None, n_inputs),name = 'x')
		Y_label = tf.placeholder(tf.float32, shape = (None), name = 'y')
		pkeep = tf.placeholder(tf.float32)
		
	Ylogits = fc_layers(X, Y_label)
	
	with tf.name_scope('loss'):
		#cross-entropy loss function (= -sum(Y_i*log(Yi)) ), normalised for batches of all train X
		#TensorFlow provides the softmax_cross_entropywith_logits function to avoid numerical stability problems with log(0)which is NaN
		cross_entropy = tf.nn.softmax_cross_entropywith_logits(logits = Ylogits, labels = Y_label)
		loss = tf.reduce_mean(cross_entropy, name ='loss')
		loss_summary = tf.summary.scalar('loss', loss)
	
	global_step = tf.Variable(0, trainable = False)
	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		grads_and_vars = optimizer.compute_gradients(loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

	with tf.name_scope('eval'):
		predictions = tf.argmax(Ylogits,1)
		#accuracy of the trained model, between 0(worst) and 1 (best)
		correct = tf.nn.in_top_k(Ylogits, Y_label,1)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		acc_summary = tf.summary.scalar('acc', accuracy)
		
	summary_op = tf.summary.merge([loss_summary, acc_summary])
	
	#write model info into checkpoint, incase it stops and then model could continue to train
	checkpoint_path = './checkpoints/model.ckpt'
	checkpoint_epoch_path = checkpoint_path + '.epoch'
	final_model_path = './checkpoints/model'

	now = datetime.utnow().strftime("%Y%m%d%H%M%S")
	logdir = './logs' + now
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
	saver = tf.train.Saver()
	
	n_epochs = n_epochs
	batch_size = batch_size
	n_batches = int(np.ceil(len(data)/ batch_size))

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		
		if os.path.isfile(checkpoint_epoch_path):
			# if the checkpoint file exists, restore the model and load the epoch number
			with open(checkpoint_epoch_path, 'rb') as f:
				start_epoch = int(f.read())
			print('Training was interuptted. Continuing at epoch', start_epoch)
			saver.restore(sess, checkpoint_path)
		else:
			start_epoch = 0
			sess.run(init)
		
		for epoch in range(start_epoch, n_epochs):
		# one training time.
			for batch_index in range(n_batches):
			# one training time concluds n_batches times batch_size size training sample
				X_batch, Y_batch = random_batch(train_X, train_Y, batch_size)
				sess.run(train_op, feed_dict{X: X_batch, Y_label: Y_batch, pkeep: pkeep_in})
			loss_val, summary_str, test_pred, test_acc = sess.run([loss, summary_op, predictions, accuracy], feed_dict{X: X_test, Y_label: Y_test, pkeep: 1.0})
			
			file_writer.add_summary(summary_str, epoch)
			if epoch % 50 ==0:
				print('Epoch:', epoch, '\tLoss:', loss_val, '\tAcc:', test_acc)
				saver.save(sess, checkpoint_path)
				with open(checkpoint_epoch_path, 'wb') as f:
					f.write(b'%d' %(epoch + 1))
		saver.save(sess, final_model_path)
		Y_pred = predictions.eval(feed_dict = {X:X_test, y: Y_test})
		print('precision_score', precision_score(Y_test, Y_pred))
		print('recall_score', recall_score(Y_test, Y_pred))
		
		sess.close()


if __name__ == '__main__':

	print('TensorFlow version ' + tf.__version__)
	tf.set_random_seed(0) # a

	X, Y = load_data()
	
	#training times
	n_epochs = 5000
	
	#how many samples train at a time
	batch_size = 64
	
	#probability of keeping a node during dropout, 1.0 at test time(no dropout) and 0.75 at training time
	pkeep = 0.75
	
	#learning rate decay 
	# the learning rate is : 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
	lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
	
	train(X, Y, lr, pkeep, n_epochs, batch_size)

















