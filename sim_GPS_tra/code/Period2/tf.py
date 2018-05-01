import dataWashing
import math
import tensorflow as tf



print('TensorFlow version ' + tf.__version__)
tf.set_random_seed(0) # a

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
train_X, train_Y = dataWashing.data_to_feature()

#feature_length = the total number of POI
feature_length = len(train_X[0]) 
label_length = len(set(train_Y))
X = tf.placeholder(tf.float32, [None,feature_length,len(train_X[0][0])]) # attention, not pretty clear about dimension size.

# correct answers will go here
Y_ = tf.placeholder(tf.float32,[None,label_num])
# variable learning rate
lr = tf.placeholder(tf.float32)
#Probability of keeping a node during dropout, 1.0 at test time(no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)
#step for variabe learning rate
step = tf.placeholder(tf.int32)

#five layers and their number of neurons (the last layer has label_num softmax neurons)
L1,L2,L3,L4,L5 = 200,100,60,30,label_num
batch_size = 60

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

#cross-entropy loss function (= -sum(Y_i*log(Yi)) ), normalised for batches of all train X
#TensorFlow provides the softmax_cross_entropywith_logits function to avoid numerical stability problems with log(0)which is NaN
cross_entropy = tf.nn.softmax_cross_entropywith_logits(logits = Ylogits, labels = Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

#accuracy of the trained model, between 0(worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))#attention, didn't know the function of this sentence
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#matplotlib visualisation attention, ignore this all
allweights = tf.concat([tf.reshape(W1,[-1]), tf.reshape(W2, [-1]),tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])],0 )
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)

#training step
# the learning rate is : 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

#init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# you can call this function in a loop to train the model, len(train_X)/10 samples at a time
def training_step(i, update_test_data, update_train_data):
	
	global batch_size
	p
	#training on batches of batch_size samples with batch_size labels
	
	for i in range(len(train_X) - step -1):
		batch_X = train_X[i:i+step]
		batch_Y = train_Y[i:i+step]
	batch_X, batch_Y = 

















