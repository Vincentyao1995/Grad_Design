import data_to_feature as DTF
import math
import tensorflow as tf
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score

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
	# this function transfer ['类别1', '类别2', '类别1'..'类别n']共n个类别的m个训练label （train_Y） into [[0,0,0,0,1,0...共n个]...[]](共m个vecotr，vector长度为n)其中vector的第n位表示的类别n的具体是哪个类，可以由label_number_table这个字典得到。{小红：0, ..小明:n} 代表小明是vector[n]位置上的预测数字。如果这个sample是小明，在label集中这个vector[n] == 1 其他都是0。
	
    train_X, train_Y = DTF.data_to_feature()
	label_number_table = {}
	index = 0
	label_num = len(set(train_Y))
	for key in sorted(set(train_Y)):
		label_number_table.setdefault(key, index)
		index += 1
	for index, y in enumerate(train_Y):
		position = label_number_table[y]
		train_Y[index] = [0.0 for i in range(label_num)]
		train_Y[index][position] = 1.0
    return train_X, train_Y, label_number_table

def random_batch(train_X, train_Y, batch_size):
    #this function create batch_size number train batch, each train time input a batch into training, instead of inputing all train data at the same time.
    rnd_indices = np.random.randint(0,len(train_X), batch_size)
    X_batch = train_X[rnd_indices]
    Y_batch = train_Y[rnd_indices]
    return X_batch, Y_batch

def fc_layers(train_X, train_Y):

    #feature_length = the total number of POI
    POI_num = int(train_X.shape[1])
	feature_num = int(train_X.shape[2])
    label_length = int(train_Y.shape[1])
	
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
    XX = tf.reshape(X,[-1, POI_num*feature_num])
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
    return Y,Ylogits
    
def train(data, label, learning_rate_in, pkeep_in, n_epochs, batch_size):
    label_length = label.shape[1]
	test_ratio = 0.2 
    test_size = int(len(data)* test_ratio)
    train_X = data[:-test_size]
    train_Y = label[:-test_size]
    X_test = data[-test_size:]
    Y_test = label[-test_size:]

    n_inputs1 = len(train_X[0])
    n_inputs2 = len(train_X[0][0])
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, shape = (None, n_inputs1,n_inputs2, ),name = 'x')
        Y_label = tf.placeholder(tf.float32, shape = (None, label_length), name = 'y')
        pkeep = tf.placeholder(tf.float32)
        step = tf.placeholder(tf.int32)
    Y_pred, Ylogits = fc_layers(X, Y_label)
    
    with tf.name_scope('loss'):
        #cross-entropy loss function (= -sum(Y_i*log(Yi)) ), normalised for batches of all train X
        #TensorFlow provides the softmax_cross_entropywith_logits function to avoid numerical stability problems with log(0)which is NaN
        cross_entropy = tf.nn.softmax_cross_entropywith_logits(logits = Ylogits, labels = Y_label)
        loss = tf.reduce_mean(cross_entropy, name ='loss')
        loss_summary = tf.summary.scalar('loss', loss)
    
    global_step = tf.Variable(0, trainable = False)
    with tf.name_scope('train'):
		lr = 0.0001 +  tf.train.exponential_decay(learning_rate_in, step, 2000, 1/math.e)
		train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    with tf.name_scope('eval'):
		predictions = tf.argmax(Ylogits, 1)
        correct_prediction = tf.equal(tf.argmax(Y_label, 1), tf.argmax(Y_pred, 1))
        #accuracy of the trained model, between 0(worst) and 1 (best)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_summary = tf.summary.scalar('acc', accuracy)
        
    summary_op = tf.summary.merge([loss_summary, acc_summary])
    
    #write model info into checkpoint, incase it stops and then model could continue to train
    checkpoint_path = './checkpoints/model.ckpt'
    checkpoint_epoch_path = checkpoint_path + '.epoch'
    final_model_path = './checkpoints/model'

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
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
                sess.run(train_step, feed_dict = {X: X_batch, Y_label: Y_batch, pkeep: pkeep_in, step:epoch})
            loss_val, summary_str, test_pred, test_acc = sess.run([loss, summary_op, predictions, accuracy], feed_dict = {X: X_test, Y_label: Y_test, pkeep: 1.0})
            
            file_writer.add_summary(summary_str, epoch)
            if epoch % 50 ==0:
                print('Epoch:', epoch, '\tLoss:', loss_val, '\tAcc:', test_acc)
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, 'wb') as f:
                    f.write(b'%d' %(epoch + 1))
        saver.save(sess, final_model_path)
        Y_pred = predictions.eval(feed_dict = {X:X_test, Y_label: Y_test, pkeep: 1.0})
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
    batch_size = 3
    
    #probability of keeping a node during dropout, 1.0 at test time(no dropout) and 0.75 at training time
    pkeep = 0.75
    
    #learning rate decay 
    # the learning rate is : 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
    lr = 0.003
    
    train(np.array(X), np.array(Y), lr, pkeep, n_epochs, batch_size)

















