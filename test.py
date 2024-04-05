import tensorflow as tf

train_q = tf.train.RMSPropOptimizer(learning_rate=0.001, epsilon=1e-5)

train_q.learning_rate = 0.001

print(train_q.learning_rate)

train_q.learning_rate = 0.01

print(train_q.learning_rate)