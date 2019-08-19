import tensorflow as tf

train_x = [[1,1],[0,1],[0,0],[1,0]]
train_y = [1,0,0,0]
s1, s2 = len(train_x), len(train_x[0])

print(s1)
print(s2)


train_y = [0, 0, 1, 1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([s2,1],-1.0, 1.0))
h = tf.matmul(X , W)
hypothesis = tf.div(1.,1. + tf.exp(-h))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(300):
    sess.run(train, feed_dict={X: train_x, Y: train_y})
    if step % 500 == 0:
        print(step, sess.run(cost, feed_dict={X: train_x, Y: train_y}), sess.run(W))

print('-----------------------------------------')

print('[1, 0] :', sess.run(hypothesis, feed_dict={X: [[1,0]]}) > 0.5)
print('[1, 1] :', sess.run(hypothesis, feed_dict={X: [[1,1]]}) > 0.5)
sess.close()
