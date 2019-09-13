import tensorflow as tf

dataX = [[3.5, 4.7, 2.3],
         [4.4, 5.7, 4.1],
         [2.5, 7.3, 1.2],
         [8.5, 3.3, 4.8],
         [4.9, 6.4, 5.7],
         [7.2, 7.1, 7.4],
         [5.6, 8.2, 6, 5]]

dataY = [[20.8], [29.1], [21.7], [30.5], [35.8], [44.6], [42.5]]

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

cost = tf.reduce_mean(tf.square(H - y))

optimization = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimization.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    _, _w, _b = sess.run([train, w, b], feed_dict={x: [3.9, 2.1, 4.7, 8.5, 1.9, 6.3, 8.9],
                                                   y: [9.1, 4.8, 10.7, 18.0, 4.9, 13.2, 18.4]})
    print(_w, _b)
