import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

H = x * w + b

cost = tf.reduce_mean(tf.square(H - y))

optimization = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimization.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    _, _w, _b = sess.run([train, w, b], feed_dict={x: [3.9, 2.1, 4.7, 8.5, 1.9, 6.3, 8.9],
                                                   y: [9.1, 4.8, 10.7, 18.0, 4.9, 13.2, 18.4]})
    print(_w, _b)
