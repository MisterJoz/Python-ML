import tensorflow as tf

print(tf.__version__)

node1 = tf.placeholder(dtype=tf.float32)
node2 = tf.placeholder(dtype=tf.float32)
node3 = tf.placeholder(dtype=tf.float32)
node4 = node1 * node2
node5 = node4 + node3

sess = tf.Session()
output = sess.run(node5, feed_dict={node1: [3.0], node2: [4.0], node3: [5.0]})

print(output)