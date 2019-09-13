import tensorflow as tf

node1 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
node2 = tf.multiply(node1, node1)
node3 = tf.sqrt(tensordot(node1, node1, 1))
node4 = tf.tensordot(node1, node1, 0)
node5 = tf.
sess = tf.Session()
out1, out2, out3 = sess.run([node2, node3, node4])
print(out1, out2, out3)