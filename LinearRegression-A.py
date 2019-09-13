import tensorflow as tf
#linear regression

X = tf.constant([[1.0, 1.0],
                 [2.0, 1.0],
                 [3., 1.0],
                 [4., 1.0]])

y = tf.constant([[2.0],
                 [3.0],
                 [4.0],
                 [5.0]])

w = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(X), X)), tf.transpose(X)), y)

sess = tf.Session()
output = sess.run(w)
print(output)