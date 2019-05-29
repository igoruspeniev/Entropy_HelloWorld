'''
    Research origin:
    Igor Uspeniev, Entropy of Implicit Functions (April 2019). DOI: 10.13140/RG.2.2.22249.34406
    In case of using this theoretical principle reference to paper is mandatory.

    Code:
    Helloworld example for this paper
    Igor Uspeniev, Ihor Tanyenkov
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rnd

np.random.seed(0)  # determinated random
rnd.seed(0)
tf.set_random_seed(1)  # determinated random

# Parameters
n_layer_1 = 2  #
nptype = 'float64'
tftype = tf.float64

LR = 0.01
num_steps = 500
batch_size = n_layer_1 + 1
display_step = 1

x_from = -5.0  # interval domain start_point
x_to = 5.0  # interval domain end_point
point_count = 1024  # number of points
noise_STD = 0.1


def unknown_function(x): return np.sin(np.pi * x + 1)


data_x = np.arange(x_from, x_to, (x_to - x_from) / point_count, dtype=nptype)
np.random.shuffle(data_x)  # blend values
data_y = unknown_function(data_x)  # + np.random.normal(0, noise_STD, point_count)
print(",".join(list(map(str, data_x[:batch_size]))))  # first batch of arguments
print(",".join(list(map(str, data_y[:batch_size]))))  # first bach of values

zero = tf.constant(0.0, dtype=tftype)

X = tf.placeholder(dtype=tftype, shape=[batch_size])
Y = tf.placeholder(dtype=tftype, shape=[batch_size])

w = tf.Variable(tf.random_normal([n_layer_1], dtype=tftype), name='w')  # weights
# w = tf.Variable(initial_value=[-0.04431111, 0.9475519 ], name='w') # weights
wmccp = [tf.reshape(tf.stack([w[i], w[i + 1], - w[i + 1], w[i]]), (2, 2), name='wm')
         for i in range(0, (n_layer_1 // 2) * 2, 2)]  # weights in matrices: complex conjugate positive
wmccn = [tf.reshape(tf.stack([w[i], - w[i + 1], w[i + 1], w[i]]), (2, 2), name='wm')
         for i in range(0, (n_layer_1 // 2) * 2, 2)]  # weights in matrices: complex conjugate negative
wmczi = [tf.reshape(tf.stack([w[i], zero, zero, w[i]]), (2, 2), name='wm')
         for i in range((n_layer_1 // 2) * 2, n_layer_1)]  # weights in matrices: complex zero imaginary
wm = wmccp + wmccn + wmczi  # weights in matrices


def misc_det(matrix):
    if matrix.shape[0] == 1:
        return matrix[0, 0]
    else:
        sum1 = tf.zeros((2, 2), dtype=tftype)
        for i in range(0, matrix.shape[0]):
            sum1 = tf.math.add(
                sum1,
                tf.linalg.matmul(matrix[0, i], misc_det(
                    np.concatenate([matrix[1:, :][:, :i], matrix[1:, :][:, i + 1:]], 1))) * tf.cast(tf.pow(-1, i),
                                                                                                    dtype=tftype))
        return sum1

def misc_det1(matrix):
    ret = misc_det(matrix)
    print('######', ret.shape)
    return ret

def determinant(x, y):
    matrix = np.asarray([[tf.math.exp(x[i] * wm[j]) for j in range(n_layer_1)] +
                         [tf.reshape(tf.stack([y[i], zero, zero, y[i]]), (2, 2))]
                         for i in range(batch_size)])
    # det = tf.linalg.det(matrix)
    # det = tf.matrix_determinant(matrix)
    det = misc_det1(matrix)
    return det


model_c = determinant(X, Y)

if __name__ == "__main__":
    loss = tf.reduce_mean(tf.square(model_c[0, 0]) + tf.square(model_c[0, 1]))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LR, beta1=0.9, beta2=0.999).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=0.9, epsilon=1e-10)
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        for step in range(num_steps):
            random_indexes = []
            for i in range(batch_size):
                while True:
                    new_index = rnd.randint(0, point_count - 1)
                    if not (new_index in random_indexes):
                        break
                random_indexes.append(new_index)
            feed_dict = {X: data_x[random_indexes],
                         Y: data_y[random_indexes]}

            print('random indexes', random_indexes)
            print('X by indexes', data_x[random_indexes])
            print('Y by indexes', data_y[random_indexes])
            _, loss_value = session.run([optimizer, loss], feed_dict=feed_dict)
            if step % display_step == 0 or step == 1 or step == (num_steps - 1):
                print('Step:', step, ' Loss:', loss_value)
                print('Weights:', w.eval())
                ddddd = session.run(model_c, feed_dict=feed_dict)
                print('determinant', ddddd.shape)
                for wmi in wm:
                    print('WM', wmi.eval())

    plt.plot(data_x, data_y, 'ro')
    data_x_sorted = np.arange(x_from, x_to, (x_to - x_from) / 1000)
    # plt.plot(data_x_sorted, list(map(lambda x: (c1 * np.exp(l1 * x) + c2 * np.exp(l2 * x)), data_x_sorted)))
    plt.show()
