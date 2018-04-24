import tensorflow as tf
import tflib.ops.linear
import tflib as lib

def Generator(n_samples, DIM=64, feature_dim=10, BATCH_SIZE=50, N_CLUSTER=10, noise=None, name='Generator.'):
    if noise is None:
        noise = tf.random_normal([n_samples, 64])

    output = lib.ops.linear.Linear(name + 'Generator.Input', 64, DIM, noise)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear(name + 'Generator.L1', DIM, feature_dim, output)
    return tf.reshape(output, [-1, feature_dim])

def Discriminator(inputs, DIM=64, feature_dim=10, BATCH_SIZE=50, N_CLUSTER=10, name = 'Discriminator.'):

    output = lib.ops.linear.Linear(name + 'Discriminator.Input', feature_dim, DIM, inputs)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear(name + 'Discriminator.L2', DIM, 1, output)
    return tf.reshape(output, [-1])

def MNN(inputs, DIM=64, feature_dim=10, BATCH_SIZE=50, N_CLUSTER=10, name="MMM"):
    print('MNN')

    output = lib.ops.linear.Linear('MNN.Input', feature_dim, DIM, inputs)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear('ENN.L2', DIM, N_CLUSTER, output)
    output = tf.nn.softmax(output)
    return tf.reshape(output, [-1, N_CLUSTER])