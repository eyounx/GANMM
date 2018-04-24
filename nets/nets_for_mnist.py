import tensorflow as tf
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.batchnorm
import tflib as lib

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, DIM=64, feature_dim=784, BATCH_SIZE=50, N_CLUSTER=10, noise=None, name='Generator.'):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear(name + 'Generator.Input', 128, 4*4*4*DIM, noise)
    output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, feature_dim])

def Discriminator(inputs, DIM=64, feature_dim=784, BATCH_SIZE=50, N_CLUSTER=10, name = 'Discriminator.'):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    output = lib.ops.batchnorm.Batchnorm(name + 'Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = lib.ops.batchnorm.Batchnorm(name + 'Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear(name + 'Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def MNN(inputs,DIM=64, feature_dim=784, BATCH_SIZE=50, N_CLUSTER=10,  name="MNN"):
    print('MNN')
    output = tf.reshape(inputs, [-1,1,28,28])
    print(output.get_shape().as_list())

    output = lib.ops.conv2d.Conv2D(name+'.MNN.1', 1, DIM, 5, output, stride=2)
    print(output.get_shape().as_list())
    output = tf.nn.relu(output,name=name+'.MNN.relu1')

    output = lib.ops.conv2d.Conv2D(name+'.MNN.2', DIM, 2*DIM, 5, output, stride=2)
    print(output.get_shape().as_list())
    output = tf.nn.relu(output,name=name+'.MNN.relu2')

    output = tf.reshape(output,[BATCH_SIZE,-1])
    print(output.get_shape().as_list())

    output = tf.layers.dense(output,N_CLUSTER,tf.nn.softmax,name=name+'.MNN.3')
    print(output.get_shape().as_list())

    return tf.reshape(output, [-1, N_CLUSTER])
