import os, sys
sys.path.append(os.getcwd())
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())

np.random.seed(7)

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

def Generator(n_samples, noise=None, name='Generator.'):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear(name + 'Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, name = 'Discriminator.'):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(name + 'Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(name + 'Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear(name + 'Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def ENN(inputs):
    print 'ENN'
    output = tf.reshape(inputs, [-1,1,28,28])
    print output.get_shape().as_list()

    output = lib.ops.conv2d.Conv2D('ENN.1', 1, DIM, 5, output, stride=2)
    print output.get_shape().as_list()
    output = tf.nn.relu(output,name='ENN')

    output = lib.ops.conv2d.Conv2D('ENN.2', DIM, 2*DIM, 5, output, stride=2)
    print output.get_shape().as_list()
    output = tf.nn.relu(output,name='ENN')

    output = tf.reshape(output,[BATCH_SIZE,-1])
    print output.get_shape().as_list()

    output = tf.layers.dense(output,10,tf.nn.softmax,name='ENN.3')
    print output.get_shape().as_list()

    return tf.reshape(output, [-1, 10])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
ENN_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
ENN_target = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10])

proba = ENN(ENN_input)

fake = []
for i in range(0,10):
    n = 'G' + str(i) + '.'
    fake.append(Generator(BATCH_SIZE, name=n))


disc_real = []
disc_fake = []

for i in range(0, 10):
    n = 'D' + str(i) + '.'
    disc_real.append(Discriminator(real_data, name=n))
    disc_fake.append(Discriminator(fake[i], name=n))

pre_fake = Generator(BATCH_SIZE,name='pre_G.')
pre_disc_real = Discriminator(real_data, name='pre_D.')
pre_disc_fake = Discriminator(pre_fake, name='pre_D.')

param_dict={}
param_dict['G0'] = [var for var in tf.trainable_variables() if 'G0' in var.name]
param_dict['G1'] = [var for var in tf.trainable_variables() if 'G1' in var.name]
param_dict['G2'] = [var for var in tf.trainable_variables() if 'G2' in var.name]
param_dict['G3'] = [var for var in tf.trainable_variables() if 'G3' in var.name]
param_dict['G4'] = [var for var in tf.trainable_variables() if 'G4' in var.name]
param_dict['G5'] = [var for var in tf.trainable_variables() if 'G5' in var.name]
param_dict['G6'] = [var for var in tf.trainable_variables() if 'G6' in var.name]
param_dict['G7'] = [var for var in tf.trainable_variables() if 'G7' in var.name]
param_dict['G8'] = [var for var in tf.trainable_variables() if 'G8' in var.name]
param_dict['G9'] = [var for var in tf.trainable_variables() if 'G9' in var.name]

param_dict['D0'] = [var for var in tf.trainable_variables() if 'D0' in var.name]
param_dict['D1'] = [var for var in tf.trainable_variables() if 'D1' in var.name]
param_dict['D2'] = [var for var in tf.trainable_variables() if 'D2' in var.name]
param_dict['D3'] = [var for var in tf.trainable_variables() if 'D3' in var.name]
param_dict['D4'] = [var for var in tf.trainable_variables() if 'D4' in var.name]
param_dict['D5'] = [var for var in tf.trainable_variables() if 'D5' in var.name]
param_dict['D6'] = [var for var in tf.trainable_variables() if 'D6' in var.name]
param_dict['D7'] = [var for var in tf.trainable_variables() if 'D7' in var.name]
param_dict['D8'] = [var for var in tf.trainable_variables() if 'D8' in var.name]
param_dict['D9'] = [var for var in tf.trainable_variables() if 'D9' in var.name]
param_dict['D9'] = [var for var in tf.trainable_variables() if 'D9' in var.name]

param_dict['ENN'] = [var for var in tf.trainable_variables() if 'ENN' in var.name]
param_dict['pre_G'] = [var for var in tf.trainable_variables() if 'pre_G' in var.name]
param_dict['pre_D'] = [var for var in tf.trainable_variables() if 'pre_D' in var.name]


gen_cost = []
disc_cost = []
gen_train_op = []
disc_train_op = []

pretrain_gen_cost = -tf.reduce_mean(pre_disc_fake)
pretrain_disc_cost = tf.reduce_mean(pre_disc_fake) - tf.reduce_mean(pre_disc_real)
pretrain_gen_op = tf.train.RMSPropOptimizer(learning_rate=5e-5
                                                      ).minimize(pretrain_gen_cost, var_list=param_dict['pre_G'])
pretrain_disc_op = tf.train.RMSPropOptimizer(learning_rate=5e-5
                                                       ).minimize(pretrain_disc_cost, var_list=param_dict['pre_D'])

ENN_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ENN_target, logits=proba))
ENN_train_op = tf.train.RMSPropOptimizer(
    learning_rate=5e-5
).minimize(ENN_cost, var_list=param_dict['ENN'])

for i in range(0,10):
    gen_cost.append(-tf.reduce_mean(disc_fake[i]))
    disc_cost.append(tf.reduce_mean(disc_fake[i]) - tf.reduce_mean(disc_real[i]))
    gen_train_op.append(tf.train.RMSPropOptimizer(learning_rate=5e-5
                                                 ).minimize(gen_cost[i], var_list=param_dict['G'+str(i)]))
    disc_train_op.append(tf.train.RMSPropOptimizer(learning_rate=5e-5
                                                  ).minimize(disc_cost[i], var_list=param_dict['D'+str(i)]))


clip_ops = []
for var in lib.params_with_name('Discriminator'):
    clip_bounds = [-.01, .01]
    clip_ops.append(
        tf.assign(
            var,
            tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
        )
    )
clip_disc_weights = tf.group(*clip_ops)


# For saving samples
def generate_image(idx, frame):
    sample = session.run(fake[idx])
    lib.save_images.save_images(
        sample.reshape((BATCH_SIZE, 28, 28)),
        'pic/mnist/samples_{}_{}.png'.format(frame, idx)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE,filepath='/data/zhouwj/Dataset/mnist.pkl.gz')
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images,targets

saver = tf.train.Saver(max_to_keep=9999999)

# Train loop
with tf.Session() as session:

    np.random.seed(7)
    tf.set_random_seed(7)

    session.run(tf.initialize_all_variables())

    for v in tf.trainable_variables():
        print v.name

    gen = inf_train_gen()

    print 'Train First GAN ...'
    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _gen_cost, _ = session.run([pretrain_gen_cost, pretrain_gen_op])

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data,_targets = gen.next()
            _disc_cost, _ = session.run(
                [pretrain_disc_cost, pretrain_disc_op],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost7', _disc_cost)
        if iteration > 0:
            lib.plot.plot('train gen cost7', _gen_cost)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    pretrain_disc_cost,
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost7', np.mean(dev_disc_costs))

            generate_image(0,iteration)

        if iteration % 8000==7999:
            model_path = 'pic/model/mnist/model/'
            for v in tf.trainable_variables():
                var = session.run(v)
                print var.shape
                np.save(model_path+v.name.replace('/','_'),var)
            print model_path
            break

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush('pic/mnist/')

        lib.plot.tick()

    print 'Copy 10 GAN ...'
    for i in range(0,10):
        for j in range(0,len(param_dict['G'+str(i)])):
            rand_num = (1e-7)*random.random()
            assign_op = param_dict['G' + str(i)][j].assign(param_dict['pre_G'][j]+rand_num)
            session.run(assign_op)
        generate_image(i, iteration)
    for i in range(0,10):
        for j in range(0,len(param_dict['D'+str(i)])):
            rand_num = (1e-7)*random.random()
            assign_op = param_dict['D' + str(i)][j].assign(param_dict['pre_D'][j]+rand_num)
            session.run(assign_op)

    saver.save(session,'model/mnist/model')
    print 'save done'

    iteration = 0
    for iteration in xrange(ITERS):

        #train ENN
        for cccc in range(0,1):
            _fake = session.run(fake)
            _ENN_cost = np.zeros([10])
            for i in range(0,10):
                _ENN_target = np.zeros([BATCH_SIZE,10])
                _ENN_target[:,i]=1
                _, _ENN_cost[i] = session.run([ENN_train_op, ENN_cost] ,feed_dict={ENN_input:_fake[i], ENN_target:_ENN_target})
            lib.plot.plot('ENN cost7', np.mean(_ENN_cost))

        #train GAN
        _disc_cost = [0]*10
        _gen_cost = [0]*10

        for i in range(0,10):

            _, _gen_cost[i] = session.run([gen_train_op[i], gen_cost[i]])

            for it in xrange(disc_iters):

                _chosen_data = []
                while True:
                    _data, _targets = gen.next()
                    _proba = session.run(proba,feed_dict={ENN_input:_data})
                    idx = np.argmax(_proba,axis=1)
                    if (idx==i).any():
                        tmp = _data[idx==i, :]
                    else:
                        idx = np.argmax(_proba, axis=0)
                        tmp = _data[idx[i],:]
                        tmp = tmp.reshape([1,784])
                    if len(_chosen_data):
                        _chosen_data = np.vstack((_chosen_data, tmp))
                    else:
                        _chosen_data = tmp
                    if _chosen_data.shape[0] >= BATCH_SIZE:
                        break
                _chosen_data = _chosen_data[0:BATCH_SIZE,:]

                _disc_cost[i], _ = session.run(
                    [disc_cost[i], disc_train_op[i]],
                    feed_dict={real_data: _chosen_data}
                )
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)
            lib.plot.plot('G{} gen cost7'.format(i), _gen_cost[i])
            lib.plot.plot('G{} disc cost7'.format(i), _disc_cost[i])

        if iteration%100==99:
            for i in range(0,10):
                generate_image(i, iteration)

        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush('pic/mnist/')
        lib.plot.tick()

        if iteration%2000==1999:
            model_path = 'model/mnist/iter_{}'.format(iteration)
            os.mkdir(model_path)
            saver.save(session,model_path+'/model')
