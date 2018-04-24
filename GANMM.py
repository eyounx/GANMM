import os
import tensorflow as tf
import numpy as np

class GANMM:
    def __init__(self,
                 feature_dim,
                 n_cluster,
                 generator,
                 discriminator,
                 classifier,
                 graph=None,
                 batch_size=50,
                 critic_iters=5,
                 name="GANMM"
                 ):
        self.Generator = generator
        self.Discriminator = discriminator
        self.Classifier = classifier

        self.feature_dim = feature_dim
        self.n_cluster = n_cluster
        self.batch_size=batch_size
        self.critic_iters = critic_iters

        self.graph = tf.Graph() if graph is None else graph

        with self.graph.as_default() as g:
            self._create_graph()
            self.sess = tf.Session(
                graph=self.graph
            )
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=9999)

    def _create_graph(self):
        self.real_data = tf.placeholder(tf.float32, shape=[self.batch_size, self.feature_dim])
        self.MNN_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.feature_dim])
        self.MNN_target = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_cluster])

        self.proba = self.Classifier(self.MNN_input, feature_dim=self.feature_dim, BATCH_SIZE=self.batch_size, N_CLUSTER=self.n_cluster, name="MNN")

        self.fake = []
        for i in range(0, self.n_cluster):
            n = 'G' + str(i) + '.'
            self.fake.append(self.Generator(self.batch_size, feature_dim=self.feature_dim, BATCH_SIZE=self.batch_size, N_CLUSTER=self.n_cluster, name=n))

        disc_real = []
        disc_fake = []

        for i in range(0, self.n_cluster):
            n = 'D' + str(i) + '.'
            disc_real.append(self.Discriminator(self.real_data, feature_dim=self.feature_dim, BATCH_SIZE=self.batch_size, N_CLUSTER=self.n_cluster, name=n))
            disc_fake.append(self.Discriminator(self.fake[i], feature_dim=self.feature_dim, BATCH_SIZE=self.batch_size, N_CLUSTER=self.n_cluster, name=n))

        self.param_dict = {}

        for i in range(0, self.n_cluster):
            self.param_dict['G{}'.format(i)] = [var for var in tf.trainable_variables() if 'G{}'.format(i) in var.name]
            self.param_dict['D{}'.format(i)] = [var for var in tf.trainable_variables() if 'D{}'.format(i) in var.name]

        self.param_dict['MNN'] = [var for var in tf.trainable_variables() if 'MNN' in var.name]

        self.gen_cost = []
        self.disc_cost = []
        self.gen_train_op = []
        self.disc_train_op = []

        self.MNN_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.MNN_target, logits=self.proba))
        self.MNN_train_op = tf.train.RMSPropOptimizer(
            learning_rate=5e-5
        ).minimize(self.MNN_cost, var_list=self.param_dict['MNN'])

        for i in range(0, self.n_cluster):
            self.gen_cost.append(-tf.reduce_mean(disc_fake[i]))
            self.disc_cost.append(tf.reduce_mean(disc_fake[i]) - tf.reduce_mean(disc_real[i]))
            self.gen_train_op.append(tf.train.RMSPropOptimizer(learning_rate=5e-5
                                                          ).minimize(self.gen_cost[i], var_list=self.param_dict['G' + str(i)]))
            self.disc_train_op.append(tf.train.RMSPropOptimizer(learning_rate=5e-5
                                                           ).minimize(self.disc_cost[i], var_list=self.param_dict['D' + str(i)]))

        clip_ops = []
        for var in [var for var in tf.trainable_variables() if 'Discriminator' in var.name]:
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        self.clip_disc_weights = tf.group(*clip_ops)


    def train(self,
              data_gen,
              gen_set,
              full_data=None,
              n_pretrain=500,
              n_iter=200000,
              log_path=None,
              save_path="Result"
              ):


        print("pretraining ...")
        for iteration in range(n_pretrain):
            for model_idx in range(0, self.n_cluster):
                _gen_cost, _ = self.sess.run([self.gen_cost[model_idx], self.gen_train_op[model_idx]])

                disc_iters = self.critic_iters
                for i in range(disc_iters):
                    _data, _targets = next(gen_set[model_idx])
                    _disc_cost, _ = self.sess.run(
                        [self.disc_cost[model_idx], self.disc_train_op[model_idx]],
                        feed_dict={self.real_data: _data}
                    )


        print("epsilon-EM ...")
        gen = data_gen
        for iteration in range(n_iter):

            # train MNN
            for cccc in range(0, 1):
                _fake = self.sess.run(self.fake)
                _MNN_cost = np.zeros([self.n_cluster])
                for i in range(0, self.n_cluster):
                    _MNN_target = np.zeros([self.batch_size, self.n_cluster])
                    _MNN_target[:, i] = 1
                    _, _MNN_cost[i] = self.sess.run([self.MNN_train_op, self.MNN_cost],
                                                  feed_dict={self.MNN_input: _fake[i], self.MNN_target: _MNN_target})

            # train GAN
            _disc_cost = [0] * self.n_cluster
            _gen_cost = [0] * self.n_cluster

            for i in range(0, self.n_cluster):

                _, _gen_cost[i] = self.sess.run([self.gen_train_op[i], self.gen_cost[i]])

                if iteration < 500:
                    num_chose = 25
                elif iteration < 1000:
                    num_chose = 40
                elif iteration < 2000:
                    num_chose = 45
                else:
                    num_chose = 48

                for it in range(disc_iters):

                    _chosen_data = []
                    _rest_data = []
                    _rest_data_proba = []
                    while True:
                        _data, _targets = next(gen)
                        _proba = self.sess.run(self.proba, feed_dict={self.MNN_input: _data})
                        _proba = np.array(_proba)
                        # record choosen data
                        tmp = []
                        idx = np.argmax(_proba, axis=1)
                        if (idx == i).any():
                            tmp = _data[idx == i, :]
                        else:
                            idx = np.argmax(_proba, axis=0)
                            tmp = _data[idx[i], :]
                            tmp = tmp.reshape([1, self.feature_dim])
                        if len(_chosen_data):
                            _chosen_data = np.vstack((_chosen_data, tmp))
                        else:
                            _chosen_data = tmp
                        # record rest data
                        tmp = []
                        idx = np.argmax(_proba, axis=1)
                        if (idx != i).any():
                            tmp = _data[idx != i, :]
                            tmp_proba = _proba[idx != i, :]
                        else:
                            idx = np.argmin(_proba, axis=0)
                            tmp = _data[idx[i], :]
                            tmp = tmp.reshape([1, self.feature_dim])
                            tmp_proba = _proba[idx[i], :]
                        if len(_rest_data):
                            _rest_data = np.vstack((_rest_data, tmp))
                        else:
                            _rest_data = tmp
                        if len(_rest_data_proba):
                            _rest_data_proba = np.vstack((_rest_data_proba, tmp_proba))
                        else:
                            _rest_data_proba = tmp_proba

                        if _chosen_data.shape[0] >= num_chose and _rest_data.shape[0] >= self.batch_size - num_chose:
                            break

                    _chosen_data = np.vstack((_chosen_data[0:num_chose, :],
                                              self.sampleRestData(_rest_data, _rest_data_proba, i,
                                                             self.batch_size - num_chose)))

                    _disc_cost[i], _ = self.sess.run(
                        [self.disc_cost[i], self.disc_train_op[i]],
                        feed_dict={self.real_data: _chosen_data}
                    )
                    _ = self.sess.run(self.clip_disc_weights)


            if iteration % 100 == 99:
                trn_img, trn_target = full_data

                pred_lbl = []
                iter_num = int(np.floor(trn_img.shape[0] / 50))
                for i in range(0, iter_num):
                    batch = trn_img[50 * i:50 * (i + 1), :]
                    _proba = self.sess.run(self.proba, feed_dict={self.MNN_input: batch})
                    tmp = np.argmax(_proba, axis=1)
                    if len(pred_lbl) == 0:
                        pred_lbl = tmp
                    else:
                        pred_lbl = np.hstack((pred_lbl, tmp))
                purity, nmi, ari = self.get_performance(trn_target, pred_lbl)
                print("iter={}, purity={:.4f}, nmi={:.4f}, ari={:.4f}".format(
                            iteration, purity, nmi, ari
                        ))
                if log_path is not None:
                    os.makedirs(log_path,exist_ok=True)
                    logger = open(os.path.join(log_path,"log.txt"), 'a')
                    logger.write(
                        "iter={}, purity={:.4f}, nmi={:.4f}, ari={:.4f}\n".format(
                            iteration, purity, nmi, ari
                        )
                    )
                    logger.close()

            if iteration % 5000 == 4999:
                iter_path = save_path + 'iter_{}'.format(iteration)
                os.makedirs(iter_path,exist_ok=True)
                self.saver.save(self.sess, iter_path + '/model')

    def predict(self,x):
        x = np.reshape(x,[-1,self.feature_dim])
        _proba = self.sess.run(self.proba, feed_dict={self.MNN_input: x})
        return np.argmax(_proba, axis=1)


    def sampleRestData(self, data, pred_lbl, idx, num):

        P = pred_lbl[:, idx]
        P = P / P.sum()
        sample_idx = np.random.choice(len(P), num, replace=False, p=P)

        return data[sample_idx, :]

    def get_purity(self,lbl, pred_lbl):
        avg_acc = 0
        for i in range(0, self.n_cluster):
            c = lbl[pred_lbl == i]
            total_num = c.shape[0]
            max_num = 0
            max_label = -1
            for j in range(0, self.n_cluster):
                tmp = (c == j).sum()
                if tmp > max_num:
                    max_num = tmp
                    max_label = j
            avg_acc += max_num
        avg_acc = avg_acc / (pred_lbl.shape[0] + 0.0)
        return avg_acc

    def get_performance(self,y_true, y_pred):
        purity = self.get_purity(y_true, y_pred)
        from sklearn.metrics import normalized_mutual_info_score as NMI
        from sklearn.metrics import adjusted_rand_score as ARI
        nmi = NMI(y_true, y_pred)
        ari = ARI(y_true, y_pred)
        return purity, nmi, ari

    def close(self):
        self.sess.close()
