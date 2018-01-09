# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt


class DataDistribution(object):
    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        # 生产在（-self.range, self.range） 之间元素个数为N的等间隔的数据
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01



def linear(input, output_dim, scop=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scop or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def discriminator(input, h_dim):
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.tanh(linear(h1, h_dim * 2, 'd2'))
    h3 = tf.sigmoid(linear(h2, 1, 'd3'))
    return h3

# 定义优化器 学习率 按步数衰减
def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, log_every):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every

        self.mlp_hidden_size = 4
        self.learning_rate = 0.03
        self._create_model()

    def _create_model(self):
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        with tf.variable_scope('Gen'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        with tf.variable_scope('Disc') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size)
            scope.reuse_variables()  # 共享变量 (相当于把G依赖的z的placeholder变量依赖进来)
            self.D2 = discriminator(self.G, self.mlp_hidden_size)
        # 对于判别网络 希望 D1趋近于1  D2趋近于0  他们的计算差值为0 误差最小
        # 找到D1和D2趋近的值就是判别网络需要做的事情
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        # 对于生成网络 希望 D2趋近于1 希望生成值就是真实值 所有值为1
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))
        # loss_d和loss_g的D2只有矛盾，正是通过这种矛盾是噪音的输入生成出来一个真实的数据
        # 主要是定义损失函数，接下来定义优化不断的进行梯度下降求解就可以了

        # 拿到初始化参数
        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D_pre")
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Gen")

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        with tf.Session() as session:
            tf.global_variables_initializer().run()

            # 预训练判别网络
            num_pretrain_steps = 1000
            for step in range(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                   self.pre_input: np.reshape(d, (self.batch_size, 1)),
                   self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })

            self.weightD = session.run(self.d_pre_params)

            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):
                session.run(v.assign(self.weightD[i]))

            for step in range(self.num_steps):
                # 训练判别网络
                x = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # 训练生成网络
                z = self.gen.sample(self.batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # 打印日志
                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))
                if step % 100 == 0 or step == 0 or step == self.num_steps - 1:
                    self._plot_distributions(session)

    def _samples(self, session, num_points=10000, num_bins=100):
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return pd, pg

    def _plot_distributions(self, session):
        pd, pg = self._samples(session)
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

def main(args):
    model = GAN(
        DataDistribution(),
        GeneratorDistribution(range=8),
        args.num_steps,
        args.batch_size,
        args.log_every
    )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=1200, help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12, help='the batch size')
    parser.add_argument('--log-every', type=int, default=10, help='print loss after this many steps')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())


