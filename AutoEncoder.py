import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 控制训练过程的参数
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 5
examples_to_show = 10
# w网络模型参数
n_input_units = 784    # 输入神经元数量 MNIST data input (img shape : 28*28)
n_hidden1_units = 256  # 编码起第一隐藏层神经元数量（让编码器和解码器都有同样规模的隐藏层
n_hidden2_units = 128  # 编码起第二隐藏层神经元数量（让编码器和解码器都有同样规模的隐藏层
n_output_units = n_input_units  # 解码器输出层神经元数量必须等于输入数据的units数量

# 对一个张量进行全面汇总(均值，标准差，最大最小值，直方图)
def varible_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 根据输入输出节点数量返回权重
def WeightsVarible(n_in, n_out, name_str='weights'):
    return tf.Variable(tf.random_normal([n_in, n_out]), dtype=tf.float32, name=name_str)


# 根据输出节点数量返回偏置
def BiasesVarible(n_out, name_str='biases'):
    return tf.Variable(tf.random_normal([n_out]), dtype=tf.float32, name=name_str)


# 构建编码器
def Encoder(x_origin, activate_func=tf.nn.sigmoid):
    # 编码器第一隐藏层
    with tf.name_scope('Layer1'):
        weights = WeightsVarible(n_input_units, n_hidden1_units)
        biases = BiasesVarible(n_hidden1_units)
        x_code1 = activate_func(tf.nn.xw_plus_b(x_origin, weights, biases))
        varible_summaries(weights)
    # 编码器第二隐藏层
    with tf.name_scope('Layer2'):
        weights = WeightsVarible(n_hidden1_units, n_hidden2_units)
        biases = BiasesVarible(n_hidden2_units)
        x_code = activate_func(tf.nn.xw_plus_b(x_code1, weights, biases))
        varible_summaries(weights)
    return x_code


# 构建解吗器
def Decoder(x_code, activate_func=tf.nn.sigmoid):
    # 解码器第一隐藏层
    with tf.name_scope('Layer'):
        weights = WeightsVarible(n_hidden2_units, n_hidden1_units)
        biases = BiasesVarible(n_hidden1_units)
        x_decode1 = activate_func(tf.nn.xw_plus_b(x_code, weights, biases))
        varible_summaries(weights)
    # 解码器第二隐藏层
    with tf.name_scope('Layer'):
        weights = WeightsVarible(n_hidden1_units, n_output_units)
        biases = BiasesVarible(n_output_units)
        x_decode = activate_func(tf.nn.xw_plus_b(x_decode1, weights, biases))
        varible_summaries(weights)
    return x_decode


# 调用上面写的函数构造计算图
with tf.Graph().as_default():
    # 计算图输入
    with tf.name_scope('X_origin'):
        X_origin = tf.placeholder(tf.float32, [None, n_input_units])
    # 构建编码器
    with tf.name_scope('Encoder'):
        X_code = Encoder(X_origin)
    # 构建解吗器
    with tf.name_scope('Decoder'):
        X_decode = Decoder(X_code)
    # 定义损失节点
    with tf.name_scope('Loss'):
        Loss = tf.reduce_mean(tf.pow(X_origin - X_decode, 2))
    # 定义优化器
    with tf.name_scope('Train'):
        Optimizer = tf.train.RMSPropOptimizer(learning_rate)
        Train = Optimizer.minimize(Loss)

    # 为计算图添加损失节点的标量汇总(scalar summary)
    with tf.name_scope('LossSummary'):
        tf.summary.scalar('loss', Loss)
        tf.summary.scalar('learning_rate', learning_rate)

    # 为计算图添加图像汇总
    with tf.name_scope('ImageSummary'):
        image_origin = tf.reshape(X_origin, [-1, 28, 28, 1])
        image_reconstructed = tf.reshape(X_decode, [-1, 28, 28, 1])
        tf.summary.image('image_origin', image_origin, 10)
        tf.summary.image('image_reconstructed', image_reconstructed, 10)

    # 聚合所有汇总节点
    merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    print("把计算图写入事件文件，在TensorBoard里面查看")
    writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
    writer.flush()

    # 读取数据集
    mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, loss = sess.run([Train, Loss], feed_dict={X_origin: batch_xs})

            if epoch % display_step == 0:
                print("epoch : %03d, loss = %.3f" % (epoch + 1, loss))
                # 运行汇总节点，更新事件文件
                summary_str = sess.run(merged_summary, feed_dict={X_origin: batch_xs})
                writer.add_summary(summary_str, epoch)
                writer.flush()

        writer.close()
        print("训练完毕！")

        # 把训练好的编码器-解码器模型用在测试集上，输出重建后的样本数据
        reconstructions = sess.run(X_decode, feed_dict={X_origin: mnist.test.images[:examples_to_show]})
        # 比较原始图像与重建后的图像
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(reconstructions[i], (28, 28)))
        f.show()
        plt.draw()
