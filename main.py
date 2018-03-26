# coding=utf-8
import tensorflow as tf
import reader
import inference
import losses
import os

# 第一步, 定义OSS路径
tf.flags.DEFINE_string('buckets', './tfrecord', "数据源目录")
tf.flags.DEFINE_string("summaryDir", "logs/", "TensorBoard路径")
tf.flags.DEFINE_string("checkpointDir", "checkpoint_dir/", "模型保存路径")
tf.flags.DEFINE_integer('batch_size', 50, '批大小')
tf.flags.DEFINE_integer('hidden_1_size', 512, '隐藏层1神经元数')
tf.flags.DEFINE_integer('hidden_2_size', 256, '隐藏层2神经元数')
tf.flags.DEFINE_integer('output_size', 4, '输出数')
tf.flags.DEFINE_integer('train_steps', 40000, '训练次数')
tf.flags.DEFINE_integer('num_classes', 4, '分类数')
tf.flags.DEFINE_integer('threads', 18, '读取线程')
tf.flags.DEFINE_float('learning_rate', 1e-3, '学习速率')

tf.flags.DEFINE_boolean("isTrain", True, "定义是否是在训练")

FLAGS = tf.flags.FLAGS

data_type = {'star': 0, 'unknown': 1, 'galaxy': 2, 'qso': 3}

sess = tf.InteractiveSession()

# 构造读取管线
read = reader.Reader(path=FLAGS.buckets,
                     pattem='*.tfr',
                     batch_size=FLAGS.batch_size,
                     is_training=FLAGS.isTrain,
                     num_threads=FLAGS.threads)
# 获得数据和标签
ids, labels, datas = read.read()
labels_one_hot = tf.one_hot(labels, FLAGS.num_classes)

# 构造网络
inference = inference.Inference(data_input=datas,
                                h1_size=FLAGS.hidden_1_size,
                                h2_size=FLAGS.hidden_2_size,
                                is_training=FLAGS.isTrain,
                                num_classes=FLAGS.num_classes)

logits = inference.get_inference()
logits_softmax = inference.get_softmax()

# 构造损失
losses = losses.Losses(logits=logits, labels=labels_one_hot).get_losses()

# 构造优化器
train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(losses)

# 初始化
summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
saver = tf.train.Saver(var_list=tf.trainable_variables())
tf.train.start_queue_runners(sess)
sess.run(tf.global_variables_initializer())

# 计算AUC, ACC
auc = tf.contrib.metrics.streaming_auc(logits_softmax, labels_one_hot)
tf.summary.scalar('auc', auc[1])

sess.run(tf.local_variables_initializer())
merged = tf.summary.merge_all()
# 迭代训练
for i in xrange(FLAGS.train_steps):
    # 取出文件名和数据
    sess.run(train_op)

    if i % 25 == 0 or i == FLAGS.train_steps - 1:
        summary.add_summary(sess.run(merged), i)
        print sess.run(auc[1])

    if i % 500 == 0 or i == FLAGS.train_steps - 1:
        saver.save(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'save.model'))

summary.close()
