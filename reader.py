# coding=utf-8
import tensorflow as tf
import numpy as np
import os

'''
读取数据
'''


# data_type = {'star': 0, 'unknown': 1, 'galaxy': 2, 'qso': 3}

class Reader(object):
    def __init__(self, path, pattem, batch_size=0, is_training=True, num_threads=4):
        """
        读取管线的构造函数
        :param path: 数据存放目录
        :param pattem: pattem
        :param batch_size: 批大小
        :param is_training: 是否是在训练
        :param num_threads: 读取线程数
        """
        self.reader = tf.TFRecordReader()
        self.is_training = is_training
        self.num_threads = num_threads
        if is_training:
            files = tf.gfile.Glob(os.path.join(path, pattem))  # 遍历所有文件
            self.file_queue = tf.train.string_input_producer(files)  # 构造文件队列
            self.batch_size = batch_size

    def read(self):
        _, example = self.reader.read(self.file_queue)

        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(['star', 'unknown', 'galaxy', 'qso'], [0, 1, 2, 3]), -1)
        table.init.run()

        features = tf.parse_single_example(
            example,
            features={
                'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'id': tf.FixedLenFeature([], tf.string)
            })

        data = self.parse_data(features['data'])
        id = features['id']
        label = table.lookup(features['label'])

        if self.is_training:
            ids, labels, datas = tf.train.shuffle_batch([id, label, data],
                                                        batch_size=self.batch_size,
                                                        capacity=1000 + 3 * self.batch_size,
                                                        min_after_dequeue=1000,
                                                        num_threads=self.num_threads)
            return ids, labels, datas
        else:
            ids, labels, datas = tf.train.batch([id, label, data],
                                                batch_size=self.batch_size,
                                                num_threads=self.num_threads)
            return ids, labels, datas

    def parse_data(self, content):
        splited = tf.string_split([content], ',', skip_empty=True)
        splited = tf.string_to_number(splited.values)
        splited = tf.reshape(splited, [2600])

        return splited


if __name__ == '__main__':
    # 测试读取
    sess = tf.InteractiveSession()
    reader = Reader('./tfrecord', '*.tfr', 1)
    ids, labels, datas = reader.read()
    tf.train.start_queue_runners(sess=sess)
    print(sess.run([ids, labels, datas]))
