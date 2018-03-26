# coding=utf-8
import tensorflow as tf
import os
import pandas as pd

# writer_test = tf.python_io.TFRecordWriter('test.tfr')

labels = pd.read_csv('/Users/wanqianjun/Desktop/天池比赛数据/temp_path/second_a_train_index_20180313.csv')
files = tf.gfile.Glob('/Users/wanqianjun/Desktop/天池比赛数据/temp_path/train/*.txt')
file_count = len(files)
tfrecords_part = (len(files) / 50000) + 1

writer = []

for writer_index in xrange(tfrecords_part):
    writer.append(tf.python_io.TFRecordWriter('./tfrecord/train_{}.tfr'.format(writer_index)))

for key, file_path in enumerate(files):
    data = tf.gfile.FastGFile(file_path, 'rb').read()
    id = os.path.basename(file_path).replace('.txt', '')
    label = labels.query('id == {}'.format(id))['type'].values[0]

    example = tf.train.Example(features=tf.train.Features(feature={
        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id]))
    }))

    writer[key / 50000].write(example.SerializeToString())

    if key % 1000 == 0:
        print("已经处理了 {}%".format(round((float(key) / file_count) * 100, 2)))