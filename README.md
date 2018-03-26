阿里云PAI平台, 使用TensorFlow文件操作的例程
-----------------------------
目录结构:
```
├── checkpoint_dir 模型保存路径
│   └── ...
├── generateTFR.py 生成TFRecords
├── inference.py 构造模型
├── logs tensorboard保存路径
│   └── ...
├── losses.py 构造损失函数
├── main.py 程序主入口
├── reader.py 构造读取管线
├── tfrecord TFRecords存放目录 ( 本地开发使用 )
│   ├── train_0.tfr
│   └── ...
└── upload_to_oss.py 打包上传脚本
```

代码下载地址:
[这里](https://github.com/wqj97/PAI_Tensorflow_IO_example/archive/1.0.zip)
