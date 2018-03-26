# coding=utf-8
import oss2
import sys
import zipfile

file_list = ['inference.py', 'losses.py', 'main.py', 'predict.py', 'reader.py']  # 需要打包的文件
Access_Key_ID = ''  # 阿里云access_key_Id
Access_Key_Secret = ''  # 阿里云access_key_secret
OSS_Bucket_endpoint = 'oss-cn-shanghai.aliyuncs.com'  # buckets的endpoint
Bucket_Name = 'mlearn'  # bucket 名
OSS_Put_Path = 'scene/train.zip'  # 上传到oss的路径

with open('main.py') as file:
    flags = False
    for line in file:
        if line == 'tf.flags.DEFINE_boolean("isTrain", True, "定义是否是在训练")\n':
            flags = True
    if not flags:
        sys.stdout.write('FLAGS:is_train set to False, are you sure to upload? y/n\r\n')
        sys.stdout.flush()
        if not raw_input() == 'y':
            exit('script terminate')
        else:
            exit('force put object')

auth = oss2.Auth(access_key_id=Access_Key_ID, access_key_secret=Access_Key_Secret)
bucket = oss2.Bucket(auth, bucket_name=Bucket_Name, endpoint=OSS_Bucket_endpoint)

azip = zipfile.ZipFile('train.zip', 'w')
for file_name in file_list:
    azip.write(file_name, compress_type=zipfile.ZIP_STORED)
azip.close()
bucket.put_object_from_file(OSS_Put_Path, 'train.zip')
print "train script uploaded"
