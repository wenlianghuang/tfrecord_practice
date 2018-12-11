###############concatenate_images##########################################
#Check the images which are download from website, if the #
#size of each image, change them into size (640,485)      # 
########################################################### 
import numpy as np
from PIL import Image
#scikit_image

import tensorflow as tf
import matplotlib.pyplot as plt 

#layer = plt.imread("./100_2.jpg")
#plt.imshow(layer)
#plt.show()

#dog_img = plt.imread('dog_1.jpg')



###########################################################
#Start to write into file of TFRecords                    #
###########################################################

#Binary data 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

#Int data
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

#Floating data 
def _float32_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

#for i in range(1,4):
#    im = Image.open('100_{}.jpg'.format(i))
#    im.save('100_{}.png'.format(i))


image_filename_list = ['100_1.png','100_2.png','100_3.png']
#image_filename_list = ['100_1.jpg','100_2.jpg','100_3.jpg']
label_list = ['1','2','3']
label_list = map(int,label_list)
tfrecord_filename = 'layer.tfrecords'

#Set the compression from gzip
compression = tf.python_io.TFRecordCompressionType.GZIP

#writer = tf.python_io.TFRecordWriter(tfrecord_filename)
writer = tf.python_io.TFRecordWriter(tfrecord_filename,options = tf.python_io.TFRecordOptions(compression))


#zip(image_filename_list,label_list) : (100_1.jpg,1),
#(100_2.jpg,2),(100_3.jpg,3)
for image_filename,label in zip(image_filename_list,label_list):
    
    #image = plt.imread(image_filename)
    #height,width,depth = image.shape
    
    #set image to string
    #image_string = image.tostring()#
     
    image_string = tf.gfile.FastGFile(('/Users/Apple/tmp_tensorflow_keras/tfrecord_practice/'+image_filename),'rb').read() 
    #build in several Features in the example
    example = tf.train.Example(features = tf.train.Features(feature = {
        #'height':_int64_feature(height),
        #'width':_int64_feature(width),
        'image_string':_bytes_feature(image_string),
        'label':_int64_feature(label)
        #'label':_bytes_feature(label)
        }))

    writer.write(example.SerializeToString())
writer.close()

