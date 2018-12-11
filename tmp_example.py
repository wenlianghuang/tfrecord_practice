###############concatenate_images##########################################
#Check the images which are download from website, if the #
#size of each image, change them into size (640,485)      # 
########################################################### 
import numpy as np
#scikit_image

import tensorflow as tf
#from skimage.transform import resize
from skimage import transform
import matplotlib.pyplot as plt 

#dog_img = plt.imread('dog_1.jpg')

#resize the image in skimage
#dog_img = transform.resize(dog_img,(640,480))

#save image
#plt.imsave("dog_1c.jpg",dog_img)
#plt.imshow(dog_img)
#plt.show()


###########################################################
#Start to write into file of TFRecords                    #
###########################################################


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _float32_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

image_filename_list = ['dog_1.jpg','dog_2.jpg','dog_3.jpg']

label_list = [1.0,1.2,0.6]

tfrecord_filename = 'dog.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecord_filename)

for image_filename,label in zip(image_filename_list,label_list):
    image = plt.imread(image_filename)

    height,width,depth = image.shape

    image_string = image.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'image_string': _bytes_feature(image_string),
      'label': _float32_feature(label)}))
    writer.write(example.SerializeToString())
writer.close()

###############
#read tfrecord#
###############

#record_iterator = tf.python_io.tf_record_iterator(tfrecord_filename)

#for string_record in record_iterator:
#    example = tf.train.Example()

#    example.ParseFromString(string_record)
    
#    height = int(example.features.feature['height'].int64_list.value[0])
#    width = int(example.features.feature['width'].int64_list.value[0])
#    image_string = (example.features.feature['image_string'].bytes_list.value[0])
#    label = float(example.features.feature['label'].float_list.value[0])

#    image_1d = np.fromstring(image_string,dtype=np.uint8)
#    images = image_1d.reshape((height,width,3))

#plt.imshow(image)
#plt.show()
