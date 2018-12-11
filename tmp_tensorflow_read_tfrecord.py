
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

tfrecord_filename = 'dog.tfrecords'

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = {
        'height':tf.FixedLenFeature([],tf.int64),
        'width':tf.FixedLenFeature([],tf.int64),
        'image_string':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.float32)
        })

    image = tf.decode_raw(features['image_string'],tf.uint8)
    label = tf.cast(features['label'],tf.float32)

    height = tf.cast(features['height'],tf.int64)
    width = tf.cast(features['width'],tf.int64)

    image = tf.reshape(image,[height,width,3])

    image_size_const = tf.constant((IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype=tf.int32)

    resize_image = tf.image.resize_image_with_crop_or_pad(image=image,target_height=IMAGE_HEIGHT,target_width=IMAGE_WIDTH)

    #images,label = tf.train.shuffle_batch([resize_image,label],batch_size=2,capacity=30,num_threads=1,min_after_dequeue=10)

    #return images,label

    return resize_image,label
filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=10)

images, labels = read_and_decode(filename_queue)

#init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

init_op = tf.local_variables_initializer()

with tf.Session()  as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    for i in range(3):
        img, lab = sess.run([images, labels])

        #print(img.shape)
        #print(lab)
        #plt.imshow(img[0,:,:,:])
        plt.imshow(img[:,:,:])
        plt.show()

    
