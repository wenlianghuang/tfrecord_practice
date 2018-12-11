
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
import keras
#height first
IMAGE_HEIGHT = 320
#width second
IMAGE_WIDTH = 480

tfrecord_filename = 'layer.tfrecords'

def read_and_decode(filename_queue):
    #reader = tf.TFRecordReader()
    
    #We need to do our uncompression
    compression = tf.python_io.TFRecordCompressionType.GZIP
    reader = tf.TFRecordReader(options = tf.python_io.TFRecordOptions(compression))


    #tf.TFRecordReader.read() return tuples of tensor(key,value)
    _,serialized_example = reader.read(filename_queue)
    
    #Also call tf.io.parse_single_example,a "dict" mapping return 
    features = tf.parse_single_example(serialized_example,features = {
        #'height':tf.FixedLenFeature([],tf.int64),
        #'width':tf.FixedLenFeature([],tf.int64),
        'image_string':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
        #'label':tf.FixedLenresize_nearest_neighbore([],tf.string)
        })
    #decode_raw to 0 & 1

    #image = tf.decode_raw(features['image_string'],tf.uint8)
    image = tf.image.decode_png(features['image_string'],channels = 3)
    
    #label = tf.decode_raw(features['label'],tf.uint8)
    label = tf.cast(features['label'],tf.int64)
    
    ##iamge = tf.image.crop_to_bounding_box(image,offset_height=100,offset_width=100,target_height=100,target_width=100)
    #height = tf.cast(features['height'],tf.int64)
    #width = tf.cast(features['width'],tf.int64)
    
    #Binary of image 0,1 actually is the same...
    #image = tf.reshape(image,[IMAGE_HEIGHT,IMAGE_WIDTH,3])
    #image = tf.reshape(image,[height,width,3])
    
    

    #pad: a soft, stuffed cushion used as a saddle

    #To resize the plot again with IMAGE_HEIGHT and IMAGE_WIDTH
    ##resize_image = tf.image.resize_image_with_crop_or_pad(image=image,target_height=IMAGE_HEIGHT,target_width=IMAGE_WIDTH)
    resize_image = tf.image.crop_to_bounding_box(image = image,offset_height=80,offset_width=80,target_height=300,target_width=440)
    resize_image = tf.image.resize_images(resize_image,(32,32),method=1)
    #https://www.tensorflow.org/api_docs/python/tf/train/shuffle_batch
    ##images,label = tf.train.shuffle_batch([resize_image,label],batch_size=2,capacity=30,num_threads=1,min_after_dequeue=10)

    ##return images,label
    
    #Do not mess the resize_image and label
    return resize_image,label
    ##return image,label 

####################################
#Tensorflow starts to read TFRecord#
####################################


filename_queue = tf.train.string_input_producer([tfrecord_filename])

images, labels = read_and_decode(filename_queue)
#print(images)
#print(labels)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

#The name of image file is add by myself
image_filename_list = ['100_1','100_2','100_3']

with tf.Session()  as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tmp_img = []
    tmp_lab = []
    for i in range(3):
        img, lab = sess.run([images, labels])

        #print(img.shape) 
        #plt.imshow(img[1,:,:,:])
        #print(lab)
        plt.imshow(img[:,:,:])
        tmp_img.append(img)
        tmp_lab.append(lab)
        titlename = str(image_filename_list[i]) + '_c.jpg'
        plt.savefig(titlename)
        plt.show()

    coord.request_stop()
    coord.join(threads)
    myarray_img = np.asarray(tmp_img)
    myarray_lab = np.asarray(tmp_lab)
    #print(myarray_img.shape)
    #print(myarray_lab.shape)
    
    label_test_OneHot = keras.utils.to_categorical(myarray_lab)
    #print(label_test_OneHot)
    
