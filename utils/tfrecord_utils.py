import os
import cv2
import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def encode_image(imagefilename):
    """
    Encodes an image for populating the TFRecord
    :param imagefilename: Full path to the image file.
    :return: A dictionary with following keys-
             1. image: The encoded byte-string representation of the image.
             2. height: The height of the image.
             3. width: The width of the image.
             4. depth: Number of image channels.
             5. filename: Name of the original image file.
             6. extension: Extension of the image file.

    :raises OSError if the image file is not found.
    """
    if not os.path.exists(imagefilename):
        raise OSError('The image {} was not found.'.format(imagefilename))

    image = cv2.imread(imagefilename)
    height, width, depth = image.shape
    image_ext = os.path.splitext(
        os.path.basename(imagefilename)
    )[1]
    encoded_image = cv2.imencode(image_ext, image)[1].tostring()
    encoded_image = tf.compat.as_bytes(encoded_image)
    image_dict = dict(
        image=encoded_image,
        height=height,
        width=width,
        depth=depth,
        filename=os.path.basename(imagefilename).encode('utf8'),
        extension=image_ext.encode('utf8')
    )
    return image_dict
