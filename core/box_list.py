from loguru import logger
import tensorflow as tf


class BoxList(object):
    def __init__(self,
                 boxes
                 ):
        if tf.shape(boxes)[1] != 4:
            num_boxes = tf.shape(boxes)[0]
            dim_boxes = tf.shape(boxes)[1]
            logger.error('The box shape must be [N,4] where N is number of '
                         'boxes. Instead a box of shape [{},{}] was found. '
                         'Aborting.'.format(num_boxes, dim_boxes))
            raise ValueError('The box shape must be [N,4] where N is number of '
                             'boxes. Instead a box of shape [{},{}] was found. '
                             'Aborting.'.format(num_boxes, dim_boxes))
        self._data = {
            'boxes': boxes.ref()
        }

    @property
    def get(self):
        return self._data['boxes']

    @property
    def num_boxes(self):
        return tf.shape(
            self._data['boxes']
        )[0]

    def get_center_coordinates_and_sizes(self):
        with tf.name_scope('Center_coordinates_and_sizes'):
            box_corners = self.get()
            ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
            width = xmax - xmin
            height = ymax - ymin
            ycenter = ymin + height / 2.
            xcenter = xmin + width / 2.
            return [ycenter, xcenter, height, width]

    def area(self):
        with tf.name_scope('Box_area'):
            y_min, x_min, y_max, x_max = tf.split(
                value=self.get(), num_or_size_splits=4, axis=1)
            return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    def height_width(self):
        with tf.name_scope('Height_width'):
            y_min, x_min, y_max, x_max = tf.split(
                value=self.get(), num_or_size_splits=4, axis=1)
            return tf.squeeze(y_max - y_min, [1]), tf.squeeze(x_max - x_min,
                                                              [1])

    def scale(self, y_scale, x_scale):
        with tf.name_scope('Scale'):
            y_scale = tf.cast(y_scale, tf.float32)
            x_scale = tf.cast(x_scale, tf.float32)
            y_min, x_min, y_max, x_max = tf.split(
                value=self.get(), num_or_size_splits=4, axis=1)
            y_min = y_scale * y_min
            y_max = y_scale * y_max
            x_min = x_scale * x_min
            x_max = x_scale * x_max
            self._data['boxes'] = tf.concat([y_min, x_min, y_max, x_max],
                                            1).ref()
