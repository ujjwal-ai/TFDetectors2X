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

    def add_field(self, fieldname, fieldtensor):
        if fieldname in self._data.keys():
            logger.warning('The field {} was alrady found. Overwriting.'.format(
                fieldname
            )
            )
        self._data[fieldname] = fieldtensor.ref()
        return None

    def get_field(self, fieldname):
        if fieldname not in self._data.keys():
            logger.error('The filed {} was not found.'.format(fieldname))
            raise KeyError('The filed {} was not found.'.format(fieldname))

        return self._data[fieldname]

    @property
    def data(self):
        return self._data

    @property
    def num_boxes(self):
        return tf.shape(
            self._data['boxes']
        )[0]

    def get_center_coordinates_and_sizes(self):
        with tf.name_scope('Center_coordinates_and_sizes'):
            box_corners = self.get_field('boxes')
            ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
            width = xmax - xmin
            height = ymax - ymin
            ycenter = ymin + height / 2.
            xcenter = xmin + width / 2.
            return [ycenter, xcenter, height, width]
