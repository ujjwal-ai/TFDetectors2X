from loguru import logger
import tensorflow as tf


class FasterRCNN(object):
    """
    A Class implementing the FasterRCNN object detector described in the
    following paper:
    Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with
    region proposal networks."
     Advances in neural information processing systems. 2015.
    """

    def __init__(self,
                 backbone
                 ):
        """
        Class initializer
        :param backbone: A tf.keras.Model instance representing the backbone
        network.
        """
        self._backbone = backbone

    def predict(self, input_batch):
        """
        Performs the forward-pass for the Faster-RCNN object detector.
        :param input_batch: A tensor representing a batch of images either in
        NHWC or HCHW format.
        :return:
        """
        features = self._backbone(input_batch)
