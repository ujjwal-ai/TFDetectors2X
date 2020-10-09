import os

import tensorflow as tf
from google.protobuf import text_format

from protos import labelmap_pb2


def _validate_label_map(label_map):
    """Checks if a label map is valid.
    Args:
      label_map: StringIntLabelMap to validate.
    Raises:
      ValueError: if label map is invalid.
    """
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
        if (item.id == 0 and item.name != 'background' and
                item.display_name != 'background'):
            raise ValueError(
                'Label map id 0 is reserved for the background label')


def load_labelmap(labelmapfile):
    """
    Loads a labelmap from text file
    :param labelmapfile: Full path to the labelmap text file
    :return: A message of type labelmap_pb2.LabelMap()
    :raises OSError if the labelmap text file was not found.
            ValueError if the labelmap is not valid.
    """
    if not os.path.exists(labelmapfile):
        raise OSError(
            'The labelmap file {} was not found.'.format(labelmapfile))

    with tf.io.gfile.GFile(labelmapfile) as fid:
        labelmap_string = fid.read()
        labelmap = labelmap_pb2.LabelMap()
        text_format.Merge(labelmap_string, labelmap)

    _validate_label_map(labelmap)
    return labelmap


def get_max_label_map_index(label_map):
    """Get maximum index in label map.
    Args:
      label_map: a StringIntLabelMapProto
    Returns:
      an integer
    """
    return max([item.id for item in label_map.item])


def get_labelmap_dict(labelmapfilename, use_display_name=False):
    """
    Returns a dictionary of labelmap categories mapped to their corresponding integer IDs
    :param labelmapfilename: Name of the labelmap file.
    :param use_display_name: If categories' display_name is used as keys.
    :return: A dictionary where keys are labelmap categories' names and values are the corresponding
             integer IDs.
    """
    labelmap = load_labelmap(labelmapfilename)
    labelmap_dict = dict()
    for item in labelmap.item:
        if use_display_name:
            labelmap_dict[item.display_name] = item.id
        else:
            labelmap_dict[item.name] = item.id

    return labelmap_dict
