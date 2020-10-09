import argparse
import multiprocessing as mp
import os
import cv2
import numpy as np
from utils.labelmap_utils import get_labelmap_dict

import data.tfrecord_db as tfrecord_db
from utils.colormap import colormap
import logging

parser = argparse.ArgumentParser(
    prog='Code to visualize the annotations in TFRecords.')
parser.add_argument('--tfrecord_dir', type=str, required=True,
                    help='Folder containing TFRecord files.')
parser.add_argument('--save_dir', type=str, required=True,
                    help='Folder where to save the visualizations.')
parser.add_argument('--labelmap_file', type=str, required=False,
                    help='Full path to the labelmap file.')

colorlist = colormap()


def plot_bboxes(image, bboxes, category_names):
    for bbox, category_name in zip(bboxes, category_names):
        image = cv2.rectangle(
            image,
            (bbox[1], bbox[0]),
            (bbox[3], bbox[2]),
            (0, 255, 0),
            1
        )
        image = cv2.putText(
            image,
            category_name,
            (bbox[1], bbox[0]),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return image


def plot_instances(image, instances, alpha=0.4, show_border=True,
                   border_thick=1):
    image = image.astype(np.float32)
    for index in range(instances.shape[0]):
        instance = instances[index, :, :]
        color_mask = colorlist[index % len(colorlist), 0:3]
        idx = np.nonzero(instance)
        image[idx[0], idx[1], :] *= 1.0 - alpha
        image[idx[0], idx[1], :] += alpha * color_mask

        if show_border:
            contours = cv2.findContours(
                instance.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
            )[-2]
            cv2.drawContours(image, contours, -1,
                             color=list(map(int, color_mask)),
                             thickness=border_thick, lineType=cv2.LINE_AA)

    return image


def preprocess(serialized_example):
    parsed_features = tfrecord_db.parse_example_train(serialized_example)
    image = tfrecord_db.get_image(parsed_features)
    bboxes = tfrecord_db.convert_bboxes_to_absolute(parsed_features)
    category_names = tfrecord_db.get_category_names(parsed_features)
    image_filename = tfrecord_db.get_filename(parsed_features)
    instance_masks = tfrecord_db.get_instance_masks(parsed_features)
    instance_masks = tfrecord_db.decode_instance_masks(instance_masks,
                                                       parsed_features)
    category_ids = tfrecord_db.get_category_ids(parsed_features)
    return image, bboxes, category_names, category_ids, instance_masks, image_filename


if __name__ == "__main__":
    arguments = parser.parse_args()
    tfrecord_dir = arguments.tfrecord_dir
    save_dir = arguments.save_dir
    use_display_name = False
    if arguments.labelmap_file is not None:
        use_display_name = True
        labelmap_dict_display = get_labelmap_dict(arguments.labelmap_file,
                                                  use_display_name=True)
        labelmap_dict_display_reversed = dict(
            [
                (v, k) for k, v in labelmap_dict_display.items()
            ]
        )

    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(save_dir, 'logfile.log'))
    if not os.path.isdir(tfrecord_dir):
        raise OSError('The path {} was not found.'.format(tfrecord_dir))

    tfdb = tfrecord_db.TFRecordDB(os.path.join(tfrecord_dir, '*.tfrecord'))

    tfdb = tfdb.map(preprocess, num_parallel_calls=mp.cpu_count())

    for image, bboxes, category_names, category_ids, instance_masks, image_filename in iter(
            tfdb):
        image = image.numpy()
        category_names = category_names.numpy()
        category_ids = category_ids.numpy()
        bboxes = bboxes.numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if instance_masks.numpy().size > 1:
            image = plot_instances(image, instance_masks.numpy())
        else:
            logging.info('No masks were found for {}.'.format(image_filename))
        if use_display_name:
            display_names = [labelmap_dict_display_reversed[x] for x in
                             category_ids]
            image = plot_bboxes(image, bboxes, display_names)
        else:
            category_names = [x.decode('utf8') for x in category_names]
            image = plot_bboxes(image, bboxes, category_names)
        image_filename = image_filename.numpy().decode(encoding='utf8')
        image_filename = os.path.join(save_dir, image_filename)
        cv2.imwrite(image_filename, image)
        print('Wrote {}'.format(image_filename))
