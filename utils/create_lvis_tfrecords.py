import argparse
import logging
import multiprocessing as mp
import os

import numpy as np
import tensorflow as tf

import utils.annotation_utils as ann_utils
import utils.tfrecord_utils as tfrecord_utils

parser = argparse.ArgumentParser(
    prog='Code to create TFRecords for MSCOCO Dataset.')

parser.add_argument('--lvis_annotation_dir', type=str, required=True,
                    help='Path where the LVIS annotations are stored.')
parser.add_argument('--lvis_version', type=str, required=False, default='0.5',
                    help='Version of the LVIS dataset.')
parser.add_argument('--coco_dir', type=str, required=True,
                    help='Root folder where the MSCOCO Dataset has been extracted.')
parser.add_argument('--coco_year', type=str, required=False,
                    default='2017',
                    help='Year in the format YYYY of the MSCOCO Dataset.')
parser.add_argument('--subset', choices=['train', 'val', 'test', 'test-dev'],
                    default='train',
                    help='Subset for which the TFRecords need to be created.')
parser.add_argument('--save_dir', type=str, required=True,
                    help='Path where the TFRecords will be stored.')
parser.add_argument('--numshards', type=int, required=False,
                    default=128,
                    help='Number of shards to split the TFRecords into.')


def parse_arguments():
    args = parser.parse_args()
    output_dict = dict(
        lvis_annotation_dir=args.lvis_annotation_dir,
        lvis_version=args.lvis_version,
        coco_dir=args.coco_dir,
        coco_year=args.coco_year,
        subset=args.subset,
        save_dir=args.save_dir,
        numshards=args.numshards,
    )
    return output_dict


def validate_arguments(arguments):
    if not os.path.isdir(arguments['lvis_annotation_dir']):
        raise OSError(
            'The root folder containing the LVIS annotations {} was not found.'.format(
                arguments['lvis_annotation_dir']))

    if not os.path.isdir(arguments['coco_dir']):
        raise OSError(
            'The root folder containing the COCO dataset {} was not found.'.format(
                arguments['coco_dir']))

    os.makedirs(arguments['save_dir'], exist_ok=True)
    if not len(arguments['coco_year']) == 4:
        raise ValueError(
            'The year of the COCO dataset should be in the format YYYY. Found {}.'.format(
                arguments['coco_year']))

    if arguments['numshards'] <= 0:
        raise ValueError(
            'The argument "--numshards" must be a positive integer.')

    return None


def preprocess_image_details(image_details):
    imagefilename = image_details['file_name']
    image_dict = tfrecord_utils.encode_image(imagefilename)
    if 'annotations' in image_details.keys():
        image_height = image_dict['height']
        image_width = image_dict['width']
        image_dict['annotations'] = list()
        for index, annotation in enumerate(image_details['annotations']):
            bbox = annotation['bbox']
            x1, x2, y1, y2 = [
                bbox[0],
                bbox[0] + bbox[2],
                bbox[1],
                bbox[1] + bbox[3]
            ]
            x1 /= image_width
            x2 /= image_width
            y1 /= image_height
            y2 /= image_height
            bbox = [y1, x1, y2, x2]
            if any([x < 0 or x > 1 for x in bbox]):
                logging.warning(
                    'Skipping one annotation in {} because bounding box coordinates {}'
                    'were not in [0,1] even after normalization'
                    .format(imagefilename, ','.join(list(str(bbox)))))
                continue
            area = annotation['area']
            iscrowd = annotation['iscrowd']
            segmentation = annotation['segmentation']
            segmentation_mask_bytes = ann_utils.encode_coco_mask_as_image(
                segmentation,
                image_height,
                image_width,
                iscrowd)
            if segmentation_mask_bytes is None:
                logging.warning(
                    'Skipping one annotation in {} because there was some problem'
                    'in encoding the corresponding mask.'.format(imagefilename))
                continue
            category = annotation['category']
            supercategory = annotation['supercategory']
            category_id = annotation['category_id']
            annotation_dict = dict(
                bbox=bbox,
                area=area,
                iscrowd=iscrowd,
                segmentation=segmentation_mask_bytes,
                category=category,
                supercategory=supercategory,
                category_id=category_id
            )

            image_dict['annotations'].append(annotation_dict)

    return image_dict


def get_tfrecord_single_example(image_dict):
    xmin = list()
    ymin = list()
    xmax = list()
    ymax = list()
    area = list()
    is_crowd = list()
    category = list()
    supercategory = list()
    segmentation = list()
    category_id = list()
    feature_dict = {
        'image/encoded':
            tfrecord_utils.bytes_feature(image_dict['image']),
        'image/height':
            tfrecord_utils.int64_feature(image_dict['height']),
        'image/width':
            tfrecord_utils.int64_feature(image_dict['width']),
        'image/depth':
            tfrecord_utils.int64_feature(image_dict['depth']),
        'image/filename':
            tfrecord_utils.bytes_feature(image_dict['filename']),
        'image/format':
            tfrecord_utils.bytes_feature(image_dict['extension']),
    }
    if 'annotations' in image_dict.keys():
        for annotation in image_dict['annotations']:
            xmin.append(annotation['bbox'][1])
            ymin.append(annotation['bbox'][0])
            xmax.append(annotation['bbox'][3])
            ymax.append(annotation['bbox'][2])
            area.append(annotation['area'])
            is_crowd.append(annotation['iscrowd'])
            category.append(annotation['category'])
            supercategory.append(annotation['supercategory'])
            segmentation.append(annotation['segmentation'])
            category_id.append(annotation['category_id'])
        feature_dict_annotation = {
            'image/object/bbox/xmin':
                tfrecord_utils.float_list_feature(xmin),
            'image/object/bbox/ymin':
                tfrecord_utils.float_list_feature(ymin),
            'image/object/bbox/xmax':
                tfrecord_utils.float_list_feature(xmax),
            'image/object/bbox/ymax':
                tfrecord_utils.float_list_feature(ymax),
            'image/object/area':
                tfrecord_utils.float_list_feature(area),
            'image/object/is_crowd':
                tfrecord_utils.int64_list_feature(is_crowd),
            'image/object/class/category_text':
                tfrecord_utils.bytes_list_feature(category),
            'image/object/class/supercategory_text':
                tfrecord_utils.bytes_list_feature(supercategory),
            'image/object/mask':
                tfrecord_utils.bytes_list_feature(segmentation),
            'image/object/class/category_id':
                tfrecord_utils.int64_list_feature(category_id)
        }
        feature_dict.update(feature_dict_annotation)

    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    )

    return example


def create_tfrecord(image_details_list, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for image_details in image_details_list:
            image_dict = preprocess_image_details(image_details)
            example = get_tfrecord_single_example(image_dict)
            writer.write(example.SerializeToString())
    logging.info('Record {} written with {} records.'.format(filename, len(
        image_details_list)))
    return None


if __name__ == "__main__":
    arguments = parse_arguments()
    validate_arguments(arguments)
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(arguments['save_dir'],
                                              'logfile.log'))
    ann_utils.create_lvis_labelmap(
        lvis_annotation_dir=arguments['lvis_annotation_dir'],
        save_dir=arguments['save_dir'],
        lvis_version=arguments['lvis_version']
    )

    all_images_details = ann_utils.lvis_generator(
        arguments['lvis_annotation_dir'],
        arguments['coco_dir'],
        arguments['coco_year'],
        arguments['subset'],
        arguments['lvis_version']
    )

    all_images_details_sharded = np.array_split(all_images_details,
                                                arguments['numshards'])

    filenames = [
        os.path.join(
            arguments['save_dir'],
            'lvis_v{}-{}-{}.tfrecord'.format(arguments['lvis_version'],
                                             arguments['subset'],
                                             x
                                             )
        )
        for x in range(1, arguments['numshards'] + 1)
    ]

    pool_args = [
        (x, y) for x, y in zip(all_images_details_sharded, filenames)
    ]
    pool = mp.Pool(processes=mp.cpu_count())
    pool.starmap(create_tfrecord, pool_args)
