import glob
import io
import os
import re

import cv2
import lvis
import numpy as np
import pandas as pd
from pycocotools import mask
from pycocotools.coco import COCO

from protos import labelmap_pb2
from utils.labelmap_utils import get_labelmap_dict


def create_coco_labelmap(coco_data_dir, save_dir, coco_year='2017'):
    """
    Creates and saves the MSCOCO labelmap
    :param coco_data_dir: Root folder containing the MSCOCO dataset.
    :param save_dir: Folder where the labelmap should be stored.
    :param coco_year: Complete year ( YYYY ) of the MSCOCO dataset.
    :return: None
    """
    if not os.path.isdir(coco_data_dir):
        raise OSError(
            'The COCO data folder {} was not found.'.format(coco_data_dir))

    annotation_file_name = get_coco_annotation_file(coco_data_dir, coco_year,
                                                    'train')

    coco = get_coco_object(annotation_file_name)
    cats = coco.loadCats(coco.getCatIds())
    categories = [x['name'] for x in cats]
    items = list()
    for index, category in enumerate(categories):
        items.append(
            labelmap_pb2.LabelMapItem(
                name=category,
                display_name=category,
                id=index + 1
            )
        )

    labelmap = labelmap_pb2.LabelMap(item=items)

    labelmap_filename = os.path.join(save_dir,
                                     'coco_{}_labelmap.pbtxt'.format(coco_year))
    with open(labelmap_filename, 'w') as fid:
        fid.write(str(labelmap))

    return None


def create_voc_labelmap(devkit_dir, save_dir, pascal_year='2012'):
    """
        Creates and saves the Pascal VOC labelmap
        :param devkit_dir: Root folder containing the Pascal VOC dataset.
        :param save_dir: Folder where the labelmap should be stored.
        :param pascal_year: Complete year ( YYYY ) of the VOC dataset.
        :return: None
        """
    if not os.path.isdir(devkit_dir):
        raise OSError(
            'The Pascal VOC Devkit folder {} was not found.'.format(devkit_dir))

    search_folder = os.path.join(
        devkit_dir,
        'VOC{}'.format(pascal_year),
        'ImageSets',
        'Main'
    )
    categories_files = glob.glob(
        os.path.join(search_folder, '*_val.txt')
    )
    categories_files = list(
        map(lambda x: os.path.basename(x), categories_files))
    REGEX_COMPILED = re.compile(r'^(.*)_val.txt$')
    categories = list(
        map(
            lambda x: re.match(REGEX_COMPILED, x)[1],
            categories_files
        )
    )
    items = list()
    for index, category in enumerate(categories):
        items.append(
            labelmap_pb2.LabelMapItem(
                name=category,
                display_name=category,
                id=index + 1
            )
        )

    labelmap = labelmap_pb2.LabelMap(
        item=items
    )
    labelmap_filename = os.path.join(save_dir, 'VOC_{}_labelmap.pbtxt'.format(
        pascal_year))
    with open(labelmap_filename, 'w') as fid:
        fid.write(str(labelmap))

    return None


def get_coco_annotation_file(coco_dir, coco_year, subset):
    """
    Returns the full path to the annotation file for the MSCOCO dataset.
    :param coco_dir: Root folder where the MSCOCO dataset is extracted.
    :param coco_year: Year in the format YYYY for the MSCOCO Dataset.
    :param subset: one of train, val, test and test-dev
    :return: Full path  to the annotation file. If the subset is in ['train', 'val]
             the instances json filename is returned else, the image_info json
             file is returned.
    """
    if subset in ['train', 'val']:
        annotation_file = os.path.join(
            coco_dir,
            'annotations',
            'instances_{}{}.json'.format(subset, coco_year)
        )
    else:
        annotation_file = os.path.join(
            coco_dir,
            'annotations',
            'image_info_{}{}.json'.format(subset, coco_year)
        )

    return annotation_file


def get_coco_object(annotation_file):
    """
    Returns the pycocotools object from the annotation_file
    :param annotation_file: Full path to the annotation file
    :return: A pycocotools object
    """
    coco = COCO(annotation_file)
    return coco


def get_coco_image_full_path(imagefilename, coco_dir, coco_year, subset):
    """
    Returns the full path to a MSCOCO image.
    :param imagefilename: Base name of the image file.
    :param coco_dir: Base folder where the MSCOCO dataset is extracted.
    :param coco_year: Year in YYYY format representing the MSCOCO dataset.
    :param subset: One of 'train', 'val', 'test' and 'text-dev'
    :return: Full path to the image file.
    """
    if subset == 'test-dev':
        subset = 'test'
    return os.path.join(
        coco_dir,
        '{}{}'.format(subset, coco_year),
        imagefilename
    )


def coco_generator(coco_dir, coco_year, subset):
    annotation_file = get_coco_annotation_file(coco_dir, coco_year, subset)
    coco_obj = get_coco_object(annotation_file)

    imageIDs = coco_obj.getImgIds()
    image_info = coco_obj.loadImgs(imageIDs)
    details = list()
    for image_id, image_details in zip(imageIDs, image_info):
        image_details.pop('license', None)
        image_details.pop('coco_url', None)
        image_details.pop('date_captured', None)
        image_details.pop('flickr_url', None)
        image_full_path = get_coco_image_full_path(image_details['file_name'],
                                                   coco_dir, coco_year, subset)
        image_details['file_name'] = image_full_path
        if subset in ['train', 'val']:
            ann_ids = coco_obj.getAnnIds(image_id)
            anns = coco_obj.loadAnns(ann_ids)
            for index, ann in enumerate(anns):
                cat_id = ann['category_id']
                cat_name = coco_obj.loadCats(cat_id)
                supercategory = cat_name[0]['supercategory']
                category = cat_name[0]['name']
                anns[index]['supercategory'] = supercategory.encode('utf8')
                anns[index]['category'] = category.encode('utf8')
            image_details['annotations'] = anns

        details.append(image_details)

    return details


def encode_coco_mask_as_image(segmentation, image_height, image_width, iscrowd):
    rle_encoding = mask.frPyObjects(segmentation, image_height, image_width)
    binary_mask = mask.decode(rle_encoding)
    if not iscrowd:
        binary_mask = np.amax(binary_mask, axis=2)

    is_success, bytes_buffer = cv2.imencode('.png', binary_mask)
    if not is_success:
        return None

    output_io = io.BytesIO(bytes_buffer)
    return output_io.getvalue()


def get_visible_bbox(segmentation, image_height, image_width, iscrowd):
    rle_encoding = mask.frPyObjects(segmentation, image_height, image_width)
    binary_mask = mask.decode(rle_encoding)
    if not iscrowd:
        binary_mask = np.amax(binary_mask, axis=2)

    mask_indices = np.argwhere(binary_mask != 0)
    if len(mask_indices) == 0:
        return None

    x1 = np.amin(mask_indices[:, 0])
    y1 = np.amin(mask_indices[:, 1])
    x2 = np.amax(mask_indices[:, 0])
    y2 = np.amax(mask_indices[:, 1])

    vis_bbox = [x1, y1, x2, y2]

    return vis_bbox


def get_lvis_object(annotation_file):
    """
    Returns the pycocotools object from the annotation_file
    :param annotation_file: Full path to the annotation file
    :return: A pycocotools object
    """
    lvis_obj = lvis.LVIS(annotation_file)
    return lvis_obj


def get_lvis_annotation_file(lvis_annotation_dir, subset, lvis_version='0.5'):
    if subset in ['train', 'val']:
        annotation_file = os.path.join(
            lvis_annotation_dir,
            'lvis_v{}_{}.json'.format(
                lvis_version,
                subset
            )
        )
    else:
        annotation_file = os.path.join(
            lvis_annotation_dir,
            'lvis_v{}_image_info_test.json'.format(
                lvis_version
            )
        )
    return annotation_file


def lvis_generator(lvis_annotation_dir, coco_dir, coco_year, subset,
                   lvis_version='0.5'):
    annotation_file = get_lvis_annotation_file(lvis_annotation_dir,
                                               subset, lvis_version)
    lvis_obj = get_lvis_object(annotation_file)
    imageIDs = lvis_obj.get_img_ids()
    image_info = lvis_obj.load_imgs(imageIDs)
    details = list()
    for image_id, image_details in zip(imageIDs, image_info):
        image_details.pop('license', None)
        image_details.pop('coco_url', None)
        image_details.pop('date_captured', None)
        image_details.pop('flickr_url', None)
        if image_details['file_name'].startswith('COCO'):
            image_details['file_name'] = \
            re.match(r'^.*_([0-9]+\.jpg)$', image_details['file_name'])[1]

        image_full_path = get_coco_image_full_path(image_details['file_name'],
                                                   coco_dir,
                                                   coco_year,
                                                   subset)
        image_details['file_name'] = image_full_path
        if subset in ['train', 'val']:
            ann_ids = lvis_obj.get_ann_ids([image_id])
            anns = lvis_obj.load_anns(ann_ids)
            for index, ann in enumerate(anns):
                cat_id = ann['category_id']
                cat_name = lvis_obj.load_cats([cat_id])[0]['name']
                anns[index]['supercategory'] = ''.encode('utf8')
                anns[index]['category'] = cat_name.encode('utf8')
                anns[index]['iscrowd'] = 0
            image_details['annotations'] = anns

        details.append(image_details)

    return details


def create_lvis_labelmap(lvis_annotation_dir, save_dir, lvis_version='0.5'):
    annotation_file = get_lvis_annotation_file(lvis_annotation_dir,
                                               'train', lvis_version)
    lvis_obj = get_lvis_object(annotation_file)
    category_dict = lvis_obj.cats
    items = list()
    for class_label in category_dict.keys():
        class_name = category_dict[class_label]['name']
        items.append(
            labelmap_pb2.LabelMapItem(
                name=class_name,
                display_name=class_name,
                id=class_label
            )
        )

    labelmap = labelmap_pb2.LabelMap(item=items)

    labelmap_filename = os.path.join(save_dir,
                                     'lvis_v{}_labelmap.pbtxt'.format(
                                         lvis_version))
    with open(labelmap_filename, 'w') as fid:
        fid.write(str(labelmap))

    return None


def create_openimages_labelmap(openimages_base_dir, save_dir):
    class_descriptions_file = os.path.join(
        openimages_base_dir,
        'class-descriptions-boxable.csv'
    )
    segmentation_classes_file = os.path.join(openimages_base_dir,
                                             'classes-segmentation.txt'
                                             )

    description_dict = dict()
    for line in open(class_descriptions_file, 'r'):
        line = line.strip()
        class_name, display_name = line.split(',')
        description_dict[class_name] = display_name

    segmentation_classes = list()
    for line in open(segmentation_classes_file, 'r'):
        line = line.strip()
        segmentation_classes.append(line)

    items = list()
    for index, segmentation_class in enumerate(segmentation_classes):
        display_name = description_dict[segmentation_class]
        items.append(
            labelmap_pb2.LabelMapItem(
                name=segmentation_class,
                display_name=display_name,
                id=index + 1
            )
        )

    labelmap = labelmap_pb2.LabelMap(item=items)
    labelmap_filename = os.path.join(
        save_dir,
        'openimages_v5_labelmap.pbtxt'
    )
    with open(labelmap_filename, 'w') as fid:
        fid.write(str(labelmap))

    return None


def get_oic_annotation_file(openimages_base_dir, subset):
    if subset == 'val':
        subset = 'validation'
    annotation_file = os.path.join(
        openimages_base_dir,
        '{}-annotations-object-segmentation.csv'.format(subset)
    )
    return annotation_file


def get_oic_full_image_path(image_id, openimages_base_dir, subset):
    if subset == 'val':
        subset = 'validation'

    image_full_path = os.path.join(
        openimages_base_dir,
        subset,
        '{}.jpg'.format(image_id)
    )

    return image_full_path


def get_oic_mask_image(mask_image_name, openimages_base_dir, subset):
    if subset == 'val':
        subset = 'validation'

    first_letter = mask_image_name[0]
    mask_path = os.path.join(
        openimages_base_dir,
        '{}-masks-{}'.format(subset, first_letter),
        mask_image_name
    )
    return mask_path


def oic_generator(openimages_base_dir, subset, labelmap_file):
    annotation_file = get_oic_annotation_file(openimages_base_dir, subset)
    df = pd.read_csv(annotation_file,
                     usecols=['MaskPath', 'ImageID', 'LabelName', 'BoxXMin',
                              'BoxXMax', 'BoxYMin', 'BoxYMax'])
    groupby_imageID = df.groupby(['ImageID'])
    labelmap_dict = get_labelmap_dict(labelmap_file)
    details = list()
    for image_id in df['ImageID']:
        imagefilename = get_oic_full_image_path(image_id, openimages_base_dir,
                                                subset)
        image_dict = dict()
        image_dict['file_name'] = imagefilename
        image_dict['annotations'] = list()
        annotations = groupby_imageID.get_group(image_id)
        for index, annotation in annotations.iterrows():
            mask_path = get_oic_mask_image(annotation['MaskPath'],
                                           openimages_base_dir,
                                           subset)
            class_name = annotation['LabelName']
            bbox = [
                annotation['BoxXMin'],
                annotation['BoxXMax'],
                annotation['BoxYMin'],
                annotation['BoxYMax']
            ]
            class_id = labelmap_dict[class_name]
            image_dict['annotations'].append(
                {
                    'segmentation': mask_path,
                    'category': class_name.encode('utf8'),
                    'supercategory': ''.encode('utf8'),
                    'area': -1.0,
                    'iscrowd': 0,
                    'category_id': class_id,
                    'bbox': bbox

                }
            )
        details.append(image_dict)
    return details
