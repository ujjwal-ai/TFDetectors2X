import os
import argparse
from protos import labelmap_pb2

parser = argparse.ArgumentParser(prog='Code to create labelmap for a dataset.')

parser.add_argument('--class_def_file', required=True, type=str,
                    help='File containing class information.')
parser.add_argument('--save_name', required=True, type=str,
                    help='Full path to the filename to be stored as labelmap')

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args['class_def_file']):
        raise OSError('The class definition file {} was not found.'
                      .format(args['class_def_file']))

    items = list()

    for line in open(args['class_def_file'], 'r'):
        line = line.strip()
        line = line.split(',')
        name = line[0]
        display_name = line[1]
        id = line[2]
        items.append(
            labelmap_pb2.LabelMapItem(
                name=name,
                display_name=display_name,
                id=id
            )
        )

    labelmap = labelmap_pb2.LabelMap(
        item=items
    )

    with open(args['save_name'], 'w') as fid:
        fid.write(str(labelmap))
