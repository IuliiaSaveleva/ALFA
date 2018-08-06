import pickle
import os
import sys
import xml.etree.ElementTree as ET
import json


def parse_pascal_voc_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    imagename = tree.find('filename').text
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        try:
            obj_struct['difficult'] = int(obj.find('difficult').text)
        except:
            obj_struct['difficult'] = 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects, imagename


dataset_parse_functions = {
    'PASCAL VOC': parse_pascal_voc_rec,
}

dataset_classnames = {
    'PASCAL VOC': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
}


def read_imagenames(filename, dir):
    if not os.path.isfile(filename):
        print('Can not find file:', filename)
        imagenames = []
        for (dir, subdirs, files) in os.walk(dir):
            for file in files:
                if not file.startswith('.'):
                    imagenames.append(file)
        f = open(filename, 'w')
        for imagename in imagenames:
            f.write(imagename + '\n')
        f.close()
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
    return imagenames


def read_annotations(filename, dir, imagenames, dataset_name):

    if not os.path.isfile(filename):
        print('Can not find file:', filename)
        annotation_names = []
        for (dir, subdirs, files) in os.walk(dir):
            for file in files:
                if not file.startswith('.') and file.split('.')[0] + '.jpg' in imagenames:
                    annotation_names.append(file)
        # load annots
        recs = {}
        for i, annotation_name in enumerate(annotation_names):
            rec, imagename = dataset_parse_functions[dataset_name](os.path.join(dir, annotation_name))
            if imagename in imagenames:
                recs[imagename] = rec
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(annotation_names)))
        # save
        print('Saving cached annotations to {:s}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(recs, f, protocol=2)
    else:
        # load
        with open(filename, 'rb') as f:
            recs = pickle.load(f)
    return recs


def read_detectors_detections(detections_filenames):

    """
    Method to read detections into dict

    ----------
    detections_filenames : list
        Pickles that store detections for mAP computation
        File contains a list of detections for one detector or fusion result, kept in format:

        (image filename: '00000.png', bounding boxes: [[23, 45, 180, 790], ..., [100, 39, 705, 98]],
            labels: [0, ..., 19], class_scores: [[0.0, 0.01, ...., 0.98], ..., [0.9, 0.0, ..., 0.001]])


    Returns
    -------
    detectors_detections: dict
        Dictionary, containing detections for n detectors:

        '1': [
        (image filename: '00000.png', bounding boxes: [[23, 45, 180, 790], ..., [100, 39, 705, 98]],
            labels: [0, ..., 19], class_scores: [[0.0, 0.01, ...., 0.98], ..., [0.9, 0.0, ..., 0.001]]),
        ...,
        (image filename: '00500.png', bounding boxes: [[32, 54, 81, 97], ..., [1, 93, 507, 890]],
            labels: [0, ..., 19], class_scores: [[0.0, 0.001, ...., 0.97], ..., [0.95, 0.00001, ..., 0.0001]])
        ],
        ...,
        'n': [
        (image filename: '00000.png', bounding boxes: [[33, 55, 180, 800], ..., [110, 49, 715, 108]],
            labels: [0, ..., 19], class_scores: [[0.01, 0.01, ...., 0.98], ..., [0.8, 0.0002, ..., 0.001]]),
        ...,
        (image filename: '00500.png', bounding boxes: [[13, 35, 170, 780], ..., [90, 29, 695, 88]],
            labels: [0, ..., 19], class_scores: [[0.08, 0.2, ...., 0.06], ..., [0.0, 0.0, ..., 1.0]])
        ]
    """

    detectors_full_detections = {}
    for i in range(len(detections_filenames)):
        with open(detections_filenames[i], 'rb') as f:
            if sys.version_info[0] == 3:
                detectors_full_detections[i] = pickle.load(f, encoding='latin1')
            else:
                detectors_full_detections[i] = pickle.load(f)

    if len(list(detectors_full_detections.keys())) == 0:
        print('Detections even for one detector were not provided!')
        exit(1)
    else:
        check_size_sim = True
        for i in range(1, len(list(detectors_full_detections.keys()))):
            if len(detectors_full_detections[i]) != len(detectors_full_detections[0]):
                check_size_sim = False
                break
        if not check_size_sim:
            print('All detections files should provide detections for the same amount of images with same names!')
            exit(1)
    return detectors_full_detections


def parse_parameters_json(parameters_json):
    with open(parameters_json, 'r') as f:
        parameters_dict = json.load(f)
    return parameters_dict


def check_flag(value):
    if value in ['True', 'False']:
        if value == True:
            return True
        else:
            return False
    else:
        raise argparse.ArgumentTypeError('%s is an invalid flag value, use \"True\" or \"False\"!' % value)
