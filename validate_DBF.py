import datetime
import argparse
import sys
import numpy as np
import pprint
import os
import pickle

from map_computation import Computation_mAP
from reading_methods import read_detectors_detections, read_imagenames, read_annotations, parse_parameters_json
from DBF import DBF


def validate_DBF(dataset_name, validation_dataset_dir, validation_imagenames, validation_annotations,
                 validation_detectors_detections, test_dataset_dir, test_imagenames,
                 test_annotations, test_detectors_detections, dbf_parameters_dict, map_iou_threshold, output_filename=None,
                 weighted_map=False, full_imagenames=None):
    """
    Validate DBF algorithm

    ----------
    dataset_name : string
        Dataset name, e.g. 'PASCAL VOC'

    validation_dataset_dir : string
        Path to validation images dir, e.g.'.../PASCAL VOC/VOC2007 test/VOC2007/'

    validation_imagenames : list
            List contains all or a part of validation dataset filenames

    validation_annotations : dict
        Dict of annotations from validation dataset

    validation_detectors_detections: dict
        Dict of validation detections for different detectors

    test_dataset_dir : string
        Path to test images dir, e.g.'.../PASCAL VOC/VOC2012 test/VOC2012/'

    test_imagenames : list
            List contains all or a part of test dataset filenames

    test_annotations : dict
        Dict of annotations from test dataset

    test_detectors_detections: dict
        Dict of test detections for different detectors

    dbf_parameters_dict : dict
        Contains list of one element for each parameter:

        n in the paper

    map_iou_threshold : float
        Jaccard coefficient value to compute mAP, between [0, 1]

    output_filename : str
        Pickle to store DBF output. It has the same format, as base detector\'s pickles

    weighted_map : boolean
            True - compute weighted mAP by part class samples count to all class samples count in dataset
            False - compute ordinary mAP

    full_imagenames: list
            List contains all of dataset filenames, if compute
            weighted map on a part of dataset
    """

    dbf = DBF(validation_detectors_detections.keys(), dataset_name, validation_dataset_dir, validation_imagenames,
              validation_annotations, validation_detectors_detections, map_iou_threshold)
    map_computation = Computation_mAP(None)

    total_time = 0
    time_count = 0
    dbf_detections = []

    print('Running DBF on dataset...')

    a = datetime.datetime.now()
    for j in range(len(test_detectors_detections[0])):
        imagename = test_detectors_detections[0][j][0]

        bounding_boxes = {}
        labels = {}
        class_scores = {}

        for key in test_detectors_detections.keys():
            if len(test_detectors_detections[key][j][1]) > 0:
                bb = test_detectors_detections[key][j][1]
                l = np.array(test_detectors_detections[key][j][2])
                cl_sc = np.array(test_detectors_detections[key][j][3])
                scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
                indices = np.where(scores > dbf_parameters_dict['select_threshold'])[0]
                if len(indices) > 0:
                    bounding_boxes[key] = bb[indices]
                    labels[key] = l[indices]
                    class_scores[key] = cl_sc[indices]

        if bounding_boxes != {}:
            bounding_boxes, labels, class_scores = dbf.DBF_result(bounding_boxes, class_scores, labels,
                                                                    dbf_parameters_dict['n'])
            time_count += 1
        else:
            bounding_boxes = np.array([])
            labels = np.array([])
            class_scores = np.array([])

        dbf_detections.append((imagename, bounding_boxes, labels, class_scores))

    b = datetime.datetime.now()
    total_time += (b - a).seconds

    if output_filename is not None:
        with open(output_filename, 'wb') as f:
            pickle.dump(dbf_detections, f)

    aps, mAP, pr_curves = map_computation.compute_map(dataset_name, test_dataset_dir, test_imagenames,
                                            test_annotations, dbf_detections, map_iou_threshold,
                                                      weighted_map, full_imagenames)

    print('Average DBF time: ', float(total_time) / float(time_count))

    return aps, mAP, pr_curves


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Only \"PASCAL VOC\" is supported, default=\"PASCAL VOC\"',
                        default='PASCAL VOC')
    parser.add_argument('--validation_dataset_dir', required=True, type=str,
                        help='e.g.=\"(Your path)/PASCAL VOC/VOC2007 test/VOC2007\"')
    parser.add_argument('--validation_imagenames_filename', required=True, type=str,
        help='File where images filenames to compute mAP are stored, e.g.=\"./PASCAL_VOC_files/imagesnames_2007_test.txt\"')
    parser.add_argument('--validation_pickled_annots_filename', type=str,
        help='Pickle where annotations to compute mAP are stored, e.g.=\"./PASCAL_VOC_files/annots_2007_test.pkl\"')
    parser.add_argument('--test_dataset_dir', required=True, type=str,
                        help='e.g.=\"(Your path)/PASCAL VOC/VOC2012 test/VOC2012\"')
    parser.add_argument('--test_imagenames_filename', required=True, type=str,
                        help='File where images filenames to compute mAP are stored, e.g.=\"./PASCAL_VOC_files/imagesnames_2012_test.txt\"')
    parser.add_argument('--test_pickled_annots_filename', type=str,
                        help='Pickle where annotations to compute mAP are stored, e.g.=\"./PASCAL_VOC_files/annots_2012_test.pkl\"')
    parser.add_argument('--dbf_parameters_json', required=True, type=str,
                        help='File from directory \"./Cross_validation_parameters\", that contains parameters:'
                             'n in the paper, '
                             'single, if true computes DBF prediction for mAP-s computation refered in paper, '
                             'select_threshold is the confidence threshold for detections, '
                             'detections_filenames list of pickles that store detections for mAP computation')
    parser.add_argument('--map_iou_threshold', type=float,
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5)
    parser.add_argument('--output_filename', type=str,
                        help='Pickle to store DBF output. It has the same format, as base detector\'s pickles')
    return parser.parse_args(argv)


def main(args):

    dbf_parameters_dict = parse_parameters_json(args.dbf_parameters_json)
    pprint.pprint(dbf_parameters_dict)

    validation_detectors_detections = read_detectors_detections(dbf_parameters_dict['validation_detections_filenames'])

    annotations_dir = os.path.join(args.validation_dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.validation_dataset_dir, 'JPEGImages/')

    validation_imagenames = read_imagenames(args.validation_imagenames_filename, images_dir)
    validation_annotations = read_annotations(args.validation_pickled_annots_filename, annotations_dir,
                                              validation_imagenames, args.dataset_name)

    annotations_dir = os.path.join(args.test_dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.test_dataset_dir, 'JPEGImages/')

    test_imagenames = read_imagenames(args.test_imagenames_filename, images_dir)
    test_annotations = read_annotations(args.test_pickled_annots_filename, annotations_dir,
                                              test_imagenames, args.dataset_name)

    test_detectors_detections = read_detectors_detections(dbf_parameters_dict['test_detections_filenames'])


    validate_DBF(args.dataset_name, args.validation_dataset_dir, validation_imagenames,
                    validation_annotations, validation_detectors_detections, args.test_dataset_dir, test_imagenames,
                 test_annotations, test_detectors_detections, dbf_parameters_dict, args.map_iou_threshold,
                  args.output_filename)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
