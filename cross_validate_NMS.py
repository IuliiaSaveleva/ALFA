import sys
import argparse
import numpy as np
import random
import pprint
import os

from sklearn.model_selection import KFold
from validate_NMS import validate_NMS
from reading_methods import read_imagenames, read_annotations, read_detectors_detections, parse_parameters_json,\
    get_detections_by_imagenames


def cross_validate_NMS(dataset_name, dataset_dir, imagenames, annotations, detectors_detections,
                        nms_parameters_dict, map_iou_threshold, folds_count):
    """
    Validate NMS algorithm

    ----------
    dataset_name : string
        Dataset name, e.g. 'PASCAL VOC'

    dataset_dir : string
        Path to images dir, e.g.'.../PASCAL VOC/VOC2007 test/VOC2007/'

    imagenames : list
            List contains all or a part of dataset filenames

    annotations : dict
        Dict of annotations

    detectors_detections: dict
        Dict of detections for different detectors

    nms_parameters_dict: dict

        Contains list of size folds count for each parameter:

    map_iou_threshold : float
        Jaccard coefficient value to compute mAP, between [0, 1]

    folds_count : integer
        Number of folds to cross-validate
    """

    imagenames = sorted(imagenames)
    imagenames = np.array(imagenames)

    kf = KFold(n_splits=folds_count)
    fold_index = 0
    aps_per_fold = []
    for _, test_index in kf.split(imagenames):
        print('Fold number:', fold_index)
        fold_imagenames = imagenames[test_index]

        detectors_fold_detections = {}
        for key in detectors_detections.keys():
            detector_fold_detections = get_detections_by_imagenames(detectors_detections[key], fold_imagenames)
            detectors_fold_detections[key] = detector_fold_detections

        fold_nms_parameters = {}
        for key in nms_parameters_dict:
            if type(nms_parameters_dict[key]) == list and fold_index < len(nms_parameters_dict[key]):
                fold_nms_parameters[key] = [nms_parameters_dict[key][fold_index]]
            elif type(nms_parameters_dict[key]) != list:
                fold_nms_parameters[key] = nms_parameters_dict[key]

        aps, _, _ = validate_NMS(dataset_name, dataset_dir, fold_imagenames, annotations, detectors_fold_detections,
                                  fold_nms_parameters, map_iou_threshold, weighted_map=True,
                                  full_imagenames=imagenames)
        aps_per_fold.append(aps)
        fold_index += 1
        print('Fold aps:', aps)

    print('Cross validation results:')
    print('Average mAP:', round(np.mean(np.sum(aps_per_fold, 0)) * 100, 2))
    print('Average aps:', ' & '.join([str(round(ap * 100, 2)) for ap in np.sum(aps_per_fold, 0)]))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Only \"PASCAL VOC\" is supported, default=\"PASCAL VOC\"',
                        default='PASCAL VOC')
    parser.add_argument('--dataset_dir', required=True, type=str,
                        help='e.g.=\"(Your path)/PASCAL VOC/VOC2007 test/VOC2007\"')
    parser.add_argument('--imagenames_filename', required=True, type=str,
                        help='File where images filenames to compute mAP are stored, e.g.=\"./PASCAL_VOC_files/imagesnames_2007_test.txt\"')
    parser.add_argument('--pickled_annots_filename', type=str,
                        help='Pickle where annotations to compute mAP are stored, e.g.=\"./PASCAL_VOC_files/annots_2007_test.pkl\"')
    parser.add_argument('--nms_parameters_json', required=True, type=str,
                        help='File from directory \"./Cross_validation_parameters\", that contains parameters:'
                             'tau in the paper, between [0.0, 1.0], '
                             'gamma in the paper, between [0.0, 1.0],'
                             'bounding_box_fusion_method [\"MIN\", \"MAX\", \"MOST CONFIDENT\", \"AVERAGE\", '
                             'class_scores_fusion_method [\"MOST CONFIDENT\", \"AVERAGE\", \"MULTIPLY\", '
                             'add_empty_detections, if true - low confidence class scores tuple will be added to cluster '
                             'for each detector, '
                             'epsilon in the paper, between [0.0, 1.0], '
                             'same_labels_only, if true - only detections with same class label will be added into same '
                             'cluster, '
                             'confidence_style_list ["LABEL", "ONE MINUS NO OBJECT"], '
                             'use_BC, if true - Bhattacharyya and Jaccard coefficient will be used to compute detections '
                             'max_1_box_per_detector, if true - only one detection form detector could be added to cluster, '
                             'single, if true computes NMS prediction for mAP-s computation refered in paper, '
                             'select_threshold is the confidence threshold for detections, '
                             'detections_filenames list of pickles that store detections for mAP computation, '
                             'all parametes should be of the same amount as folds count')
    parser.add_argument('--map_iou_threshold', type=float,
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5)
    parser.add_argument('--folds_count', type=int,
                        help='If used, computes NMS prediction for mAP-s computation refered in paper, default=5',
                        default=5)
    return parser.parse_args(argv)


def main(args):

    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.dataset_dir, 'JPEGImages/')

    imagenames = read_imagenames(args.imagenames_filename, images_dir)
    annotations = read_annotations(args.pickled_annots_filename, annotations_dir, imagenames, args.dataset_name)

    nms_parameters_dict = parse_parameters_json(args.nms_parameters_json)
    pprint.pprint(nms_parameters_dict)

    detectors_detections = read_detectors_detections(nms_parameters_dict['detections_filenames'])

    cross_validate_NMS(args.dataset_name, args.dataset_dir, imagenames, annotations, detectors_detections,
                        nms_parameters_dict, args.map_iou_threshold, args.folds_count)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))