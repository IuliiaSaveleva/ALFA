import os
from sklearn.model_selection import KFold
import numpy as np
import argparse
import pprint
import sys

from map_computation import Computation_mAP
from reading_methods import read_imagenames, read_annotations, read_detectors_detections, parse_parameters_json, \
    get_detections_by_imagenames
from validate_DBF import validate_DBF


def cross_validate_ALFA(dataset_name, dataset_dir, imagenames, annotations, detectors_detections,
                        dbf_parameters_dict, map_iou_threshold, folds_count):
    """
        Validate DBF algorithm

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

        dbf_parameters_dict: dict

            Contains list of size folds count for each parameter:

            n in the paper

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

    for train_index, test_index in kf.split(imagenames):
        print('Fold number:', fold_index)

        training_imagenames = imagenames[train_index]

        test_imagenames = imagenames[test_index]

        detectors_training_detections = {}
        for key in detectors_detections.keys():
            detector_training_detections = get_detections_by_imagenames(detectors_detections[key], training_imagenames)
            detectors_training_detections[key] = detector_training_detections

        detectors_test_detections = {}
        for key in detectors_detections.keys():
            detector_test_detections = get_detections_by_imagenames(detectors_detections[key], test_imagenames)
            detectors_test_detections[key] = detector_test_detections

        aps, _, _ = validate_DBF(dataset_name, dataset_dir, training_imagenames, annotations,
                                 detectors_training_detections, dataset_dir, test_imagenames, annotations,
                                 detectors_test_detections, dbf_parameters_dict, map_iou_threshold, weighted_map=True,
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
    parser.add_argument('--dbf_parameters_json', required=True, type=str,
                        help='File from directory \"./Algorithm_parameters\", that contains parameters:'
                             'n in the paper, '
                             'single, if true computes DBF prediction for mAP-s computation refered in paper, '
                             'select_threshold is the confidence threshold for detections, '
                             'detections_filenames list of pickles that store detections for mAP computation, '
                             'all parametes should be of the same amount as folds count')
    parser.add_argument('--map_iou_threshold', type=float,
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5)
    parser.add_argument('--folds_count', type=int,
                        help='If used, computes DBF prediction for mAP-s computation refered in paper, default=5',
                        default=5)
    return parser.parse_args(argv)


def main(args):

    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.dataset_dir, 'JPEGImages/')

    imagenames = read_imagenames(args.imagenames_filename, images_dir)
    annotations = read_annotations(args.pickled_annots_filename, annotations_dir, imagenames, args.dataset_name)

    dbf_parameters_dict = parse_parameters_json(args.dbf_parameters_json)
    pprint.pprint(dbf_parameters_dict)

    detectors_detections = read_detectors_detections(dbf_parameters_dict['validation_detections_filenames'])

    cross_validate_ALFA(args.dataset_name, args.dataset_dir, imagenames, annotations, detectors_detections,
                        dbf_parameters_dict, args.map_iou_threshold, args.folds_count)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))