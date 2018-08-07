import argparse
import sys
import os
import  numpy as np
from sklearn.model_selection import KFold

from map_computation import Computation_mAP
from reading_methods import read_detectors_detections, read_imagenames, read_annotations, get_detections_by_imagenames
from validate_base_detector import validate_base_detector

def cross_validate_base_detector(dataset_name, dataset_dir, imagenames, annotations, detectors_detections,
                  map_iou_threshold, folds_count):
    """
    Validate base detector

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
        Dict of full detections for detector

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

        aps, _, _ = validate_base_detector(dataset_name, dataset_dir, fold_imagenames, annotations,
                                           detectors_fold_detections, map_iou_threshold, weighted_map=True,
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
    parser.add_argument('--detections_filename', required=True, type=str,
        help='Pickle that store detections for mAP computation')
    parser.add_argument('--map_iou_threshold', type=float,
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5),
    parser.add_argument('--folds_count', type=int,
                        help='If used, computes ALFA prediction for mAP-s computation refered in paper, default=5',
                        default=5)
    return parser.parse_args(argv)


def main(args):

    detectors_detections = read_detectors_detections([args.detections_filename])

    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.dataset_dir, 'JPEGImages/')

    imagenames = read_imagenames(args.imagenames_filename, images_dir)
    annotations = read_annotations(args.pickled_annots_filename, annotations_dir, imagenames, args.dataset_name)

    cross_validate_base_detector(args.dataset_name, args.dataset_dir, imagenames,
                    annotations, detectors_detections, args.map_iou_threshold, args.folds_count)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
