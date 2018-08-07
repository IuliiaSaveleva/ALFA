import argparse
import sys
import os

from map_computation import Computation_mAP
from reading_methods import read_detectors_detections, read_imagenames, read_annotations

def validate_base_detector(dataset_name, dataset_dir, imagenames, annotations, detectors_detections,
                  map_iou_threshold, weighted_map=False, full_imagenames=None):
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

    weighted_map : boolean
            True - compute weighted mAP by part class samples count to all class samples count in dataset
            False - compute ordinary mAP

    full_imagenames: list
            List contains all of dataset filenames, if compute
            weighted map on a part of dataset
    """

    map_computation = Computation_mAP(None)

    key = list(detectors_detections.keys())[0]

    aps, mAP, pr_curves = map_computation.compute_map(dataset_name, dataset_dir, imagenames,
                                            annotations, detectors_detections[key], map_iou_threshold,
                                                      weighted_map, full_imagenames)


    return aps, mAP, pr_curves


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
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5)
    return parser.parse_args(argv)


def main(args):

    detectors_detections = read_detectors_detections([args.detections_filename])

    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.dataset_dir, 'JPEGImages/')

    imagenames = read_imagenames(args.imagenames_filename, images_dir)
    annotations = read_annotations(args.pickled_annots_filename, annotations_dir, imagenames, args.dataset_name)

    validate_base_detector(args.dataset_name, args.dataset_dir, imagenames,
                    annotations, detectors_detections, args.map_iou_threshold)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
