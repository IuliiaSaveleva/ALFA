import datetime
import argparse
import sys
import numpy as np
import pprint
import os
import pickle

from map_computation import Computation_mAP
from reading_methods import read_detectors_detections, read_imagenames, read_annotations, parse_parameters_json
from ALFA import ALFA


def validate_ALFA(dataset_name, dataset_dir, imagenames, annotations, detectors_detections, alfa_parameters_dict,
                  map_iou_threshold, output_filename=None, weighted_map=False, full_imagenames=None):
    """
    Validate ALFA algorithm

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
        Dict of full detections for different detectors

    alfa_parameters_dict : dict
        Contains list of one element for each parameter:

        tau in the paper, between [0.0, 1.0]
        gamma in the paper, between [0.0, 1.0]
        bounding_box_fusion_method ["MIN", "MAX", "MOST CONFIDENT", "AVERAGE", "WEIGHTED AVERAGE",
        "WEIGHTED AVERAGE FINAL LABEL"]
        class_scores_fusion_method ["MOST CONFIDENT", "AVERAGE", "MULTIPLY"]
        add_empty_detections, if true - low confidence class scores tuple will be added to cluster for each detector, that missed
        epsilon in the paper, between [0.0, 1.0]
        same_labels_only, if true only detections with same class label will be added into same cluster
        confidence_style means how to compute score for object proposal ["LABEL", "ONE MINUS NO OBJECT"]
        use_BC, if true - Bhattacharyya and Jaccard coefficient will be used to compute detections similarity score
        max_1_box_per_detector, if true - only one detection form detector could be added to cluster

    map_iou_threshold : float
        Jaccard coefficient value to compute mAP, between [0, 1]

    output_filename : str
        Pickle to store ALFA output. It has the same format, as base detector\'s pickles

    weighted_map : boolean
            True - compute weighted mAP by part class samples count to all class samples count in dataset
            False - compute ordinary mAP

    full_imagenames: list
            List contains all of dataset filenames, if compute
            weighted map on a part of dataset
    """

    alfa = ALFA()
    map_computation = Computation_mAP(None)

    total_time = 0
    time_count = 0
    alfa_full_detections = []

    print('Running ALFA on dataset...')

    param_idx = alfa_parameters_dict['main_fold']
    if param_idx >= len(alfa_parameters_dict['tau']):
        param_idx = 0

    a = datetime.datetime.now()
    for j in range(len(detectors_detections[0])):
        imagename = detectors_detections[0][j][0]

        bounding_boxes = {}
        labels = {}
        class_scores = {}

        for key in detectors_detections.keys():
            if len(detectors_detections[key][j][1]) > 0:
                bb = detectors_detections[key][j][1]
                l = np.array(detectors_detections[key][j][2])
                cl_sc = np.array(detectors_detections[key][j][3])
                scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
                indices = np.where(scores > alfa_parameters_dict['select_threshold'])[0]
                if len(indices) > 0:
                    bounding_boxes[key] = bb[indices]
                    labels[key] = l[indices]
                    class_scores[key] = cl_sc[indices]

        if bounding_boxes != {}:
            bounding_boxes, labels, class_scores = alfa.ALFA_result(detectors_detections.keys(),
                                                                    bounding_boxes, class_scores,
                                                                    alfa_parameters_dict['tau'][param_idx],
                                                                    alfa_parameters_dict['gamma'][param_idx],
                                                                    alfa_parameters_dict['bounding_box_fusion_method'][param_idx],
                                                                    alfa_parameters_dict['class_scores_fusion_method'][param_idx],
                                                                    alfa_parameters_dict['add_empty_detections'][param_idx],
                                                                    alfa_parameters_dict['epsilon'][param_idx],
                                                                    alfa_parameters_dict['same_labels_only'][param_idx],
                                                                    alfa_parameters_dict['confidence_style'][param_idx],
                                                                    alfa_parameters_dict['use_BC'][param_idx],
                                                                    alfa_parameters_dict['max_1_box_per_detector'][param_idx],
                                                                    alfa_parameters_dict['single'])
            time_count += 1
        else:
            bounding_boxes = np.array([])
            labels = np.array([])
            class_scores = np.array([])

        alfa_full_detections.append((imagename, bounding_boxes, labels, class_scores))

    b = datetime.datetime.now()
    total_time += (b - a).seconds

    if output_filename is not None:
        with open(output_filename, 'wb') as f:
            pickle.dump(alfa_full_detections, f, protocol=2)

    aps, mAP, pr_curves = map_computation.compute_map(dataset_name, dataset_dir, imagenames,
                                            annotations, alfa_full_detections, map_iou_threshold,
                                                      weighted_map, full_imagenames)

    print('Average ALFA time: ', float(total_time) / float(time_count))

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
    parser.add_argument('--alfa_parameters_json', required=True, type=str,
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
                             'single, if true computes ALFA prediction for mAP-s computation refered in paper, '
                             'select_threshold is the confidence threshold for detections, '
                             'detections_filenames list of pickles that store detections for mAP computation')
    parser.add_argument('--map_iou_threshold', type=float,
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5)
    parser.add_argument('--output_filename', type=str,
                        help='Pickle to store ALFA output. It has the same format, as base detector\'s pickles')
    return parser.parse_args(argv)


def main(args):

    alfa_parameters_dict = parse_parameters_json(args.alfa_parameters_json)
    pprint.pprint(alfa_parameters_dict)

    detectors_detections = read_detectors_detections(alfa_parameters_dict['detections_filenames'])

    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.dataset_dir, 'JPEGImages/')

    imagenames = read_imagenames(args.imagenames_filename, images_dir)
    annotations = read_annotations(args.pickled_annots_filename, annotations_dir, imagenames, args.dataset_name)

    validate_ALFA(args.dataset_name, args.dataset_dir, imagenames,
                    annotations, detectors_detections, alfa_parameters_dict, args.map_iou_threshold,
                  args.output_filename)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
