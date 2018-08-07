import datetime
import argparse
import sys
import numpy as np
import pprint
import os
import pickle

from map_computation import Computation_mAP
from reading_methods import read_detectors_detections, read_imagenames, read_annotations, parse_parameters_json
from NMS import bboxes_nms


def validate_NMS(dataset_name, dataset_dir, imagenames, annotations, detectors_detections, nms_parameters_dict,
                 map_iou_threshold, output_filename=None, weighted_map=False, full_imagenames=None):
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
        Dict of full detections for different detectors

    nms_parameters_dict : dict
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
        Pickle to store NMS output. It has the same format, as base detector\'s pickles

    weighted_map : boolean
            True - compute weighted mAP by part class samples count to all class samples count in dataset
            False - compute ordinary mAP

    full_imagenames: list
            List contains all of dataset filenames, if compute
            weighted map on a part of dataset
    """

    map_computation = Computation_mAP(None)

    total_time = 0
    time_count = 0
    nms_full_detections = []

    print('Running NMS on dataset...')

    a = datetime.datetime.now()
    for j in range(len(detectors_detections[0])):
        imagename = detectors_detections[0][j][0]

        bounding_boxes = np.array([])
        labels = np.array([])
        class_scores = np.array([])

        for key in detectors_detections.keys():
            if len(detectors_detections[key][j][1]) > 0:
                bb = detectors_detections[key][j][1]
                l = np.array(detectors_detections[key][j][2])
                cl_sc = np.array(detectors_detections[key][j][3])
                scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
                indices = np.where(scores > nms_parameters_dict['select_threshold'])[0]
                if len(indices) > 0:
                    bounding_boxes = np.concatenate(
                        (bounding_boxes, bb[indices])) if bounding_boxes.size else bb[
                        indices]
                    labels = np.concatenate(
                        (labels, l[indices])) if labels.size else l[
                        indices]
                    class_scores = np.concatenate(
                        (class_scores, cl_sc[indices])) if class_scores.size else cl_sc[
                        indices]

        if len(bounding_boxes) > 0:
            scores = np.array([class_scores[i, 1:][labels[i]] for i in range(len(class_scores))])

            labels, scores, bounding_boxes, class_scores, _ = bboxes_nms(
                labels, scores, bounding_boxes, class_scores,
                class_scores, None,
                nms_threshold=0.5)

            time_count += 1

        nms_full_detections.append((imagename, bounding_boxes, labels, class_scores))

    b = datetime.datetime.now()
    total_time += (b - a).seconds

    if output_filename is not None:
        with open(output_filename, 'wb') as f:
            pickle.dump(nms_full_detections, f, protocol=2)

    aps, mAP, pr_curves = map_computation.compute_map(dataset_name, dataset_dir, imagenames,
                                            annotations, nms_full_detections, map_iou_threshold,
                                                      weighted_map, full_imagenames)

    print('Average NMS time: ', float(total_time) / float(time_count))

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
                             'detections_filenames list of pickles that store detections for mAP computation')
    parser.add_argument('--map_iou_threshold', type=float,
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5)
    parser.add_argument('--output_filename', type=str,
                        help='Pickle to store NMS output. It has the same format, as base detector\'s pickles')
    return parser.parse_args(argv)


def main(args):

    nms_parameters_dict = parse_parameters_json(args.nms_parameters_json)
    pprint.pprint(nms_parameters_dict)

    detectors_detections = read_detectors_detections(nms_parameters_dict['detections_filenames'])

    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.dataset_dir, 'JPEGImages/')

    imagenames = read_imagenames(args.imagenames_filename, images_dir)
    annotations = read_annotations(args.pickled_annots_filename, annotations_dir, imagenames, args.dataset_name)

    validate_NMS(args.dataset_name, args.dataset_dir, imagenames,
                 annotations, detectors_detections, nms_parameters_dict, args.map_iou_threshold,
                 args.output_filename)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
