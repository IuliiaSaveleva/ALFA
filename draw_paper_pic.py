import sys
import argparse
import pprint
import numpy as np

from visualization import plt_bboxes_different_detectors
from reading_methods import parse_parameters_json, read_detectors_detections, read_one_image, parse_pascal_voc_rec, \
    dataset_classnames
from ALFA import ALFA


def compute_max_overlap(BBGT, bboxes):
    best_overlap_index = -1
    best_overlap = 0.0
    max_overlap_index = -1
    max_overlap = 0.0
    if BBGT.size > 0 and bboxes.size > 0:
        for i in range(len(bboxes)):
            bb = bboxes[i]
            ixmin = np.maximum(BBGT[0], bb[0])
            iymin = np.maximum(BBGT[1], bb[1])
            ixmax = np.minimum(BBGT[2], bb[2])
            iymax = np.minimum(BBGT[3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[2] - BBGT[0] + 1.) *
                   (BBGT[3] - BBGT[1] + 1.) - inters)

            overlap = inters / np.maximum(uni, np.finfo(np.float64).eps)
            if overlap > 0.5 and overlap > best_overlap:
                best_overlap = overlap
                best_overlap_index = i
            if overlap > 0 and overlap > max_overlap:
                max_overlap = overlap
                max_overlap_index = i
    return best_overlap_index, max_overlap_index, best_overlap, max_overlap


def draw_paper_pic(image_filename, annotation_filename, detectors_detections, alfa_parameters_dict):
    """
    Validate ALFA algorithm

    ----------

    image_filename : list
            Path to image from PASCAL VOC 2007

    annotation_filename : list
            Path to annotation from PASCAL VOC 2007

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
    """

    alfa = ALFA()

    param_idx = alfa_parameters_dict['main_fold']
    if param_idx >= len(alfa_parameters_dict['tau']):
        param_idx = 0

    image_index = -1
    keys = list(detectors_detections.keys())

    if len(keys) == 0:
        print('No detectors detections provided!')

    for i in range(len(detectors_detections[keys[0]])):
        if detectors_detections[keys[0]][i][0] in image_filename:
            image_index = i
            break

    bounding_boxes = {}
    labels = {}
    class_scores = {}
    for key in keys:
        if len(detectors_detections[key][image_index][1]) > 0:
            bb = np.array(detectors_detections[key][image_index][1])
            l = np.array(detectors_detections[key][image_index][2])
            cl_sc = np.array(detectors_detections[key][image_index][3])
            scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
            indices = np.where(scores > alfa_parameters_dict['select_threshold'])[0]
            if len(indices) > 0:
                bounding_boxes[key] = bb[indices]
                labels[key] = l[indices]
                class_scores[key] = cl_sc[indices]

    if bounding_boxes != {}:
        bb_ensemble, l_ensemble, cl_sc_ensemble = alfa.ALFA_result(detectors_detections.keys(),
                                                                bounding_boxes, class_scores,
                                                                alfa_parameters_dict['tau'][param_idx],
                                                                alfa_parameters_dict['gamma'][param_idx],
                                                                alfa_parameters_dict['bounding_box_fusion_method'][
                                                                    param_idx],
                                                                alfa_parameters_dict['class_scores_fusion_method'][
                                                                    param_idx],
                                                                alfa_parameters_dict['add_empty_detections'][param_idx],
                                                                alfa_parameters_dict['epsilon'][param_idx],
                                                                alfa_parameters_dict['same_labels_only'][param_idx],
                                                                alfa_parameters_dict['confidence_style'][param_idx],
                                                                alfa_parameters_dict['use_BC'][param_idx],
                                                                alfa_parameters_dict['max_1_box_per_detector'][
                                                                    param_idx],
                                                                alfa_parameters_dict['single'])

    objects, _ = parse_pascal_voc_rec(annotation_filename)
    gt_bboxes = []
    gt_labels = []
    for obj in objects:
        gt_bboxes.append(obj['bbox'])
        gt_labels.append(dataset_classnames['PASCAL VOC'].index(obj['name']))

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    bb_detector = {}
    l_detector = {}
    cl_sc_detector = {}

    for key in keys:
        if key in bounding_boxes:
            bb_detector[key] = bounding_boxes[key]
            l_detector[key] = labels[key]
            cl_sc_detector[key] = class_scores[key]
        else:
            bb_detector[key] = np.array([])
            l_detector[key] = np.array([])
            cl_sc_detector[key] = np.array([])

    unique_classes = np.unique(gt_labels)

    shift = 0.15

    for i in unique_classes:

        indices = np.where(gt_labels == i)[0]
        BBGT = gt_bboxes[indices]
        class_labels = gt_labels[indices]

        for k in range(len(BBGT)):
            bbgt = BBGT[k]
            dict_bounding_boxes = {}
            dict_labels = {}
            dict_scores = {}
            dict_iou = {}
            dict_bounding_boxes['gt'] = np.expand_dims(bbgt, 0)
            dict_labels['gt'] = np.expand_dims(class_labels[k], 0)
            dict_scores['gt'] = np.expand_dims([1.0] * 21, 0)
            dict_iou['gt'] = np.expand_dims(1.0, 0)

            indices = np.where(l_ensemble == i)[0]
            bb = bb_ensemble[indices]
            best_overlap_index, _, best_overlap, _ = compute_max_overlap(bbgt, bb)
            if best_overlap_index == -1:
                continue
            dict_bounding_boxes['ALFA'] = np.expand_dims(bb_ensemble[indices][best_overlap_index], 0)
            dict_labels['ALFA'] = np.expand_dims(l_ensemble[indices][best_overlap_index], 0)
            dict_scores['ALFA'] = np.expand_dims(cl_sc_ensemble[indices][best_overlap_index], 0)
            dict_iou['ALFA'] = np.expand_dims(best_overlap, 0)

            max_overlap_indices_count = 0

            for key in keys:
                indices = np.where(l_detector[key] == i)[0]
                bb = bb_detector[key][indices]
                _, max_overlap_index, _, max_overlap  = compute_max_overlap(bbgt, bb)
                if max_overlap_index > -1:
                    if max_overlap + shift < best_overlap:
                        max_overlap_indices_count += 1
                        dict_bounding_boxes[key] = np.expand_dims(bb_detector[key][indices][max_overlap_index], 0)
                        dict_labels[key] = np.expand_dims(l_detector[key][indices][max_overlap_index], 0)
                        dict_scores[key] = np.expand_dims(cl_sc_detector[key][indices][max_overlap_index], 0)
                        dict_iou[key] = np.expand_dims(max_overlap, 0)

            if best_overlap_index > -1 and max_overlap_indices_count == 2:
                img = read_one_image(image_filename)
                plt_bboxes_different_detectors(img, dict_labels, dict_scores, dict_iou, dict_bounding_boxes)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_filename', type=str,
                        help='Path to image from PASCAL VOC 2007, default=\"./paper_image/007541.jpg\"',
                        default="./paper_image/007541.jpg")
    parser.add_argument('--annotation_filename', type=str,
                        help='Path to annotation from PASCAL VOC 2007, default=\"./paper_image/007541.xml\"',
                        default="./paper_image/007541.xml")
    parser.add_argument('--alfa_parameters_json', required=True, type=str,
                        help='File from directory \"./Algorithm_parameters\", that contains parameters:'
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
    return parser.parse_args(argv)


def main(args):

    alfa_parameters_dict = parse_parameters_json(args.alfa_parameters_json)
    pprint.pprint(alfa_parameters_dict)

    keys = {}
    for i in range(len(alfa_parameters_dict['detections_filenames'])):
        path = alfa_parameters_dict['detections_filenames'][i]
        if 'SSD' in path:
            keys[i] = 'SSD'
        elif 'DeNet' in path:
            keys[i] = 'DeNet'
        elif 'Faster_R-CNN' in path:
            keys[i] = 'Faster R-CNN'

    detectors_detections = read_detectors_detections(alfa_parameters_dict['detections_filenames'])

    new_detectors_detections = {}
    for key in detectors_detections.keys():
        new_detectors_detections[keys[key]] = detectors_detections[key]

    detectors_detections = new_detectors_detections

    draw_paper_pic(args.image_filename, args.annotation_filename, detectors_detections, alfa_parameters_dict)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))