import os
import pickle
import datetime
import argparse
import sys
import numpy as np

from map_computation import Computation_mAP, dataset_classnames
from ALFA import ALFA

def validate_ensemble(dataset_name, dataset_dir, imagenames_filename, annotations_filename,
                    pickled_annotations_filename, detections_filenames, map_iou_threshold, select_threshold, tau, gamma,
                    bounding_box_fusion_method, class_scores_fusion_method, add_empty_detections, empty_epsilon,
                    same_labels_only, confidence_style, use_BC, max_1_box_per_detector):

    classnames = dataset_classnames[dataset_name]

    alfa = ALFA()
    map_computation = Computation_mAP(None)

    detections_filenames = detections_filenames.split(',')

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

    total_time = 0
    time_count = 0
    alfa_full_detections = []

    a = datetime.datetime.now()
    for j in range(len(detectors_full_detections[0])):
        imagename = detectors_full_detections[0][j][0]

        bounding_boxes = {}
        labels = {}
        class_scores = {}

        for key in detectors_full_detections.keys():
            if len(detectors_full_detections[key][j][1]) > 0:
                bb = detectors_full_detections[key][j][1]
                l = np.array(detectors_full_detections[key][j][2])
                cl_sc = np.array(detectors_full_detections[key][j][3])
                scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
                indices = np.where(scores > select_threshold)[0]
                if len(indices) > 0:
                    bounding_boxes[key] = bb[indices]
                    labels[key] = l[indices]
                    class_scores[key] = cl_sc[indices]

        if bounding_boxes != {}:
            bounding_boxes, labels, class_scores, scores = alfa.ALFA_result(bounding_boxes, class_scores, tau, gamma,
                                                                               bounding_box_fusion_method,
                                                                               class_scores_fusion_method,
                                                                               add_empty_detections, empty_epsilon,
                                                                               same_labels_only, confidence_style,
                                                                               use_BC, max_1_box_per_detector)
            time_count += 1
        else:
            bounding_boxes = np.array([])
            labels = np.array([])
            class_scores = np.array([])
            scores = np.array([])

        alfa_full_detections.append((imagename, bounding_boxes, labels, class_scores))
        # if len(class_scores) > 0:
        #     rscores = scores
        #     # img = read_one_image('/home/yulia/PycharmProjects/PASCAL VOC/VOC2007 test/VOC2007/JPEGImages/' + imagename)
        #     # visualization.plt_bboxes(img, labels, class_scores, bounding_boxes)
        #     for i in range(len(labels)):
        #         label = labels[i]
        #         xmin = bounding_boxes[i, 0]
        #         ymin = bounding_boxes[i, 1]
        #         xmax = bounding_boxes[i, 2]
        #         ymax = bounding_boxes[i, 3]
        #         result = '{imagename} {rclass} {rscore} {xmin} {ymin} {xmax} {ymax}\n'.format(
        #             imagename=imagename, rclass=classnames[label],
        #             rscore=rscores[i], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        #         print(str(j) + '/' + str(len(detectors_full_detections[0])), result)

    b = datetime.datetime.now()
    total_time += (b - a).seconds

    # total_time = float(total_time) / float(1e6)

    _, mAP, _ = map_computation.compute_map(dataset_name, dataset_dir, imagenames_filename, annotations_filename,
                                            pickled_annotations_filename, alfa_full_detections, map_iou_threshold)
    print('Average ensemble time: ', float(total_time) / float(time_count))

    print('\n\n')



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Only \"PASCAL VOC\" is supported, default=\"PASCAL VOC\"', default='PASCAL VOC')
    parser.add_argument('dataset_dir', type=str,
        help='e.g.=\"(Your path)/PASCAL VOC/VOC2007 test/VOC2007\"')
    parser.add_argument('detections_filenames', type=str,
        help='Path to detections pickles, '
             'e.g.=\"./SSD_detections/SSD_ovthresh_0.015_unique_detections_PASCAL_VOC_2007_test.pkl\", '
             '\"./DeNet_detections/DeNet_ovthresh_0.015_unique_detections_PASCAL_VOC_2007_test.pkl\", '
             '\"./Faster_R-CNN_detections/Faster_R-CNN_ovthresh_0.015_unique_detections_PASCAL_VOC_2007_test.pkl\"')
    parser.add_argument('imagenames_filename', type=str,
        help='File where images filenames to compute mAP are stored, e.g.=\"./PASCAL_VOC_pickles/imagesnames_2007_test.txt\"')
    parser.add_argument('annotations_filename', type=str,
        help='File where annotations filenames to compute mAP are stored, e.g.=\"./PASCAL_VOC_pickles/annotations_2007_test.txt\"')
    parser.add_argument('pickled_annots_filename', type=str,
        help='Pickle where annotations to compute mAP are stored, e.g.=\"./PASCAL_VOC_pickles/annots_2007_test.pkl\"')
    parser.add_argument('--map_iou_threshold', type=float,
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5)
    parser.add_argument('select_threshold', type=float,
                        help='Confidence threshold for detections')
    parser.add_argument('tau', type=float,
                        help='Parameter tau in the paper, between [0.0, 1.0]')
    parser.add_argument('gamma', type=float,
                        help='Parameter gamma in the paper, between [0.0, 1.0]')
    parser.add_argument('bounding_box_fusion_method', type=str,
                        help='Bounding box fusion method [\"MIN\", \"MAX\", \"MOST CONFIDENT\", \"AVERAGE\", '
                             '\"WEIGHTED AVERAGE\", \"WEIGHTED AVERAGE FINAL LABEL\"')
    parser.add_argument('class_scores_fusion_method', type=str,
                        help='Bounding box fusion method [\"MOST CONFIDENT\", \"AVERAGE\", \"MULTIPLY\"')
    parser.add_argument('add_empty_detections', type=bool,
                        help='True - low confidence class scores tuple will be added to cluster for each detector, '
                             'that missed, False - low confidence class scores tuple won\'t be added to cluster for '
                             'each detector, that missed')
    parser.add_argument('empty_epsilon', type=float,
                        help='Parameter epsilon in the paper, between [0.0, 1.0]')
    parser.add_argument('same_labels_only', type=bool,
                        help='True - only detections with same class label will be added into same cluster, '
                             'False - detections labels won\'t be taken into account while clustering')
    parser.add_argument('--confidence_style', type=str,
                        help='How to compute score for object proposal ["LABEL", "ONE MINUS NO OBJECT"], '
                             'we used "LABEL" in every experiment, default=\"LABEL\"', default='LABEL')
    parser.add_argument('--use_BC', type=bool,
                        help='True - Bhattacharyya and Jaccard coefficient will be used to compute detections '
                             'similarity score, False - only Jaccard coefficient will be used to compute detections '
                             'similarity score, default=True', default=True)
    parser.add_argument('--max_1_box_per_detector', type=bool,
                        help='True - only one detection form detector could be added to cluster, '
                             'False - multiple detections from same detector could be added to cluster, '
                             'default=True', default=True)
    return parser.parse_args(argv)


def main(args):
    validate_ensemble(args.dataset_name, args.dataset_dir, args.imagenames_filename, args.annotations_filename,
                    args.pickled_annots_filename, args.detections_filenames, args.map_iou_threshold,
                    args.select_threshold, args.tau, args.gamma, args.bounding_box_fusion_method,
                    args.class_scores_fusion_method, args.add_empty_detections, args.empty_epsilon,
                    args.same_labels_only, args.confidence_style, args.use_BC, args.max_1_box_per_detector)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
