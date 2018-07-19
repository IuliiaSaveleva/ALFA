# (c) Evgeny Razinkov, Kazan Federal University, 2017
import os
import pickle
import datetime
import numpy as np
import argparse
import sys

import bbox_clustering as bbox_clustering
from read_image import read_one_image
from map_computation import Computation_mAP, dataset_classnames
import visualization
from NMS import bboxes_nms


class Object:
    def __init__(self, detector_name, bounding_box, class_scores, ensemble_bounding_box, ensemble_scores,
                 add_empty_detections, empty_epsilon, confidence_style):
        self.ensemble_bounding_box = ensemble_bounding_box
        self.ensemble_scores = ensemble_scores
        self.add_empty_detections = add_empty_detections
        self.empty_epsilon = empty_epsilon
        self.confidence_style = confidence_style

        self.bounding_boxes = []
        self.class_scores = []
        self.thresholds_for_blanks = []
        self.detected_by = []
        self.number_of_classes = len(class_scores)
        self.bounding_boxes.append(bounding_box)
        self.class_scores.append(class_scores)
        self.detected_by.append(detector_name)
        self.epsilon = 0.0
        self.finalized = False


    def get_final_bounding_box(self):
        if self.ensemble_bounding_box == 'MAX':
            return self.max_bounding_box()
        elif self.ensemble_bounding_box == 'MIN':
            return self.min_bounding_box()
        elif self.ensemble_bounding_box == 'AVERAGE':
            return self.average_bounding_box()
        elif self.ensemble_bounding_box == 'WEIGHTED_AVERAGE':
            return self.weighted_average_bounding_box()
        elif self.ensemble_bounding_box == 'WEIGHTED_AVERAGE_FINAL_LABEL':
            return self.weighted_average_final_label_bounding_box()
        elif self.ensemble_bounding_box == 'MOST_CONFIDENT':
            return self.np_bounding_boxes[np.argmax(self.np_scores)]
        else:
            print('Unknown value for ensemble_bounding_box. Using AVERAGE')
            return self.average_bounding_box()


    def average_scores(self):
        return np.average(self.np_class_scores[:self.effective_scores], axis=0)


    def multiply_scores(self):
        temp = np.prod(np.clip(self.np_class_scores[:self.effective_scores], a_min=self.epsilon, a_max=None), axis=0)
        return temp/np.sum(temp)


    def get_final_class_scores(self):
        if self.ensemble_scores == 'AVERAGE':
            return self.average_scores()
        elif self.ensemble_scores == 'MULTIPLY':
            return self.multiply_scores()
        elif self.ensemble_scores == 'MOST_CONFIDENT':
            return self.np_class_scores[np.argmax(self.np_scores)]
        else:
            print('Unknown value for ensemble_scores. Using AVERAGE')


    def finalize(self, detectors_names):
        self.detected_by_all = True
        for detector in detectors_names:
            if detector not in self.detected_by:
                self.detected_by_all = False
                self.class_scores.append(
                    [1.0 - self.empty_epsilon] + [float(self.empty_epsilon)/(self.number_of_classes - 1)] * (self.number_of_classes - 1))

        self.np_bounding_boxes = np.array(self.bounding_boxes)
        self.np_bounding_boxes = np.reshape(self.np_bounding_boxes,
                                            (len(self.bounding_boxes), len(self.bounding_boxes[0])))

        self.np_class_scores = np.array(self.class_scores)
        score_sum = np.average(self.np_class_scores, axis=0)
        self.label = np.argmax(score_sum)
        if self.confidence_style == 'ONE_MINUS_NO_OBJECT':
            self.np_scores = np.sum(self.np_class_scores[:, 1:], axis=1)
        elif self.confidence_style == 'LABEL':
            self.np_scores = np.amax(self.np_class_scores[:, 1:], axis=1)

        self.finalized = True


    def add_info(self, detector_name, bounding_box, class_scores):
        self.bounding_boxes.append(bounding_box)
        self.class_scores.append(class_scores)
        self.detected_by.append(detector_name)


    def max_bounding_box(self):
        lx = np.amin(self.np_bounding_boxes[:, 0])
        ly = np.amin(self.np_bounding_boxes[:, 1])
        rx = np.amax(self.np_bounding_boxes[:, 2])
        ry = np.amax(self.np_bounding_boxes[:, 3])
        return np.array([lx, ly, rx, ry])


    def min_bounding_box(self):
        lx = np.amax(self.np_bounding_boxes[:, 0])
        ly = np.amax(self.np_bounding_boxes[:, 1])
        rx = np.amin(self.np_bounding_boxes[:, 2])
        ry = np.amin(self.np_bounding_boxes[:, 3])
        return np.array([lx, ly, rx, ry])


    def average_bounding_box(self):
        bounding_box = np.average(self.np_bounding_boxes, axis=0)
        return bounding_box


    def weighted_average_bounding_box(self):
        bounding_box = np.average(self.np_bounding_boxes, axis=0, weights=self.np_scores[:len(self.np_bounding_boxes)])
        return bounding_box


    def weighted_average_final_label_bounding_box(self):
        bounding_box = np.average(self.np_bounding_boxes, axis=0, weights=self.np_class_scores[:len(self.np_bounding_boxes), self.label + 1])
        return bounding_box


    def get_object(self, thresholds):
        if not self.finalized:
            self.finalize(thresholds)
        if len(self.class_scores) > 0:
            if self.add_empty_detections:
                self.effective_scores = len(self.np_scores)
            else:
                self.effective_scores = len(self.detected_by)

            self.final_class_scores = self.get_final_class_scores()
            self.label = np.argmax(self.final_class_scores[1:])
            self.final_bounding_box = self.get_final_bounding_box()

            return self.final_bounding_box, self.final_class_scores, self.label
        else:
            print('Zero objects, mate!')


class ObjectDetectionEnsemble:
    def __init__(self):
        pass


    def ensemble_result(self, bounding_boxes, class_scores, iou_threhsold, iou_power,
                        ensemble_bounding_box, ensemble_scores, add_empty_detections, empty_epsilon, confidence_style):

        bc = bbox_clustering.BoxClustering(bounding_boxes=bounding_boxes, class_scores=class_scores,
                                           hard_threshold=iou_threhsold, power_iou=iou_power)
        object_boxes, object_detector_names, object_class_scores = bc.get_raw_candidate_objects()

        objects = []
        for i in range(0, len(object_boxes)):
            number_of_boxes = len(object_boxes[i])
            objects.append(
                Object(object_detector_names[i][0],
                       object_boxes[i][0],
                       object_class_scores[i][0], ensemble_bounding_box, ensemble_scores, add_empty_detections,
                       empty_epsilon, confidence_style))
            for j in range(1, number_of_boxes):
                objects[-1].add_info(object_detector_names[i][j],
                                  object_boxes[i][j],
                                  object_class_scores[i][j])

        ensemble_bounding_boxes = []
        ensemble_class_scores = []
        ensemble_labels = []

        for detected_object in objects:
            bounding_box, class_scores, label = detected_object.get_object(bounding_boxes.keys())
            ensemble_bounding_boxes.append(bounding_box)
            ensemble_class_scores.append(class_scores)
            ensemble_labels.append(label)
        ensemble_bounding_boxes = np.array(ensemble_bounding_boxes)
        ensemble_class_scores = np.array(ensemble_class_scores)
        ensemble_labels = np.array(ensemble_labels)

        return ensemble_bounding_boxes, ensemble_labels, ensemble_class_scores


detectors_names = ['ssd', 'denet', 'frcnn']
select_threshold = {
    'ssd': 0.015,
    'denet': 0.015,
    'frcnn': 0.015
}


def validate_ensemble(args):
    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(dataset_dir, 'JPEGImages/')
    classnames = dataset_classnames['PASCAL VOC']

    ensemble = ObjectDetectionEnsemble()
    map_computation = Computation_mAP(None)

    global ENSEMBLE_BOUNDING_BOX
    global ENSEMBLE_SCORES
    global ADD_EMPTY_DETECTIONS
    global IOU_THRESHOLD
    global IOU_POWER
    global CONFIDENCE_STYLE
    global EMPTY_EPSILON

    ENSEMBLE_BOUNDING_BOX = args.ensemble_bounding_box
    print('ENSEMBLE_BOUNDING_BOX:', ENSEMBLE_BOUNDING_BOX)
    ENSEMBLE_SCORES = args.ensemble_scores
    print('ENSEMBLE_SCORES:', ENSEMBLE_SCORES)
    ADD_EMPTY_DETECTIONS = args.add_empty_detections
    print('ADD_EMPTY_DETECTIONS:', ADD_EMPTY_DETECTIONS)
    IOU_THRESHOLD = args.iou_threshold
    print('IOU_THRESHOLD:', IOU_THRESHOLD)
    IOU_POWER = args.iou_power
    print('IOU_POWER:', IOU_POWER)
    CONFIDENCE_STYLE = args.confidence_style
    print('CONFIDENCE_STYLE:', CONFIDENCE_STYLE)
    EMPTY_EPSILON = args.empty_epsilon
    print('EMPTY_EPSILON', EMPTY_EPSILON)

    total_time = 0
    time_count = 0

    validation_imagenames_filename = args.imagenames_filename
    validation_annotations_filename = args.annotations_filename
    validation_pickled_annotations_filename = os.path.join(cache_dir,
                                                                  'ssd_results/ssd_annots.pkl')
    cross_validation_ensemble_detections_filename = os.path.join(cache_dir,
                        'cross_validation/' + '_'.join(detectors_names) + '_validation_ensemble_detections.pkl')

    ssd_imagenames_filename = os.path.join(cache_dir, 'ssd_results/ssd_imagesnames.txt')
    ssd_annotations_filename = os.path.join(cache_dir, 'ssd_results/ssd_annotations.txt')
    ssd_pickled_annotations_filename = os.path.join(cache_dir, 'ssd_results/ssd_annots.pkl')

    if not os.path.exists(cross_validation_ensemble_imagenames_filename):
        with open(ssd_imagenames_filename, 'r') as f:
            content = f.read()
        with open(cross_validation_ensemble_imagenames_filename, 'w') as f:
            f.write(content)

    if not os.path.exists(cross_validation_ensemble_annotations_filename):
        with open(ssd_annotations_filename, 'r') as f:
            content = f.read()
        with open(cross_validation_ensemble_annotations_filename, 'w') as f:
            f.write(content)

    if not os.path.exists(cross_validation_ensemble_pickled_annotations_filename):
        with open(ssd_pickled_annotations_filename, 'rb') as f:
            content = pickle.load(f)
        with open(cross_validation_ensemble_pickled_annotations_filename, 'wb') as f:
            pickle.dump(content, f)

    ssd_full_detections_filename = os.path.join(cache_dir, 'original_ssd_results/SSD_ovthresh_0.015_unique_detections_PASCAL_VOC_2007_test.pkl')
    yolo_full_detections_filename = os.path.join(cache_dir, 'yolo_results/yolo_0.01_full_detections.pkl')
    frcnn_full_detections_filename = os.path.join(cache_dir, 'frcnn_results/Faster_R-CNN_ovthresh_0.015_unique_detections_PASCAL_VOC_2007_test.pkl')
    denet_full_detections_filename = os.path.join(cache_dir, 'denet_results/DeNet_ovthresh_0.015_unique_detections_PASCAL_VOC_2007_test.pkl')

    with open(ssd_full_detections_filename, 'rb') as f:
        ssd_full_detections = pickle.load(f)
    with open(yolo_full_detections_filename, 'rb') as f:
        yolo_full_detections = pickle.load(f)
    with open(frcnn_full_detections_filename, 'rb') as f:
        frcnn_full_detections = pickle.load(f)
    with open(denet_full_detections_filename, 'rb') as f:
        denet_full_detections = pickle.load(f)

    a = datetime.datetime.now()

    full_detections = []
    if not os.path.exists(cross_validation_ensemble_detections_filename):
        f = open(cross_validation_ensemble_detections_filename, 'w')
        for j in range(len(denet_full_detections)):
            imagename = denet_full_detections[j][0]

            bounding_boxes = {}
            labels = {}
            class_scores = {}

            if 'ssd' in detectors_names and len(ssd_full_detections[j][1]) > 0:
                bb = ssd_full_detections[j][1]
                l = np.array(ssd_full_detections[j][2])
                cl_sc = np.array(ssd_full_detections[j][3])
                scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
                indices = np.where(scores > select_threshold['ssd'])[0]
                if len(indices) > 0:
                    bounding_boxes['ssd'] = bb[indices]
                    labels['ssd'] = l[indices]
                    class_scores['ssd'] = cl_sc[indices]
                    if not vanilla:
                        labels['ssd'] = np.argmax(class_scores['ssd'][:, 1:], 1)
            if 'frcnn' in detectors_names and len(frcnn_full_detections[j][1]) > 0:
                bb = frcnn_full_detections[j][1]
                l = frcnn_full_detections[j][2]
                cl_sc = frcnn_full_detections[j][3]
                scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
                indices = np.where(scores > select_threshold['frcnn'])[0]
                if len(indices) > 0:
                    bounding_boxes['frcnn'] = bb[indices]
                    labels['frcnn'] = l[indices]
                    class_scores['frcnn'] = cl_sc[indices]
                    if not vanilla:
                        labels['frcnn'] = np.argmax(class_scores['frcnn'][:, 1:], 1)
            if 'denet' in detectors_names and len(denet_full_detections[j][1]) > 0:
                bb = denet_full_detections[j][1]
                l = denet_full_detections[j][2]
                cl_sc = denet_full_detections[j][3]
                scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
                indices = np.where(scores > select_threshold['denet'])[0]
                if len(indices) > 0:
                    bounding_boxes['denet'] = bb[indices]
                    labels['denet'] = l[indices]
                    class_scores['denet'] = cl_sc[indices]
                    if not vanilla:
                        labels['denet'] = np.argmax(class_scores['denet'][:, 1:], 1)

            if 'ssd' in bounding_boxes or 'denet' in bounding_boxes or 'frcnn' in bounding_boxes:
                bounding_boxes, labels, class_scores, _ = ensemble.ensemble_result(bounding_boxes, class_scores,
                                                                          ensemble.thresholds)

                scores = np.concatenate(class_scores[:, 1:])
                bounding_boxes = np.concatenate(
                    np.stack([bounding_boxes] * 20, axis=1))
                labels = np.concatenate([range(20)] * len(labels))
                class_scores = np.concatenate(
                    np.stack([class_scores] * 20, axis=1))

                indices = np.where(scores > 0.01)[0]
                bounding_boxes = bounding_boxes[indices]
                labels = labels[indices]
                class_scores = class_scores[indices]
                scores = scores[indices]

                # scores = np.array([class_scores[i, 1:][labels[i]] for i in range(len(class_scores))])

                labels, scores, bounding_boxes, class_scores, _ = bboxes_nms(
                    labels, scores, bounding_boxes, class_scores,
                    class_scores, None,
                    nms_threshold=0.5)
            else:
                bounding_boxes = np.array([])
                labels = np.array([])
                class_scores = np.array([])
                scores = np.array([])

            time_count += 1

            full_detections.append((imagename, bounding_boxes, labels, class_scores))
            if len(class_scores) > 0:
                rscores = scores
                # img = read_one_image('/home/yulia/PycharmProjects/PASCAL VOC/VOC2007 test/VOC2007/JPEGImages/' + imagename)
                # visualization.plt_bboxes(img, labels, class_scores, bounding_boxes)
                for i in range(len(labels)):
                    label = labels[i]
                    xmin = bounding_boxes[i, 0]
                    ymin = bounding_boxes[i, 1]
                    xmax = bounding_boxes[i, 2]
                    ymax = bounding_boxes[i, 3]
                    result = '{imagename} {rclass} {rscore} {xmin} {ymin} {xmax} {ymax}\n'.format(
                        imagename=imagename, rclass=classnames[label],
                        rscore=rscores[i], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                    print(str(j) + '/' + str(len(denet_full_detections)), result)
                    f.write(result)
        f.close()
        with open(cross_validation_ensemble_full_detections_filename, 'wb') as f:
            pickle.dump(full_detections, f)

    b = datetime.datetime.now()
    total_time += (b - a).seconds

    _, mAP, _ = map_computation.compute_map(dataset_name, images_dir, annotations_dir, cache_dir, cross_validation_ensemble_imagenames_filename,
                                cross_validation_ensemble_annotations_filename, cross_validation_ensemble_pickled_annotations_filename,
                                cross_validation_ensemble_detections_filename, cross_validation_ensemble_full_detections_filename)
    print('mAP:', mAP)
    print('Average ensemble time: ', float(total_time) / time_count)

    print('\n\n')



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
        help='\"(Your path)/PASCAL VOC/VOC2007 test/VOC2007\"')
    parser.add_argument('--detections_filename', type=str,
        help='Path to detections pickle, default=\"./SSD_detections/SSD_ovthresh_0.015_unique_detections_PASCAL_VOC_2007_test.pkl\"',
        default='./SSD_detections/SSD_ovthresh_0.015_unique_detections_PASCAL_VOC_2007_test.pkl')
    parser.add_argument('--imagenames_filename', type=str,
        help='File where images filenames to compute mAP are stored, default=\"./PASCAL_VOC_pickles/imagesnames_2007_test.txt\"',
        default='./PASCAL_VOC_pickles/imagesnames_2007_test.txt')
    parser.add_argument('--annotations_filename', type=str,
        help='File where annotations filenames to compute mAP are stored, default=\"./PASCAL_VOC_pickles/annotations_2007_test.txt\"',
        default='./PASCAL_VOC_pickles/annotations_2007_test.txt')
    parser.add_argument('--pickled_annots_filename', type=str,
        help='Pickle where annotations to compute mAP are stored, default=\"./PASCAL_VOC_pickles/annots_2007_test.pkl\"',
        default='./PASCAL_VOC_pickles/annots_2007_test.pkl')
    return parser.parse_args(argv)


def main(args):
    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.dataset_dir, 'JPEGImages/')

    map_computation = Computation_mAP(None)
    map_computation.compute_map('PASCAL VOC', images_dir, annotations_dir,
                                args.imagenames_filename, args.annotations_filename,
                                args.pickled_annots_filename, args.detections_filename)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



dataset_name = 'PASCAL VOC'
dataset_dir = '/home/yulia/PycharmProjects/PASCAL VOC/VOC2007 test/VOC2007/'
annotations_dir = os.path.join(dataset_dir, 'Annotations/')
images_dir = os.path.join(dataset_dir, 'JPEGImages/')
cache_dir = './'

cross_validate_ensemble_parameters(ens, map_computation)
