# (c) Evgeny Razinkov, Kazan Federal University, 2017
import datetime
import pickle

from computation_map import Computation_mAP
# import pyximport
# pyximport.install()
from sklearn.model_selection import KFold

import bbox_clustering_another_version_4 as bbox_clustering
from computation_weighted_map import Computation_mAP as Computation_mAP_weighted
from ssd_detector_module.sdd_detector import *
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import ssd
# import yolo2
# import faster_rcnn
from ssd_lib_razinkov import *


ENSEMBLE_BOUNDING_BOX = 'MAX'
ENSEMBLE_BOUNDING_BOX = 'MIN'
ENSEMBLE_BOUNDING_BOX = 'AVERAGE'
ENSEMBLE_BOUNDING_BOX = 'WEIGHTED_AVERAGE'
ENSEMBLE_BOUNDING_BOX = 'WEIGHTED_AVERAGE_FINAL_LABEL'
ENSEMBLE_BOUNDING_BOX = 'MOST_CONFIDENT'
ENSEMBLE_SCORES = 'MULTIPLY'
ENSEMBLE_SCORES = 'AVERAGE'
ENSEMBLE_SCORES = 'MOST_CONFIDENT'
ADD_EMPTY_DETECTIONS = False
ADD_EMPTY_DETECTIONS = True
IOU_THRESHOLD = 0.5
IOU_POWER = 0.5
EMPTY_EPSILON = 0.1
CONFIDENCE_STYLE = 'LABEL'
CONFIDENCE_STYLE = 'ONE_MINUS_NO_OBJECT'
SAME_LABELS_ONLY = False
SAME_LABELS_ONLY = True

vanilla = True
detectors_names = ['ssd', 'denet', 'frcnn']
select_threshold = {
    'ssd': 0.015,
    'denet': 0.015,
    'frcnn': 0.015
}

class Object:
    def __init__(self, detector_name, bounding_box, class_scores, thresholds):
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

    def check(self, detector_name, bounding_box, class_scores):
        does_it_overlap = True
        for bbox in self.bounding_boxes:
            if jaccard(bbox, bounding_box) < IOU_THRESHOLD:
                does_it_overlap = False
        if does_it_overlap:
            self.add_info(detector_name, bounding_box, class_scores)
        return does_it_overlap

    def get_final_bounding_box(self):
        if ENSEMBLE_BOUNDING_BOX == 'MAX':
            return self.max_bounding_box()
        elif ENSEMBLE_BOUNDING_BOX == 'MIN':
            return self.min_bounding_box()
        elif ENSEMBLE_BOUNDING_BOX == 'AVERAGE':
            return self.average_bounding_box()
        elif ENSEMBLE_BOUNDING_BOX == 'WEIGHTED_AVERAGE':
            return self.weighted_average_bounding_box()
        elif ENSEMBLE_BOUNDING_BOX == 'WEIGHTED_AVERAGE_FINAL_LABEL':
            return self.weighted_average_final_label_bounding_box()
        elif ENSEMBLE_BOUNDING_BOX == 'MOST_CONFIDENT':
            return self.np_bounding_boxes[np.argmax(self.np_scores[:len(self.np_bounding_boxes)])]
        else:
            print('Unknown value for ENSEMBLE_BOUNDING_BOX. Using AVERAGE')
            return self.average_bounding_box()

    def average_scores(self):
        return np.average(self.np_class_scores[:self.effective_scores], axis=0)

    def multiply_scores(self):
        temp = np.prod(np.clip(self.np_class_scores[:self.effective_scores], a_min=self.epsilon, a_max=None), axis=0)
        return temp/np.sum(temp)

    def get_final_class_scores(self):
        if ENSEMBLE_SCORES == 'AVERAGE':
            return self.average_scores()
        elif ENSEMBLE_SCORES == 'MULTIPLY':
            return self.multiply_scores()
        elif ENSEMBLE_SCORES == 'MOST_CONFIDENT':
            return self.np_class_scores[np.argmax(self.np_scores[:self.effective_scores])]
        else:
            print('Unknown value for ENSEMBLE_SCORES. Using AVERAGE')

    def finalize(self, thresholds):
        self.detected_by_all = True

        for detector in thresholds.keys():
            if detector not in self.detected_by:
                self.detected_by_all = False
                self.class_scores.append(
                    [1.0 - EMPTY_EPSILON] + [float(EMPTY_EPSILON)/(self.number_of_classes - 1)] * (self.number_of_classes - 1))
                    #([1.0/self.number_of_classes]* self.number_of_classes)
                    # ([1.0 - thresholds[detector]] + [thresholds[detector] / (self.number_of_classes - 1.0)] * (
                    # self.number_of_classes - 1))

        self.np_bounding_boxes = np.array(self.bounding_boxes)
        self.np_bounding_boxes = np.reshape(self.np_bounding_boxes,
                                            (len(self.bounding_boxes), len(self.bounding_boxes[0])))
        # print(self.np_bounding_boxes.shape)
        # print(self.np_bounding_boxes[:].shape)
        self.np_class_scores = np.array(self.class_scores)
        # score_sum = np.average(self.np_class_scores, axis=0)
        # self.label = np.argmax(score_sum)
        # self.np_scores = self.np_class_scores[:][self.label]
        if CONFIDENCE_STYLE == 'ONE_MINUS_NO_OBJECT':
            self.np_scores = np.sum(self.np_class_scores[:, 1:], axis=1)
        elif CONFIDENCE_STYLE == 'LABEL':
            self.np_scores = np.amax(self.np_class_scores[:, 1:], axis = 1)

        self.finalized = True


    def add_info(self, detector_name, bounding_box, class_scores):
        self.bounding_boxes.append(bounding_box)
        self.class_scores.append(class_scores)
        self.detected_by.append(detector_name)

    # def add_blank(self, threshold):
    #	self.thresholds_for_blanks.append(threshold)

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
        # lx = np.average(self.np_bounding_boxes[:][0])
        # ly = np.average(self.np_bounding_boxes[:][1])
        # rx = np.average(self.np_bounding_boxes[:][2])
        # ry = np.average(self.np_bounding_boxes[:][3])
        bounding_box = np.average(self.np_bounding_boxes, axis=0)
        # return np.array([lx, ly, rx, ry])
        return bounding_box

    def weighted_average_bounding_box(self):
        # lx = np.average(self.np_bounding_boxes[:][0], weights = self.np_scores)
        # ly = np.average(self.np_bounding_boxes[:][1], weights = self.np_scores)
        # rx = np.average(self.np_bounding_boxes[:][2], weights = self.np_scores)
        # ry = np.average(self.np_bounding_boxes[:][3], weights = self.np_scores)
        bounding_box = np.average(self.np_bounding_boxes, axis=0, weights=self.np_scores[:len(self.np_bounding_boxes)])
        # return np.array([lx, ly, rx, ry])
        return bounding_box

    def weighted_average_final_label_bounding_box(self):
        bounding_box = np.average(self.np_bounding_boxes, axis=0, weights=self.np_class_scores[:len(self.np_bounding_boxes), self.label + 1])
        return bounding_box

    def get_object(self, thresholds):
        if not self.finalized:
            self.finalize(thresholds)
        if len(self.class_scores) > 0:
            self.effective_scores = len(self.detected_by)
            if ADD_EMPTY_DETECTIONS:
                self.effective_scores = len(self.np_class_scores)

            self.final_class_scores = self.get_final_class_scores()

            self.label = np.argmax(self.final_class_scores[1:])

            self.final_bounding_box = self.get_final_bounding_box()

            return self.final_bounding_box, self.final_class_scores, self.label
        else:
            print('Zero objects, mate!')



class ObjectDetectionEnsemble:
    def __init__(self):

        self.detectors = {}
        self.thresholds = {}

        if 'ssd' in detectors_names:
            self.detectors['ssd'] = None
            self.thresholds['ssd'] = select_threshold['ssd']
        if 'denet' in detectors_names:
            self.detectors['denet'] = None
            self.thresholds['denet'] = select_threshold['denet']
        if 'frcnn' in detectors_names:
            self.detectors['frcnn'] = None
            self.thresholds['frcnn'] = select_threshold['frcnn']


    def predict(self, image):
        thresholds = self.thresholds
        bounding_boxes = {}
        labels = {}
        class_scores = {}
        for detector_name in self.detectors.keys():
            print(detector_name)
            bounding_boxes[detector_name], labels[detector_name], class_scores[detector_name] = self.detectors[
                detector_name].predict(image, threshold=thresholds[detector_name])
        return self.ensemble_result(bounding_boxes, class_scores, thresholds)


    def ensemble_result(self, bounding_boxes, class_scores, thresholds):
        objects = []

        bc = bbox_clustering.BoxClustering(bounding_boxes=bounding_boxes, class_scores=class_scores,
                                           hard_threshold=IOU_THRESHOLD, power_iou=IOU_POWER,
                                           same_labels_only=SAME_LABELS_ONLY, silent = True)
        object_boxes, object_detector_names, object_class_scores = bc.get_raw_candidate_objects()

        for i in range(0, len(object_boxes)):
            number_of_boxes = len(object_boxes[i])
            objects.append(
                Object(object_detector_names[i][0],
                       object_boxes[i][0],
                       object_class_scores[i][0],
                       thresholds))
            for j in range(1, number_of_boxes):
                objects[-1].add_info(object_detector_names[i][j],
                                  object_boxes[i][j],
                                  object_class_scores[i][j])

        ensemble_bounding_boxes = []
        ensemble_class_scores = []
        ensemble_labels = []
        # for detector_name in self.detectors.keys():
        #     for detected_object in objects[detector_name]:

        number_of_complete_detections = 0.0
        for detected_object in objects:
            bounding_box, class_scores, label = detected_object.get_object(thresholds)
            if detected_object.detected_by_all:
                number_of_complete_detections += 1.0
            ensemble_bounding_boxes.append(bounding_box)
            ensemble_class_scores.append(class_scores)
            ensemble_labels.append(label)
        ensemble_bounding_boxes = np.array(ensemble_bounding_boxes)
        ensemble_class_scores = np.array(ensemble_class_scores)
        ensemble_labels = np.array(ensemble_labels)
        # for i in range(0, len(ensemble_labels)):
        #     print(max(ensemble_class_scores[i][1:]))
        # with open('times_used_info' + str(datetime.datetime.now()) + '.p', 'wb') as f:
        #    to_save = (self.detectors.keys(), times_used, objects)
        #    pickle.dump(to_save, f)
        # print('Complete detections ratio:', number_of_complete_detections/len(ensemble_bounding_boxes))
        if len(ensemble_bounding_boxes) > 0:
            complete_ratio = number_of_complete_detections/len(ensemble_bounding_boxes)
        else:
            complete_ratio = 0
        return ensemble_bounding_boxes, ensemble_labels, ensemble_class_scores, complete_ratio



def cross_validate_ensemble_parameters(ensemble, map_computation, map_computation_weighted):
    dataset_name = 'PASCAL VOC'
    dataset_dir = '/home/yulia/PycharmProjects/VOC2007 test/VOC2007/'
    annotations_dir = os.path.join(dataset_dir, 'Annotations/')
    images_dir = os.path.join(dataset_dir, 'JPEGImages/')
    cache_dir = './'
    classnames = dataset_classnames[dataset_name]

    global ENSEMBLE_BOUNDING_BOX
    global ENSEMBLE_SCORES
    global ADD_EMPTY_DETECTIONS
    global IOU_THRESHOLD
    global IOU_POWER
    global CONFIDENCE_STYLE
    global EMPTY_EPSILON
    global SAME_LABELS_ONLY

    cross_validation_ensemble_imagenames_filename = os.path.join(cache_dir, 'ssd_results/ssd_imagesnames.txt')
    cross_validation_ensemble_annotations_filename = os.path.join(cache_dir, 'ssd_results/ssd_annotations.txt')
    cross_validation_ensemble_pickled_annotations_filename = os.path.join(cache_dir,
                                                                  'ssd_results/ssd_annots.pkl')
    cross_validation_ensemble_detections_filename = os.path.join(cache_dir,
                            'cross_validation/' +  '_'.join(detectors_names) + '_cross_validation_ensemble_detections.txt')
    cross_validation_ensemble_full_detections_filename = os.path.join(cache_dir,
                        'cross_validation/' + '_'.join(detectors_names) + '_cross_validation_ensemble_full_detections.pkl')

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

    ssd_full_detections_filename = os.path.join(cache_dir, 'original_ssd_results/original_ssd_trully_0.015_vanilla_unique_full_detections.pkl')
    yolo_full_detections_filename = os.path.join(cache_dir, 'yolo_results/yolo_0.01_full_detections.pkl')
    frcnn_full_detections_filename = os.path.join(cache_dir, 'frcnn_results/frcnn_0.015_vanilla_unique_full_detections.pkl')
    denet_full_detections_filename = os.path.join(cache_dir, 'denet_results/denet_0.015_vanilla_unique_full_detections.pkl')

    with open(ssd_full_detections_filename, 'rb') as f:
        ssd_full_detections = pickle.load(f)
    with open(yolo_full_detections_filename, 'rb') as f:
        yolo_full_detections = pickle.load(f)
    with open(frcnn_full_detections_filename, 'rb') as f:
        frcnn_full_detections = pickle.load(f)
    with open(denet_full_detections_filename, 'rb') as f:
        denet_full_detections = pickle.load(f)

    def compute_mAP(ssd_full_detections, frcnn_full_detections,
                              denet_full_detections, yolo_full_detections, imagenames_filename, weighted):
        if os.path.exists(cross_validation_ensemble_detections_filename):
            os.remove(cross_validation_ensemble_detections_filename)
        f = open(cross_validation_ensemble_detections_filename, 'w')
        for j in range(len(ssd_full_detections)):
            imagename = ssd_full_detections[j][0]
            bounding_boxes = {}
            labels = {}
            class_scores = {}
            bounding_boxes_ = np.array([])
            labels_ = np.array([])
            class_scores_ = np.array([])
            if 'ssd' in detectors_names and len(ssd_full_detections[j][1]) > 0:
                bb = ssd_full_detections[j][1]
                l = np.array(ssd_full_detections[j][2])
                cl_sc = ssd_full_detections[j][3]
                scores = np.array([cl_sc[i, 1:][l[i]] for i in range(len(l))])
                indices = np.where(scores > select_threshold['ssd'])[0]
                if len(indices) > 0:
                    bounding_boxes['ssd'] = bb[indices]
                    labels['ssd'] = l[indices]
                    class_scores['ssd'] = cl_sc[indices]
                    bounding_boxes_ = np.concatenate(
                        (bounding_boxes_, bb[indices])) if bounding_boxes_.size else bb[
                        indices]
                    labels_ = np.concatenate(
                        (labels_, l[indices])) if labels_.size else l[
                        indices]
                    class_scores_ = np.concatenate(
                        (class_scores_, cl_sc[indices])) if class_scores_.size else cl_sc[
                        indices]
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
                    bounding_boxes_ = np.concatenate(
                        (bounding_boxes_, bb[indices])) if bounding_boxes_.size else bb[
                        indices]
                    labels_ = np.concatenate(
                        (labels_, l[indices])) if labels_.size else l[
                        indices]
                    class_scores_ = np.concatenate(
                        (class_scores_, cl_sc[indices])) if class_scores_.size else cl_sc[
                        indices]
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
                    bounding_boxes_ = np.concatenate(
                        (bounding_boxes_, bb[indices])) if bounding_boxes_.size else bb[
                        indices]
                    labels_ = np.concatenate(
                        (labels_, l[indices])) if labels_.size else l[
                        indices]
                    class_scores_ = np.concatenate(
                        (class_scores_, cl_sc[indices])) if class_scores_.size else cl_sc[
                        indices]

            if 'ssd' in bounding_boxes or 'denet' in bounding_boxes or 'frcnn' in bounding_boxes:
                bounding_boxes, labels, class_scores, _ = ensemble.ensemble_result(bounding_boxes, class_scores,
                                                                                   ensemble.thresholds)

                # bounding_boxes = bounding_boxes_
                # labels = labels_
                # class_scores = class_scores_

                # scores = np.concatenate(class_scores[:, 1:])
                # bounding_boxes = np.concatenate(
                #     np.stack([bounding_boxes] * 20, axis=1))
                # labels = np.concatenate([range(20)] * len(labels))
                # class_scores = np.concatenate(
                #     np.stack([class_scores] * 20, axis=1))
                #
                # indices = np.where(scores > 0.01)[0]
                # bounding_boxes = bounding_boxes[indices]
                # labels = labels[indices]
                # class_scores = class_scores[indices]
                # scores = scores[indices]

                scores = np.array([class_scores[i, 1:][labels[i]] for i in range(len(class_scores))])

                labels, scores, bounding_boxes, class_scores, _ = np_methods.bboxes_nms(
                    labels, scores, bounding_boxes, class_scores,
                    class_scores, None,
                    nms_threshold=0.5)
            else:
                bounding_boxes = np.array([])
                labels = np.array([])
                class_scores = np.array([])
                scores = np.array([])

            if len(class_scores) > 0:
                rscores = scores
                # img = read_one_image('/home/yulia/PycharmProjects/PASCAL VOC/VOC2007 test/VOC2007/JPEGImages/' + imagename)
                # visualization.plt_bboxes(img, labels, class_scores, bounding_boxes)
                for i in range(len(labels)):
                    xmin = bounding_boxes[i, 0]
                    ymin = bounding_boxes[i, 1]
                    xmax = bounding_boxes[i, 2]
                    ymax = bounding_boxes[i, 3]
                    result = '{imagename} {rclass} {rscore} {xmin} {ymin} {xmax} {ymax}\n'.format(
                        imagename=imagename, rclass=classnames[labels[i]],
                        rscore=rscores[i], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                    # print(str(j) + '/' + str(len(ssd_full_detections)), result)
                    f.write(result)
        f.close()

        if weighted:
            aps = map_computation_weighted.compute_map(dataset_name, images_dir, annotations_dir, cache_dir,
                                                    cross_validation_ensemble_imagenames_filename,
                                                    imagenames_filename,
                                                    cross_validation_ensemble_annotations_filename,
                                                    cross_validation_ensemble_pickled_annotations_filename,
                                                    cross_validation_ensemble_detections_filename,
                                                    cross_validation_ensemble_full_detections_filename)
            return aps
        else:
            aps, mAP, _ = map_computation.compute_map(dataset_name, images_dir, annotations_dir, cache_dir,
                                                      imagenames_filename,
                                                      cross_validation_ensemble_annotations_filename,
                                                      cross_validation_ensemble_pickled_annotations_filename,
                                                      cross_validation_ensemble_detections_filename,
                                                      cross_validation_ensemble_full_detections_filename)

            return mAP

    n_splits = 5
    parameters_per_fold = []
    aps_per_fold = []
    train_part_images_filename = 'cross_validation/' + '_'.join(detectors_names) + '_cross_validation_train_part_imagenames.txt'
    test_part_images_filename = 'cross_validation/' + '_'.join(detectors_names) + '_cross_validation_test_part_imagenames.txt'
    with open(cross_validation_ensemble_imagenames_filename, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    imagenames = np.array(imagenames)
    random.seed(123)
    random_indices = random.sample(range(len(imagenames)), len(imagenames))
    imagenames = imagenames[random_indices]
    kf = KFold(n_splits=n_splits)
    fold_index = 0

    for train_index, test_index in kf.split(imagenames):

        # ensemble_bounding_box_values_ = [['WEIGHTED_AVERAGE_FINAL_LABEL']]
        # # ensemble_bounding_box_values_ = [['MOST_CONFIDENT']]
        # ensemble_scores_values_ = [['MULTIPLY']]
        # # ensemble_scores_values_ = [['MOST_CONFIDENT']]
        # add_empty_detections_values_ = [[True]]
        # # add_empty_detections_values_ = [[False]]
        # iou_threshold_values_ = [[0.7520381588802675]]
        # iou_power_values_ = [[0.28014759247209853]]
        # # iou_power_values_ = [[1.0]]
        # confidence_style_values_ = [['LABEL']]
        # empty_epsilon_values_ = [[0.169091521845813]]
        # same_labels_only_values_ = [[True]]

        # ensemble_bounding_box_values_ = [['WEIGHTED_AVERAGE_FINAL_LABEL']]
        ensemble_bounding_box_values_ = [['MOST_CONFIDENT']]
        # ensemble_scores_values_ = [['MULTIPLY']]
        ensemble_scores_values_ = [['MOST_CONFIDENT']]
        # add_empty_detections_values_ = [[True]]
        add_empty_detections_values_ = [[False]]
        iou_threshold_values_ = [[0.7]]
        # iou_power_values_ = [[0.4]]
        iou_power_values_ = [[1.0]]
        confidence_style_values_ = [['LABEL']]
        empty_epsilon_values_ = [[0.4]]
        same_labels_only_values_ = [[True]]

        print('Training fold number:', fold_index)
        best_fold_parameters = {}
        best_fold_mAP = 0.0
        train_imagenames = imagenames[train_index]
        test_imagenames = imagenames[test_index]

        def write_imagenames(imagenames, imagenames_file):
            if os.path.isfile(imagenames_file):
                os.remove(imagenames_file)
            f = open(imagenames_file, 'w')
            for imagename in imagenames:
                f.write(imagename + '\n')
            f.close()

        write_imagenames(train_imagenames, train_part_images_filename)
        write_imagenames(test_imagenames, test_part_images_filename)

        def get_splitted_detections(full_detections):
            train_full_detections = []
            test_full_detections = []
            for detection in full_detections:
                if detection[0] in train_imagenames:
                    train_full_detections.append(detection)
                elif detection[0] in test_imagenames:
                    test_full_detections.append(detection)
            return train_full_detections, test_full_detections

        ssd_train_full_detections, ssd_test_full_detections = get_splitted_detections(ssd_full_detections)
        frcnn_train_full_detections, frcnn_test_full_detections = get_splitted_detections(frcnn_full_detections)
        denet_train_full_detections, denet_test_full_detections = get_splitted_detections(denet_full_detections)
        yolo_train_full_detections, yolo_test_full_detections = get_splitted_detections(yolo_full_detections)

        indicator = False
        ind = 1

        block_count = 1

        for parameters_index in range(block_count):

            print('Block', parameters_index)

            ensemble_bounding_box_values = ensemble_bounding_box_values_[parameters_index]
            ensemble_scores_values = ensemble_scores_values_[parameters_index]
            add_empty_detections_values = add_empty_detections_values_[parameters_index]
            iou_threshold_values = iou_threshold_values_[parameters_index]
            iou_power_values = iou_power_values_[parameters_index]
            confidence_style_values = confidence_style_values_[parameters_index]
            empty_epsilon_values = empty_epsilon_values_[parameters_index]
            same_labels_only_values = same_labels_only_values_[parameters_index]

            for same_labels_only_value in same_labels_only_values:
                for ensemble_bounding_box_value in ensemble_bounding_box_values:
                    for ensemble_scores_value in ensemble_scores_values:
                        for add_empty_detections_value in add_empty_detections_values:
                            for iou_threshold_value in iou_threshold_values:
                                for iou_power_value in iou_power_values:
                                    for confidence_style in confidence_style_values:
                                        for empty_epsilon in empty_epsilon_values:

                                            if ind == 0:
                                                a = datetime.datetime.now()

                                            ENSEMBLE_BOUNDING_BOX = ensemble_bounding_box_value
                                            ENSEMBLE_SCORES = ensemble_scores_value
                                            ADD_EMPTY_DETECTIONS = add_empty_detections_value
                                            IOU_THRESHOLD = iou_threshold_value
                                            IOU_POWER = iou_power_value
                                            CONFIDENCE_STYLE = confidence_style
                                            EMPTY_EPSILON = empty_epsilon
                                            SAME_LABELS_ONLY = same_labels_only_value

                                            if indicator:

                                                mAP = compute_mAP(ssd_train_full_detections, frcnn_train_full_detections,
                                                                  denet_train_full_detections, yolo_train_full_detections,
                                                                     train_part_images_filename, False)

                                                if mAP > best_fold_mAP:
                                                    best_fold_mAP = mAP
                                                    best_fold_parameters['ENSEMBLE_BOUNDING_BOX'] = ENSEMBLE_BOUNDING_BOX
                                                    best_fold_parameters['ENSEMBLE_SCORES'] = ENSEMBLE_SCORES
                                                    best_fold_parameters['ADD_EMPTY_DETECTIONS'] = ADD_EMPTY_DETECTIONS
                                                    best_fold_parameters['IOU_THRESHOLD'] = IOU_THRESHOLD
                                                    best_fold_parameters['IOU_POWER'] = IOU_POWER
                                                    best_fold_parameters['CONFIDENCE_STYLE'] = CONFIDENCE_STYLE
                                                    best_fold_parameters['EMPTY_EPSILON'] = EMPTY_EPSILON
                                                    best_fold_parameters['SAME_LABELS_ONLY'] = SAME_LABELS_ONLY
                                                    print('Recent fold best mAP:', best_fold_mAP)
                                                    print('With parameters:', best_fold_parameters)
                                                else:
                                                    dict = {}
                                                    dict['ENSEMBLE_BOUNDING_BOX'] = ENSEMBLE_BOUNDING_BOX
                                                    dict['ENSEMBLE_SCORES'] = ENSEMBLE_SCORES
                                                    dict['ADD_EMPTY_DETECTIONS'] = ADD_EMPTY_DETECTIONS
                                                    dict['IOU_THRESHOLD'] = IOU_THRESHOLD
                                                    dict['IOU_POWER'] = IOU_POWER
                                                    dict['CONFIDENCE_STYLE'] = CONFIDENCE_STYLE
                                                    dict['EMPTY_EPSILON'] = EMPTY_EPSILON
                                                    dict['SAME_LABELS_ONLY'] = SAME_LABELS_ONLY
                                                    print('NOT best mAP:', mAP)
                                                    print('With parameters:', dict)
                                            else:
                                                best_fold_parameters['ENSEMBLE_BOUNDING_BOX'] = ENSEMBLE_BOUNDING_BOX
                                                best_fold_parameters['ENSEMBLE_SCORES'] = ENSEMBLE_SCORES
                                                best_fold_parameters['ADD_EMPTY_DETECTIONS'] = ADD_EMPTY_DETECTIONS
                                                best_fold_parameters['IOU_THRESHOLD'] = IOU_THRESHOLD
                                                best_fold_parameters['IOU_POWER'] = IOU_POWER
                                                best_fold_parameters['CONFIDENCE_STYLE'] = CONFIDENCE_STYLE
                                                best_fold_parameters['EMPTY_EPSILON'] = EMPTY_EPSILON
                                                best_fold_parameters['SAME_LABELS_ONLY'] = SAME_LABELS_ONLY

                                            if ind == 0:
                                                ind = 1
                                                b = datetime.datetime.now()
                                                total_time = (b - a).seconds
                                                print('Estimated one validation time:', total_time)

            ENSEMBLE_BOUNDING_BOX = best_fold_parameters['ENSEMBLE_BOUNDING_BOX']
            ENSEMBLE_SCORES = best_fold_parameters['ENSEMBLE_SCORES']
            ADD_EMPTY_DETECTIONS = best_fold_parameters['ADD_EMPTY_DETECTIONS']
            IOU_THRESHOLD = best_fold_parameters['IOU_THRESHOLD']
            IOU_POWER = best_fold_parameters['IOU_POWER']
            CONFIDENCE_STYLE = best_fold_parameters['CONFIDENCE_STYLE']
            EMPTY_EPSILON = best_fold_parameters['EMPTY_EPSILON']
            SAME_LABELS_ONLY = best_fold_parameters['SAME_LABELS_ONLY']

            # if parameters_index == 0:
            #     add_empty_detections_values_.append([ADD_EMPTY_DETECTIONS])
            #     iou_power_values_.append([IOU_POWER])
            #     iou_threshold_values_.append([IOU_THRESHOLD])
            #     ensemble_scores_values_.append([ENSEMBLE_SCORES])
            #     empty_epsilon_values_.append([EMPTY_EPSILON])
            #     ensemble_bounding_box_values_.append(['MOST_CONFIDENT', 'AVERAGE', 'WEIGHTED_AVERAGE', 'WEIGHTED_AVERAGE_FINAL_LABEL'])
            #     print('Best block same labels only', SAME_LABELS_ONLY)
            #     confidence_style_values_.append([CONFIDENCE_STYLE])
            # if parameters_index == 1:
            #     add_empty_detections_values_.append([ADD_EMPTY_DETECTIONS])
            #     iou_power_values_.append([IOU_POWER])
            #     iou_threshold_values_.append([0.4, 0.5, 0.6, 0.7, 0.8])
            #     ensemble_scores_values_.append([ENSEMBLE_SCORES])
            #     empty_epsilon_values_.append([EMPTY_EPSILON])
            #     ensemble_bounding_box_values_.append([ENSEMBLE_BOUNDING_BOX])
            #     print('Best block bounding box', ENSEMBLE_BOUNDING_BOX)
            #     confidence_style_values_.append([CONFIDENCE_STYLE])
            # elif parameters_index == 2:
            #     add_empty_detections_values_.append([ADD_EMPTY_DETECTIONS])
            #     iou_power_values_.append([0.1, 0.2, 0.3, 0.4, 0.5])
            #     iou_threshold_values_.append([IOU_THRESHOLD])
            #     ensemble_scores_values_.append([ENSEMBLE_SCORES])
            #     empty_epsilon_values_.append([EMPTY_EPSILON])
            #     ensemble_bounding_box_values_.append([ENSEMBLE_BOUNDING_BOX])
            #     print('Best block iou threshold', ENSEMBLE_BOUNDING_BOX)
            #     confidence_style_values_.append([CONFIDENCE_STYLE])
            # elif parameters_index == 3:
            #     add_empty_detections_values_.append([ADD_EMPTY_DETECTIONS])
            #     iou_power_values_.append([IOU_POWER])
            #     iou_threshold_values_.append([IOU_THRESHOLD])
            #     ensemble_scores_values_.append(['MOST_CONFIDENT', 'AVERAGE', 'MULTIPLY'])
            #     empty_epsilon_values_.append([EMPTY_EPSILON])
            #     ensemble_bounding_box_values_.append([ENSEMBLE_BOUNDING_BOX])
            #     print('Best block iou power', IOU_THRESHOLD)
            #     confidence_style_values_.append([CONFIDENCE_STYLE])
            # elif parameters_index == 4:
            #     add_empty_detections_values_.append([ADD_EMPTY_DETECTIONS])
            #     iou_power_values_.append([IOU_POWER])
            #     iou_threshold_values_.append([IOU_THRESHOLD])
            #     ensemble_scores_values_.append([ENSEMBLE_SCORES])
            #     empty_epsilon_values_.append([0.1, 0.2, 0.3, 0.4, 0.5])
            #     ensemble_bounding_box_values_.append([ENSEMBLE_BOUNDING_BOX])
            #     print('Best block ensemble scores', ENSEMBLE_SCORES)
            #     confidence_style_values_.append([CONFIDENCE_STYLE])
            if parameters_index == 0:
                # print('Best block empty epsilon value', EMPTY_EPSILON)
                print('Testing fold number:', fold_index)
                aps = compute_mAP(ssd_test_full_detections, frcnn_test_full_detections,
                                  denet_test_full_detections, yolo_test_full_detections, test_part_images_filename,
                                  True)
                parameters_per_fold.append(best_fold_parameters)
                aps_per_fold.append(aps)
                fold_index += 1
                print('Fold aps:', aps)
                print('With parameters:', best_fold_parameters)

    print('FINAL RESULTS:')
    print('Folds parametes:')
    for parameters_per_fold_ in parameters_per_fold:
        print('ENSEMBLE_BOUNDING_BOX:', parameters_per_fold_['ENSEMBLE_BOUNDING_BOX'])
        print('ENSEMBLE_SCORES:', parameters_per_fold_['ENSEMBLE_SCORES'])
        print('ADD_EMPTY_DETECTIONS:', parameters_per_fold_['ADD_EMPTY_DETECTIONS'])
        print('IOU_THRESHOLD:', parameters_per_fold_['IOU_THRESHOLD'])
        print('IOU_POWER:', parameters_per_fold_['IOU_POWER'])
        print('CONFIDENCE_STYLE:', parameters_per_fold_['CONFIDENCE_STYLE'])
        print('EMPTY_EPSILON', parameters_per_fold_['EMPTY_EPSILON'])
        print('SAME_LABELS_ONLY', parameters_per_fold_['SAME_LABELS_ONLY'])
        print('\n')
    print('Average mAP:', round(np.mean(np.sum(aps_per_fold, 0)) * 100, 2))
    print('Average aps:', ' & '.join([str(round(ap * 100, 2)) for ap in np.sum(aps_per_fold, 0)]))




ens = ObjectDetectionEnsemble()
# Test ensemble on one image

# boxes, labels, scores = ens.predict(img)
# visualization.plt_bboxes(img, labels, scores, boxes)

# ssd = SSD_detector()

# os.chdir('./yad2k/')
# yolo = Yolov2()
# os.chdir('..')

# frcnn = afrcnn.FrcnnEnsemble()
# frcnn.needShowOutput = False

map_computation = Computation_mAP(None)
map_computation_weighted = Computation_mAP_weighted(None)

dataset_name = 'PASCAL VOC'
dataset_dir = '/home/yulia/PycharmProjects/PASCAL VOC/VOC2007 test/VOC2007/'
annotations_dir = os.path.join(dataset_dir, 'Annotations/')
images_dir = os.path.join(dataset_dir, 'JPEGImages/')
cache_dir = './'

# map_computation.compute_map(dataset_name, images_dir, annotations_dir, cache_dir, 'ssd_imagesnames.txt', 'ssd_annotations.txt',
#                             'ssd_annots.pkl', 'frcnn_detections_logits.txt', 'frcnn_full_detections_logits.pkl')

cross_validate_ensemble_parameters(ens, map_computation, map_computation_weighted)
