# (c) Evgeny Razinkov, Kazan Federal University, 2017
import datetime
import pickle

from computation_map import Computation_mAP

import bbox_clustering_another_version_4 as bbox_clustering
from ssd_detector_module.sdd_detector import *
# import tensorflow as tf
#import matplotlib.pyplot as plt
# import ssd
# import yolo2
# import faster_rcnn
from ssd_lib_razinkov import *

# import pyximport
# pyximport.install()

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
SAME_LABELS_ONLY = True
SAME_LABELS_ONLY = False

unique = '_unique'
vanilla = True
detectors_names = ['ssd', 'denet', 'frcnn']
select_threshold = {
    'ssd': 0.05,
    'denet': 0.05,
    'frcnn': 0.05
}

add_empty_detections_values = [True]
confidence_style_values = ['LABEL']
ensemble_bounding_box_values = ['WEIGHTED_AVERAGE_FINAL_LABEL']
ensemble_scores_values = ['MULTIPLY']
iou_threshold_values = [0.7520381588802675]
iou_power_values = [0.28014759247209853]
empty_epsilon_values = [0.169091521845813]
same_labels_only_values = [True]

# add_empty_detections_values = [True]
# confidence_style_values = ['LABEL']
# ensemble_bounding_box_values = ['WEIGHTED_AVERAGE_FINAL_LABEL']
# ensemble_scores_values = ['MULTIPLY']
# iou_threshold_values = [0.4833061430649309]
# iou_power_values = [0.2171683854488066]
# empty_epsilon_values = [0.557035919458682]
# same_labels_only_values = [True]

# add_empty_detections_values = [True]
# confidence_style_values = ['LABEL']
# ensemble_bounding_box_values = ['WEIGHTED_AVERAGE_FINAL_LABEL']
# ensemble_scores_values = ['AVERAGE']
# iou_threshold_values = [0.7266443010390319]
# iou_power_values = [0.2520069952089646]
# empty_epsilon_values = [0.2574298391671611]
# same_labels_only_values = [True]

# add_empty_detections_values = [True]
# confidence_style_values = ['LABEL']
# ensemble_bounding_box_values = ['AVERAGE']
# ensemble_scores_values = ['AVERAGE']
# iou_threshold_values = [0.4484444201087372]
# iou_power_values = [0.6214828696692978]
# empty_epsilon_values = [0.37537011352851984]
# same_labels_only_values = [False]

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
            return self.np_bounding_boxes[np.argmax(self.np_scores)]
        else:
            print('Unknown value for ENSEMBLE_BOUNDING_BOX. Using AVERAGE')
            return self.average_bounding_box()

    def average_scores(self):
        return np.average(self.np_class_scores[:self.effective_scores], axis=0)

    def multiply_scores(self):
        temp = np.prod(np.clip(self.np_class_scores[:self.effective_scores], a_min=self.epsilon, a_max=None), axis=0)
        return temp/np.sum(temp)
        # return temp

    def get_final_class_scores(self):
        if ENSEMBLE_SCORES == 'AVERAGE':
            return self.average_scores()
        elif ENSEMBLE_SCORES == 'MULTIPLY':
            return self.multiply_scores()
        elif ENSEMBLE_SCORES == 'MOST_CONFIDENT':
            return self.np_class_scores[np.argmax(self.np_scores)]
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
        score_sum = np.average(self.np_class_scores, axis=0)
        self.label = np.argmax(score_sum)
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
            if ADD_EMPTY_DETECTIONS:
                self.effective_scores = len(self.np_scores)
            else:
                self.effective_scores = len(self.detected_by)

            # print(self.np_class_scores)
            self.final_class_scores = self.get_final_class_scores()
            # print(self.final_class_scores)
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

        """
        original = {}
        times_used = {}
        old_detectors = []
        new_detectors = list(self.detectors.keys())
        for detector_name in self.detectors.keys():
            original[detector_name] = [True] * len(bounding_boxes[detector_name])
            times_used[detector_name] = [0] * len(bounding_boxes[detector_name])
        for detector_name in self.detectors.keys():
            objects[detector_name] = []
            new_detectors.remove(detector_name)

            for bb in range(len(bounding_boxes[detector_name])):
                if original[detector_name][bb]:
                    objects[detector_name].append(
                        Object(detector_name, bounding_boxes[detector_name][bb], class_scores[detector_name][bb],
                               thresholds))
                    times_used[detector_name][bb] += 1
                    for new_detector_name in new_detectors:
                        for new_bb in range(len(bounding_boxes[new_detector_name])):
                            overlaps = objects[detector_name][-1].check(new_detector_name,
                                                                        bounding_boxes[new_detector_name][new_bb],
                                                                        class_scores[new_detector_name][new_bb])
                            if overlaps:
                                original[new_detector_name][new_bb] = False
                                times_used[new_detector_name][new_bb] += 1
                    for old_detector_name in old_detectors:
                        for old_bb in range(len(bounding_boxes[old_detector_name])):
                            if not original[old_detector_name][old_bb]:
                                overlaps = objects[detector_name][-1].check(old_detector_name,
                                                                            bounding_boxes[old_detector_name][old_bb],
                                                                            class_scores[old_detector_name][old_bb])
                                if overlaps:
                                    times_used[old_detector_name][old_bb] += 1
            old_detectors.append(detector_name)
        """
        bc = bbox_clustering.BoxClustering(bounding_boxes = bounding_boxes, class_scores = class_scores, hard_threshold=IOU_THRESHOLD, power_iou=IOU_POWER, same_labels_only=SAME_LABELS_ONLY, silent = True)
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



def cross_validate_ensemble_parameters(ensemble, map_computation):
    dataset_name = 'PASCAL VOC'
    dataset_dir = '/home/yulia/PycharmProjects/VOC2012 test/VOC2012/'
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

    cross_validation_ensemble_imagenames_filename = os.path.join(cache_dir, 'ssd_results/ssd_imagesnames_2012_test.txt')
    cross_validation_ensemble_annotations_filename = os.path.join(cache_dir, 'ssd_results/ssd_annotations_2012_test.txt')
    cross_validation_ensemble_pickled_annotations_filename = os.path.join(cache_dir,
                                                                  'ssd_results/ssd_annots_2012_test.pkl')
    cross_validation_ensemble_detections_filename = os.path.join(cache_dir,
                            'cross_validation/' +  '_'.join(detectors_names) + unique + '_ensemble_fast_detections_2012_test.txt')
    cross_validation_ensemble_full_detections_filename = os.path.join(cache_dir,
                        'cross_validation/' + '_'.join(detectors_names) + unique + '_ensemble_fast_full_detections_2012_test.pkl')

    ssd_imagenames_filename = os.path.join(cache_dir, 'ssd_results/ssd_imagesnames_2012_test.txt')
    ssd_annotations_filename = os.path.join(cache_dir, 'ssd_results/ssd_annotations_2012_test.txt')
    ssd_pickled_annotations_filename = os.path.join(cache_dir, 'ssd_results/ssd_annots_2012_test.pkl')

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

    ssd_full_detections_filename = os.path.join(cache_dir, 'original_ssd_results/original_ssd_trully_0.015_vanilla_unique_full_detections_2012_test.pkl')
    yolo_full_detections_filename = os.path.join(cache_dir, 'yolo_results/yolo_0.01_full_detections.pkl')
    frcnn_full_detections_filename = os.path.join(cache_dir, 'frcnn_results/frcnn_0.015_vanilla_unique_full_detections_2012_test.pkl')
    denet_full_detections_filename = os.path.join(cache_dir, 'denet_results/denet_0.015_vanilla_unique_full_detections_2012_test.pkl')

    with open(ssd_full_detections_filename, 'rb') as f:
        ssd_full_detections = pickle.load(f)
    with open(yolo_full_detections_filename, 'rb') as f:
        yolo_full_detections = pickle.load(f)
    with open(frcnn_full_detections_filename, 'rb') as f:
        frcnn_full_detections = pickle.load(f)
    with open(denet_full_detections_filename, 'rb') as f:
        denet_full_detections = pickle.load(f)

    for same_labels_only_value in same_labels_only_values:
        for add_empty_detections_value in add_empty_detections_values:
            for confidence_style in confidence_style_values:
                for ensemble_bounding_box_value in ensemble_bounding_box_values:
                    for ensemble_scores_value in ensemble_scores_values:
                        for iou_threshold_value in iou_threshold_values:
                            for iou_power_value in iou_power_values:
                                for empty_epsilon in empty_epsilon_values:
                                    # if os.path.isfile(cross_validation_ensemble_detections_filename):
                                    #     os.remove(cross_validation_ensemble_detections_filename)

                                    ENSEMBLE_BOUNDING_BOX = ensemble_bounding_box_value
                                    print('ENSEMBLE_BOUNDING_BOX:', ENSEMBLE_BOUNDING_BOX)
                                    ENSEMBLE_SCORES = ensemble_scores_value
                                    print('ENSEMBLE_SCORES:', ENSEMBLE_SCORES)
                                    ADD_EMPTY_DETECTIONS = add_empty_detections_value
                                    print('ADD_EMPTY_DETECTIONS:', ADD_EMPTY_DETECTIONS)
                                    IOU_THRESHOLD = iou_threshold_value
                                    print('IOU_THRESHOLD:', IOU_THRESHOLD)
                                    IOU_POWER = iou_power_value
                                    print('IOU_POWER:', IOU_POWER)
                                    CONFIDENCE_STYLE = confidence_style
                                    print('CONFIDENCE_STYLE:', CONFIDENCE_STYLE)
                                    EMPTY_EPSILON = empty_epsilon
                                    print('EMPTY_EPSILON', EMPTY_EPSILON)
                                    SAME_LABELS_ONLY = same_labels_only_value
                                    print('SAME_LABELS_ONLY', SAME_LABELS_ONLY)
                                    total_time = 0
                                    time_count = 0

                                    a = datetime.datetime.now()

                                    full_detections = []
                                    if not os.path.exists(cross_validation_ensemble_detections_filename):
                                        # f = open(cross_validation_ensemble_detections_filename, 'w')
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

                                                if unique == '':

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

                                                else:

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

                                            time_count += 1

                                            full_detections.append((imagename, bounding_boxes, labels, class_scores))
                                            if len(class_scores) > 0:
                                                rscores = scores
                                                # img = read_one_image('/home/yulia/PycharmProjects/PASCAL VOC/VOC2012 test/VOC2012/JPEGImages/' + imagename)
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
                                                    # f.write(result)
                                        # f.close()
                                        with open(cross_validation_ensemble_full_detections_filename, 'wb') as f:
                                            pickle.dump(full_detections, f)

                                    b = datetime.datetime.now()
                                    total_time += (b - a).seconds

                                    # _, mAP, _ = map_computation.compute_map(dataset_name, images_dir, annotations_dir, cache_dir, cross_validation_ensemble_imagenames_filename,
                                    #                             cross_validation_ensemble_annotations_filename, cross_validation_ensemble_pickled_annotations_filename,
                                    #                             cross_validation_ensemble_detections_filename, cross_validation_ensemble_full_detections_filename)
                                    # print('mAP:', mAP)
                                    print('Average ensemble time: ', float(total_time) / time_count)

                                    print('\n\n')


ens = ObjectDetectionEnsemble()

map_computation = Computation_mAP(None)

dataset_name = 'PASCAL VOC'
dataset_dir = '/home/yulia/PycharmProjects/PASCAL VOC/VOC2012 test/VOC2012/'
annotations_dir = os.path.join(dataset_dir, 'Annotations/')
images_dir = os.path.join(dataset_dir, 'JPEGImages/')
cache_dir = './'

cross_validate_ensemble_parameters(ens, map_computation)

# unique 3 detectors mAP = 0.85474512630586064 not best parameters (parameters for 2 detectors were used) SAME_LABELS_ONLY=False
# ('Average ensemble time: ', 0.1421647819063005)

# unique 3 detectors mAP = 0.85464642388164747 not best parameters (parameters for 2 detectors were used) SAME_LABELS_ONLY=True
# ('Average ensemble time: ', 0.1425686591276252)