# (c) Evgeny Razinkov, Kazan Federal University, 2017
import os
import pickle
import random
# import pyximport
# pyximport.install()
import datetime
import math


import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import ssd
# import yolo2
# import faster_rcnn
from ssd_lib_razinkov import *
from read_image import read_one_image

from ssd_detector_module.sdd_detector import *
from ssd_detector_module.notebooks import visualization
from computation_map import Computation_mAP

detectors_names_list = ['ssd', 'denet']
class DynamicBeliefFusion:
    def __init__(self):

        self.epsilon = 0.0001

        self.dataset_name = 'PASCAL VOC'
        self.dataset_dir = '/home/yulia/PycharmProjects/VOC2007 test/VOC2007/'
        self.annotations_dir = os.path.join(self.dataset_dir, 'Annotations/')
        self.images_dir = os.path.join(self.dataset_dir, 'JPEGImages/')
        self.cache_dir = './'
        self.classnames = dataset_classnames[self.dataset_name]

        self.ssd_info_filename = os.path.join(self.cache_dir, 'dbf_results/pr_curves/1original_ssd_info.pkl')
        self.denet_info_filename = os.path.join(self.cache_dir, 'dbf_results/pr_curves/1denet_info.pkl')
        self.frcnn_info_filename = os.path.join(self.cache_dir, 'dbf_results/pr_curves/1frcnn_info.pkl')

        self.prepare_files()

        self.map_computation = Computation_mAP(None)

        self.info = {}

        if 'ssd' in detectors_names_list:
            if not os.path.exists(self.ssd_info_filename):
                _, _, ssd_info = self.map_computation.compute_map(self.dataset_name, self.images_dir, self.annotations_dir, self.cache_dir,
                                                       self.dbf_validation_imagenames_filename,
                                                       self.dbf_validation_annotations_filename,
                                                       self.dbf_validation_pickled_annotations_filename,
                                                       self.ssd_validation_detections_filename,
                                                       self.ssd_full_detections_filename)
                with open(self.ssd_info_filename, 'wb') as f:
                    pickle.dump(ssd_info, f)
            else:
                with open(self.ssd_info_filename, 'rb') as f:
                    ssd_info = pickle.load(f)
            self.info['ssd'] = ssd_info

        if 'denet' in detectors_names_list:
            if not os.path.exists(self.denet_info_filename):
                _, _, denet_info = self.map_computation.compute_map(self.dataset_name, self.images_dir, self.annotations_dir, self.cache_dir,
                                                       self.dbf_validation_imagenames_filename,
                                                       self.dbf_validation_annotations_filename,
                                                       self.dbf_validation_pickled_annotations_filename,
                                                       self.denet_validation_detections_filename,
                                                       self.denet_full_detections_filename)
                with open(self.denet_info_filename, 'wb') as f:
                    pickle.dump(denet_info, f)
            else:
                with open(self.denet_info_filename, 'rb') as f:
                    denet_info = pickle.load(f)
            self.info['denet'] = denet_info

        if 'frcnn' in detectors_names_list:
            if not os.path.exists(self.frcnn_info_filename):
                _, _, frcnn_info = self.map_computation.compute_map(self.dataset_name, self.images_dir, self.annotations_dir, self.cache_dir,
                                                       self.dbf_validation_imagenames_filename,
                                                       self.dbf_validation_annotations_filename,
                                                       self.dbf_validation_pickled_annotations_filename,
                                                       self.frcnn_validation_detections_filename,
                                                       self.frcnn_full_detections_filename)
                with open(self.frcnn_info_filename, 'wb') as f:
                    pickle.dump(frcnn_info, f)
            else:
                with open(self.frcnn_info_filename, 'rb') as f:
                    frcnn_info = pickle.load(f)
            self.info['frcnn'] = frcnn_info


    def prepare_files(self):

        self.dbf_validation_imagenames_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_validation__imagenames.txt')
        self.dbf_validation_annotations_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_validation_annotations.txt')
        self.dbf_validation_pickled_annotations_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_validation_annots.pkl')
        self.dbf_validation_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_' + '_'.join(detectors_names_list) + '_validation_detections.txt')
        self.dbf_validation_full_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_' + '_'.join(detectors_names_list) + '_validation_full_detections.pkl')

        self.dbf_test_imagenames_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_test_imagenames.txt')
        self.dbf_test_annotations_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_test_annotations.txt')
        self.dbf_test_pickled_annotations_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_test_annots.pkl')
        self.dbf_test_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_' + '_'.join(detectors_names_list) + '_test_detections.txt')
        self.dbf_test_full_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1dbf_' + '_'.join(detectors_names_list) + '_test_full_detections.pkl')

        ssd_imagenames_filename = os.path.join(self.cache_dir, 'ssd_results/ssd_imagesnames.txt')
        ssd_annotations_filename = os.path.join(self.cache_dir, 'ssd_results/ssd_annotations.txt')
        ssd_pickled_annotations_filename = os.path.join(self.cache_dir, 'ssd_results/ssd_annots.pkl')

        ssd_detections_filename = os.path.join(self.cache_dir, 'original_ssd_results/original_ssd_trully_0.015_vanilla_unique_detections.txt')
        denet_detections_filename = os.path.join(self.cache_dir, 'denet_results/denet_0.015_vanilla_unique_detections.txt')
        frcnn_detections_filename = os.path.join(self.cache_dir, 'frcnn_results/frcnn_0.015_vanilla_unique_detections.txt')

        self.ssd_validation_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1original_ssd_validation_detections.txt')
        self.denet_validation_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1denet_validation_detections.txt')
        self.frcnn_validation_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1frcnn_validation_detections.txt')

        self.ssd_test_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1original_ssd_test_detections.txt')
        self.denet_test_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1denet_test_detections.txt')
        self.frcnn_test_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1frcnn_test_detections.txt')

        self.ssd_full_detections_filename = os.path.join(self.cache_dir, 'original_ssd_results/original_ssd_trully_0.015_vanilla_unique_full_detections.pkl')
        self.denet_full_detections_filename = os.path.join(self.cache_dir, 'denet_results/denet_0.015_vanilla_unique_full_detections.pkl')
        self.frcnn_full_detections_filename = os.path.join(self.cache_dir, 'frcnn_results/frcnn_0.015_vanilla_unique_full_detections.pkl')

        self.ssd_validation_full_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1original_ssd_validation_full_detections.pkl')
        self.denet_validation_full_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1denet_validation_full_detections.pkl')
        self.frcnn_validation_full_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1frcnn_validation_full_detections.pkl')

        self.ssd_test_full_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1original_ssd_test_full_detections.pkl')
        self.denet_test_full_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1denet_test_full_detections.pkl')
        self.frcnn_test_full_detections_filename = os.path.join(self.cache_dir, 'dbf_results/1frcnn_test_full_detections.pkl')

        if os.path.exists(self.dbf_validation_imagenames_filename):
            os.remove(self.dbf_validation_imagenames_filename)

        if not (os.path.exists(self.dbf_validation_imagenames_filename) and os.path.exists(self.dbf_test_imagenames_filename)
                and os.path.exists(self.dbf_validation_annotations_filename) and os.path.exists(self.dbf_test_annotations_filename)
                and os.path.exists(self.dbf_validation_pickled_annotations_filename) and os.path.exists(self.dbf_test_pickled_annotations_filename)):

            if os.path.exists(self.dbf_validation_imagenames_filename):
                os.remove(self.dbf_validation_imagenames_filename)
            if os.path.exists(self.dbf_test_imagenames_filename):
                os.remove(self.dbf_test_imagenames_filename)
            if os.path.exists(self.dbf_validation_annotations_filename):
                os.remove(self.dbf_validation_annotations_filename)
            if os.path.exists(self.dbf_test_annotations_filename):
                os.remove(self.dbf_test_annotations_filename)
            if os.path.exists(self.dbf_validation_pickled_annotations_filename):
                os.remove(self.dbf_validation_pickled_annotations_filename)
            if os.path.exists(self.dbf_test_pickled_annotations_filename):
                os.remove(self.dbf_test_pickled_annotations_filename)
            if os.path.exists(self.ssd_info_filename):
                os.remove(self.ssd_info_filename)
            if os.path.exists(self.denet_info_filename):
                os.remove(self.denet_info_filename)
            if os.path.exists(self.frcnn_info_filename):
                os.remove(self.frcnn_info_filename)

            with open(ssd_imagenames_filename, 'r') as f:
                content = f.read()
            imagenames = content.split('\n')
            length = len(imagenames) - 1
            # validaion_indices = random.sample(range(length), length // 2)
            validaion_indices = []
            test_indices = [i for i in range(length) if i not in validaion_indices]
            validaion_indices = range(length)
            validation_imagenames = [imagenames[i] for i in validaion_indices]
            test_imagenames = [imagenames[i] for i in test_indices]
            with open(self.dbf_validation_imagenames_filename, 'w') as f:
                f.write('\n'.join(validation_imagenames) + '\n')
            with open(self.dbf_test_imagenames_filename, 'w') as f:
                f.write('\n'.join(test_imagenames) + '\n')

            with open(ssd_annotations_filename, 'r') as f:
                content = f.read()
            annotations = content.split('\n')
            validation_annotations = [annotations[i] for i in validaion_indices]
            test_annotations = [annotations[i] for i in test_indices]
            with open(self.dbf_validation_annotations_filename, 'w') as f:
                f.write('\n'.join(validation_annotations) + '\n')
            with open(self.dbf_test_annotations_filename, 'w') as f:
                f.write('\n'.join(test_annotations) + '\n')

            validation_recs = {}
            test_recs = {}
            with open(ssd_pickled_annotations_filename, 'rb') as f:
                content = pickle.load(f)
            for validation_imagename in validation_imagenames:
                validation_recs[validation_imagename] = content[validation_imagename]
            for test_imagename in test_imagenames:
                test_recs[test_imagename] = content[test_imagename]
            with open(self.dbf_validation_pickled_annotations_filename, 'wb') as f:
                pickle.dump(validation_recs, f)
            with open(self.dbf_test_pickled_annotations_filename, 'wb') as f:
                pickle.dump(test_recs, f)

            def split_detections_validation_and_test(detections_filename, validation_detections_filename, test_detections_filename,
                                                     full_detections_filename, validation_full_detections_filename, test_full_detections_filename):

                if os.path.exists(validation_detections_filename):
                    os.remove(validation_detections_filename)
                if os.path.exists(test_detections_filename):
                    os.remove(test_detections_filename)
                if os.path.exists(validation_full_detections_filename):
                    os.remove(validation_full_detections_filename)
                if os.path.exists(test_full_detections_filename):
                    os.remove(test_full_detections_filename)

                with open(detections_filename, 'r') as f:
                    detections = f.read().split('\n')
                validation_detections = []
                for i in range(len(detections) - 1):
                    detection = detections[i]
                    if detection.split(' ')[0] in validation_imagenames:
                        validation_detections.append(detection)
                with open(validation_detections_filename, 'w') as f:
                    f.write('\n'.join(validation_detections) + '\n')
                test_detections = []
                for i in range(len(detections)):
                    detection = detections[i]
                    if detection.split(' ')[0] in test_imagenames:
                        test_detections.append(detection)
                with open(test_detections_filename, 'w') as f:
                    f.write('\n'.join(test_detections) + '\n')

                with open(full_detections_filename, 'rb') as f:
                    full_detections = pickle.load(f)
                validation_full_detections = []
                for full_detection in full_detections:
                    if full_detection[0] in validation_imagenames:
                        validation_full_detections.append(full_detection)
                with open(validation_full_detections_filename, 'wb') as f:
                    pickle.dump(validation_full_detections, f)
                test_full_detections = []
                for full_detection in full_detections:
                    if full_detection[0] in test_imagenames:
                        test_full_detections.append(full_detection)
                with open(test_full_detections_filename, 'wb') as f:
                    pickle.dump(test_full_detections, f)

                return test_full_detections

            if 'ssd' in detectors_names_list:
                self.ssd_test_full_detections = split_detections_validation_and_test(ssd_detections_filename, self.ssd_validation_detections_filename,
                                                                                     self.ssd_test_detections_filename,
                                                                                     self.ssd_full_detections_filename,
                                                                                     self.ssd_validation_full_detections_filename,
                                                                                     self.ssd_test_full_detections_filename)

            if 'denet' in detectors_names_list:
                self.denet_test_full_detections = split_detections_validation_and_test(denet_detections_filename, self.denet_validation_detections_filename,
                                                                                      self.denet_test_detections_filename,
                                                                                      self.denet_full_detections_filename,
                                                                                      self.denet_validation_full_detections_filename,
                                                                                      self.denet_test_full_detections_filename)

            if 'frcnn' in detectors_names_list:
                self.frcnn_test_full_detections = split_detections_validation_and_test(frcnn_detections_filename, self.frcnn_validation_detections_filename,
                                                                                       self.frcnn_test_detections_filename,
                                                                                       self.frcnn_full_detections_filename,
                                                                                       self.frcnn_validation_full_detections_filename,
                                                                                       self.frcnn_test_full_detections_filename)
        else:
            if 'ssd' in detectors_names_list:
                with open(self.ssd_test_full_detections_filename, 'rb') as f:
                    self.ssd_test_full_detections = pickle.load(f)
            if 'denet' in detectors_names_list:
                with open(self.denet_test_full_detections_filename, 'rb') as f:
                    self.denet_test_full_detections = pickle.load(f)
            if 'frcnn' in detectors_names_list:
                with open(self.frcnn_test_full_detections_filename, 'rb') as f:
                    self.frcnn_test_full_detections = pickle.load(f)


    def get_scores_with_labels(self, class_scores, labels):
        detectors_scores_and_labels = {}
        for detector in detectors_names_list:
            detector_detections_all_classes_scores = class_scores[detector]
            if len(detector_detections_all_classes_scores) > 0:
                detector_labels = labels[detector]
                detector_detections_all_classes_scores = detector_detections_all_classes_scores[:, 1:]
                detector_detections_labels_scores = np.array([detector_detections_all_classes_scores[i, detector_labels[i]]
                                                           for i in range(len(detector_detections_all_classes_scores))])
                detector_detections_labels = detector_labels
                detectors_scores_and_labels[detector] = detector_detections_labels_scores, detector_detections_labels
            else:
                detectors_scores_and_labels[detector] = np.array([]), np.array([])
        return detectors_scores_and_labels


    def get_detection_vectors(self, bounding_boxes, scores_with_labels):
        keys = detectors_names_list
        detection_vectors = {}
        labels = {}
        joined_detections_indices = {}
        for detector in keys:
            detector_bounding_boxes = bounding_boxes[detector]
            detector_detection_vectors = np.ones((len(detector_bounding_boxes), len(keys), 3)) * -float('Inf')
            detector_labels = -1 * np.ones(len(detector_bounding_boxes), np.int32)
            detector_joined_detections_indices = -1 * np.ones((len(detector_bounding_boxes), len(keys)), np.int32)
            for i in range(len(detector_bounding_boxes)):
                detection_vector = np.ones((len(keys), 3)) * -float('Inf')
                joined_detection_indices = -1 * np.ones(len(keys), np.int32)
                bounding_box = np.array(detector_bounding_boxes[i])

                detector_index = keys.index(detector)
                score = scores_with_labels[detector][0][i]
                label = scores_with_labels[detector][1][i]
                detector_labels[i] = label
                detection_vector[detector_index][0], detection_vector[detector_index][1], \
                detection_vector[detector_index][2] = score, (1. - score) / 2., (1. - score) / 2.
                joined_detection_indices[detector_index] = i

                for j in range(len(keys)):
                    another_detector = keys[j]
                    if another_detector != detector:
                        if len(bounding_boxes[another_detector]) > 0:
                            another_detector_bounding_boxes = np.array(bounding_boxes[another_detector])
                            another_detector_scores = np.array(scores_with_labels[another_detector][0])
                            another_detector_labels = np.array(scores_with_labels[another_detector][1])

                            indices = np.where(another_detector_labels == detector_labels[i])[0]
                            if len(indices) == 0:
                                continue

                            another_detector_bounding_boxes = another_detector_bounding_boxes[indices]
                            another_detector_scores = another_detector_scores[indices]
                            another_detector_labels = another_detector_labels[indices]

                            ixmin = np.maximum(another_detector_bounding_boxes[:, 0], bounding_box[0])
                            iymin = np.maximum(another_detector_bounding_boxes[:, 1], bounding_box[1])
                            ixmax = np.minimum(another_detector_bounding_boxes[:, 2], bounding_box[2])
                            iymax = np.minimum(another_detector_bounding_boxes[:, 3], bounding_box[3])
                            iw = np.maximum(ixmax - ixmin + 1., 0.)
                            ih = np.maximum(iymax - iymin + 1., 0.)
                            inters = iw * ih

                            # union
                            uni = ((bounding_box[2] - bounding_box[0] + 1.) * (bounding_box[3] - bounding_box[1] + 1.) +
                                   (another_detector_bounding_boxes[:, 2] - another_detector_bounding_boxes[:, 0] + 1.) *
                                   (another_detector_bounding_boxes[:, 3] - another_detector_bounding_boxes[:, 1] + 1.) - inters)

                            overlaps = inters / np.maximum(uni, np.finfo(np.float64).eps)
                            max_overlap = np.max(overlaps)
                            if max_overlap > 0.5:
                                max_overlap_indices = np.where(overlaps == max_overlap)[0]
                                several_overlaps = another_detector_scores[max_overlap_indices]
                                several_labels = another_detector_labels[max_overlap_indices]
                                index = np.argmax(several_overlaps)
                                label_max_score = several_labels[index]
                                if detector_labels[i] == label_max_score:
                                    joined_detection_indices[j] = indices[max_overlap_indices[index]]
                                    max_score = several_overlaps[index]
                                    iou_score = max_score * max_overlap
                                    score = max_score
                                    detection_vector[j][0], detection_vector[j][1], detection_vector[j][2] = \
                                        score, (1. - score) / 2., (1. - score) / 2.
                detector_joined_detections_indices[i] = joined_detection_indices
                detector_detection_vectors[i] = detection_vector
            joined_detections_indices[detector] = detector_joined_detections_indices
            detection_vectors[detector] = detector_detection_vectors
            labels[detector] = detector_labels
        return detection_vectors, labels, joined_detections_indices


    def get_prec_rec_by_score(self, dict, score):
        prec = dict['prec']
        rec = dict['rec']
        thresholds = dict['thresholds']
        idx1 = np.where(thresholds < score)[0]
        if score == -np.inf or len(idx1) == 0:
            prec = prec[-1]
            rec = rec[-1]
        else:
            idx1 = idx1[0]
            if idx1 == 0:
                prec = prec[0]
                rec = rec[0]
            if idx1 > 0:
                idx2 = idx1 - 1
                threshold1 = thresholds[idx1]
                threshold2 = thresholds[idx2]
                a = score - threshold1
                b = threshold2 - score
                prec1 = prec[idx1]
                prec2 = prec[idx2]
                rec1 = rec[idx1]
                rec2 = rec[idx2]
                prec = b / (a + b) * prec1 + a / (a + b) * prec2
                rec = b / (a + b) * rec1 + a / (a + b) * rec2
        return prec, rec


    def rescore(self, dict, score, n):
        prec, rec = self.get_prec_rec_by_score(dict, score)
        m_T = prec
        prec_bpd = 1. - np.power(rec, n)
        m_notT = 1. - prec_bpd
        m_I = prec_bpd - prec
        return m_T, m_notT, m_I, prec


    def rescore_with_dbf(self, detection_vectors, labels, n):
        # not unique 2 detectors
        # n_values = [100., 10., 100., 90., 60., 80., 140., 70., 110., 40., 10., 190., 10., 70., 190., 20., 10., 140., 140., 50.]
        # unique 2 detectors
        # n_values = [30., 50., 90., 100., 60., 80., 140., 10., 120., 100., 10., 20., 10., 100., 230., 30., 10., 10., 100., 100.]
        # not unique 3 detectors
        # n_values = [40., 20., 80., 120., 100., 20., 60., 80., 40., 0., 20., 60., 20., 80., 200., 20., 60., 220., 240., 20.]
        rescored_detection_vectors = {}
        precisions = {}
        keys = detectors_names_list
        for detector in keys:
            detector_predictions = detection_vectors[detector]
            detector_labels = labels[detector]
            rescored_detector_predictions = detector_predictions.copy()
            detector_precisions = np.zeros(len(detector_predictions))
            for j in range(len(detector_predictions)):
                for i in range(len(keys)):
                    detector_name = keys[i]
                    dict = self.info[detector_name][dataset_classnames['PASCAL VOC'][detector_labels[j]]]
                    # n = n_values[detector_labels[j]]
                    rescored_detector_predictions[j][i][0], rescored_detector_predictions[j][i][1], \
                    rescored_detector_predictions[j][i][2], prec = \
                        self.rescore(dict, detector_predictions[j][i][0], n)
                    if detector_name == detector:
                        detector_precisions[j] = prec
            precisions[detector] = detector_precisions
            rescored_detection_vectors[detector]=rescored_detector_predictions
        return rescored_detection_vectors, precisions

    def calculate_joint_bpa(self, detector_prediction):
        epsilon = 0.000001

        def calculate_hypothesis_bpa(idx1, idx2):
            combination_len = len(detector_prediction)
            operands_count = np.power(2, combination_len)
            combinations = []
            combination_range = range(operands_count - 1)
            for i in combination_range:
                binary_representation = bin(i)[2:]
                combination = ''.join(['0'] * (combination_len - len(binary_representation))) + binary_representation
                combination = combination.replace('1', str(idx2))
                combination = combination.replace('0', str(idx1))
                combination = [int(ch) for ch in combination]
                combination = (np.array(range(combination_len)), np.array(combination))
                combinations.append(combination)
            m_f = 0.0
            for combination in combinations:
                m_f += np.prod(detector_prediction[combination])
            return m_f

        m_T_f = calculate_hypothesis_bpa(0, 2)
        m_notT_f = calculate_hypothesis_bpa(1, 2)
        N = m_T_f + m_notT_f + np.prod(detector_prediction[(np.array(range(len(detector_prediction))), np.array([2] * len(detector_prediction)))])
        m_T_f /= N + epsilon
        m_notT_f /= N + epsilon
        return m_T_f, m_notT_f


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))


    def dempster_combination_rule_result(self, bounding_boxes, rescored_detection_vectors, labels):
        keys = bounding_boxes.keys()

        detection_vectors = rescored_detection_vectors

        scores = {}
        for detector in keys:
            detector_predictions = detection_vectors[detector]
            detector_scores = np.zeros(len(detector_predictions))
            for j in range(len(detector_predictions)):
                detector_prediction = detector_predictions[j]
                m_T_f, m_notT_f = self.calculate_joint_bpa(detector_prediction)
                bel_T = m_T_f
                bel_notT = m_notT_f
                # detector_scores[j] = self.sigmoid(bel_T - bel_notT)
                detector_scores[j] = bel_T - bel_notT
            scores[detector] = detector_scores

        bounding_boxes_list = []
        labels_list = []
        scores_list = []

        for detector in keys:
            bounding_boxes_list.extend(bounding_boxes[detector])
            scores_list.extend(scores[detector])
            labels_list.extend(labels[detector])

        bounding_boxes = np.array(bounding_boxes_list)
        labels = np.array(labels_list)
        scores = np.array(scores_list)

        return bounding_boxes, labels, scores


    def compute_dbf_map(self, map_computation, select_threshold, n):

        if os.path.isfile(self.dbf_test_detections_filename):
            os.remove(self.dbf_test_detections_filename)

        time_count = 0

        a = datetime.datetime.now()

        if not os.path.exists(self.dbf_test_detections_filename):
            f = open(self.dbf_test_detections_filename, 'w')
            for j in range(len(self.ssd_test_full_detections)):
                imagename = self.ssd_test_full_detections[j][0]

                bounding_boxes = {}
                labels = {}
                class_scores = {}

                if 'ssd' in detectors_names_list:
                    if len(self.ssd_test_full_detections[j][1]) > 0:
                        bounding_boxes_ = self.ssd_test_full_detections[j][1]
                        labels_ = np.array(self.ssd_test_full_detections[j][2])
                        class_scores_ = self.ssd_test_full_detections[j][3]
                        scores = np.array([class_scores_[i, 1:][labels_[i]] for i in range(len(labels_))])
                        indices = np.where(scores > select_threshold)[0]
                        bounding_boxes['ssd'] = bounding_boxes_[indices]
                        labels['ssd'] = labels_[indices]
                        class_scores['ssd'] = class_scores_[indices]
                    else:
                        bounding_boxes['ssd'] = np.array([])
                        labels['ssd'] = np.array([])
                        class_scores['ssd'] = np.array([])
                if 'denet' in detectors_names_list:
                    if len(self.denet_test_full_detections[j][1]) > 0:
                        bounding_boxes_ = self.denet_test_full_detections[j][1]
                        labels_ = self.denet_test_full_detections[j][2]
                        class_scores_ = np.array(self.denet_test_full_detections[j][3])
                        scores = np.array([class_scores_[i, 1:][labels_[i]] for i in range(len(labels_))])
                        indices = np.where(scores > select_threshold)[0]
                        bounding_boxes['denet'] = bounding_boxes_[indices]
                        labels['denet'] = labels_[indices]
                        class_scores['denet'] = class_scores_[indices]
                    else:
                        bounding_boxes['denet'] = np.array([])
                        labels['denet'] = np.array([])
                        class_scores['denet'] = np.array([])
                if 'frcnn' in detectors_names_list:
                    if len(self.frcnn_test_full_detections[j][1]) > 0:
                        bounding_boxes_ = self.frcnn_test_full_detections[j][1]
                        labels_ = self.frcnn_test_full_detections[j][2]
                        class_scores_ = np.array(self.frcnn_test_full_detections[j][3])
                        scores = np.array([class_scores_[i, 1:][labels_[i]] for i in range(len(labels_))])
                        indices = np.where(scores > select_threshold)[0]
                        bounding_boxes['frcnn'] = bounding_boxes_[indices]
                        labels['frcnn'] = labels_[indices]
                        class_scores['frcnn'] = class_scores_[indices]
                    else:
                        bounding_boxes['frcnn'] = np.array([])
                        labels['frcnn'] = np.array([])
                        class_scores['frcnn'] = np.array([])

                scores_with_labels = self.get_scores_with_labels(class_scores, labels)
                detection_vectors, labels, joined_detections_indices = self.get_detection_vectors(bounding_boxes, scores_with_labels)
                rescored_detection_vectors, precisions = self.rescore_with_dbf(detection_vectors, labels, n)
                bounding_boxes, labels, scores = self.dempster_combination_rule_result(bounding_boxes, rescored_detection_vectors, labels)

                # img = read_one_image('/home/yulia/PycharmProjects/PASCAL VOC/VOC2007 test/VOC2007/JPEGImages/' + imagename)
                # visualization.plt_bboxes(img, labels, class_scores, bounding_boxes)

                # print(len(bounding_boxes))
                # print([len(bounding_boxes[labels == i]) for i in range(20)])

                joined_detections_indices_list = []
                precisions_list = []

                for detector in precisions.keys():
                    joined_detections_indices_list.extend(joined_detections_indices[detector])
                    precisions_list.extend(precisions[detector])

                joined_detections_indices = np.array(joined_detections_indices_list)
                precisions = np.array(precisions_list)

                # mode = 'precision'
                mode = 'dbf_score_precision'

                if mode == 'precision':
                    sort_indices = np.argsort(-precisions)
                    bounding_boxes = bounding_boxes[sort_indices]
                    labels = labels[sort_indices]
                    scores = scores[sort_indices]
                    joined_detections_indices = joined_detections_indices[sort_indices]
                    precisions = precisions[sort_indices]
                elif mode == 'dbf_score_precision':
                    sort_indices = np.argsort(-scores)
                    bounding_boxes = bounding_boxes[sort_indices]
                    labels = labels[sort_indices]
                    scores = scores[sort_indices]
                    joined_detections_indices = joined_detections_indices[sort_indices]
                    precisions = precisions[sort_indices]
                    transposition_by_prec_indices = np.zeros(len(scores), np.int32)
                    i = 0
                    while i < len(scores):
                        score_i = scores[i]
                        k = i + 1
                        while k < len(scores):
                            score_k = scores[k]
                            if score_i != score_k:
                                break
                            k += 1
                        same_indices = np.array(range(i, k))
                        prec_for_same = precisions[same_indices]
                        sorted_prec_for_same_indices = same_indices[np.argsort(-prec_for_same)]
                        transposition_by_prec_indices[i: k] = sorted_prec_for_same_indices
                        i = k
                    # if (transposition_by_prec_indices != range(len(scores))).any():
                    #     print('DIFFERS!')
                    bounding_boxes = bounding_boxes[transposition_by_prec_indices]
                    labels = labels[transposition_by_prec_indices]
                    scores = scores[transposition_by_prec_indices]
                    joined_detections_indices = joined_detections_indices[transposition_by_prec_indices]
                    precisions = precisions[transposition_by_prec_indices]

                if len(scores) > 0:
                    labels, scores, bounding_boxes, _, _ = np_methods.bboxes_nms(labels, scores, bounding_boxes, scores,
                                                                                 scores, None, nms_threshold=0.5,
                                                                                 sort_detections=False)

                # print(len(bounding_boxes))
                # print([len(bounding_boxes[labels == i]) for i in range(20)])

                # img = read_one_image(
                #     '/home/yulia/PycharmProjects/PASCAL VOC/VOC2007 test/VOC2007/JPEGImages/' + imagename)
                # visualization.plt_bboxes(img, labels, class_scores, bounding_boxes)

                time_count += 1

                if len(class_scores) > 0:
                    rscores = scores
                    for i in range(len(labels)):
                        xmin = bounding_boxes[i, 0]
                        ymin = bounding_boxes[i, 1]
                        xmax = bounding_boxes[i, 2]
                        ymax = bounding_boxes[i, 3]
                        result = '{imagename} {rclass} {rscore} {xmin} {ymin} {xmax} {ymax}\n'.format(
                            imagename=imagename, rclass=self.classnames[labels[i]],
                            rscore=rscores[i], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                        # print(str(j) + '/' + str(len(self.ssd_test_full_detections)), result)
                        f.write(result)
            f.close()
            b = datetime.datetime.now()
            total_time = (b - a).seconds
            aps, map, dict = map_computation.compute_map(self.dataset_name, self.images_dir, self.annotations_dir, self.cache_dir, self.dbf_test_imagenames_filename,
                                        self.dbf_test_annotations_filename, self.dbf_test_pickled_annotations_filename, self.dbf_test_detections_filename,
                                        self.dbf_test_full_detections_filename)
            print('mAP:', map)
            print('aps:', aps)
            print('Average DBF time: ', float(total_time) / time_count)
            return map


if __name__ == '__main__':

    print(detectors_names_list)

    dbf = DynamicBeliefFusion()

    map_computation = Computation_mAP(dbf)

    best_n = 0
    best_map = 0.0
    for n in range(0, 40, 2):
        print('n=', n)
        map = dbf.compute_dbf_map(map_computation, 0.015, float(n))
        if map > best_map:
            best_n = n
            best_map = map
    print('Best n=', best_n)