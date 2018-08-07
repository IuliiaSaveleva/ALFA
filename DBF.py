import numpy as np

from map_computation import Computation_mAP
from reading_methods import dataset_classnames
from NMS import bboxes_nms

class DBF:
    def __init__(self, all_detectors_names, validation_dataset_name, validation_dataset_dir, validation_imagenames,
                 validation_annotations, validation_detectors_detections, map_iou_threshold):
        self.all_detectors_names = list(all_detectors_names)

        self.epsilon = 1e-4

        self.map_computation = Computation_mAP(None)

        self.detectors_pr_curves = {}

        print('Computing PR-curves...')
        for key in validation_detectors_detections.keys():
            _, _, pr_curves = self.map_computation.compute_map(validation_dataset_name, validation_dataset_dir,
                                                   validation_imagenames, validation_annotations,
                                                   validation_detectors_detections[key],
                                                   map_iou_threshold)
            self.detectors_pr_curves[key] = pr_curves


    def get_scores_with_labels(self, class_scores, labels):
        detectors_scores_and_labels = {}
        keys = self.all_detectors_names
        for detector in keys:
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
        detection_vectors = {}
        labels = {}
        joined_detections_indices = {}
        keys = self.all_detectors_names
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
        rescored_detection_vectors = {}
        precisions = {}
        keys = self.all_detectors_names
        for detector in keys:
            detector_predictions = detection_vectors[detector]
            detector_labels = labels[detector]
            rescored_detector_predictions = detector_predictions.copy()
            detector_precisions = np.zeros(len(detector_predictions))
            for j in range(len(detector_predictions)):
                for i in range(len(keys)):
                    detector_name = keys[i]
                    dict = self.detectors_pr_curves[detector_name][dataset_classnames['PASCAL VOC'][detector_labels[j]]]
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


    def DBF_result(self, detectors_bounding_boxes, detectors_class_scores, detectors_labels, n):

        for key in self.all_detectors_names:
            if key not in detectors_bounding_boxes:
                detectors_bounding_boxes[key] = np.array([])
                detectors_class_scores[key] = np.array([])
                detectors_labels[key] = np.array([])

        scores_with_labels = self.get_scores_with_labels(detectors_class_scores, detectors_labels)
        detection_vectors, labels, joined_detections_indices = self.get_detection_vectors(detectors_bounding_boxes,
                                                                                          scores_with_labels)
        rescored_detection_vectors, precisions = self.rescore_with_dbf(detection_vectors, labels, n)
        bounding_boxes, labels, scores = self.dempster_combination_rule_result(detectors_bounding_boxes,
                                                                               rescored_detection_vectors, labels)

        precisions_list = []

        for detector in precisions.keys():
            precisions_list.extend(precisions[detector])

        precisions = np.array(precisions_list)

        sort_indices = np.argsort(-scores)
        bounding_boxes = bounding_boxes[sort_indices]
        labels = labels[sort_indices]
        scores = scores[sort_indices]
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
        bounding_boxes = bounding_boxes[transposition_by_prec_indices]
        labels = labels[transposition_by_prec_indices]
        scores = scores[transposition_by_prec_indices]

        if len(scores) > 0:
            labels, scores, bounding_boxes, _, _ = bboxes_nms(labels, scores, bounding_boxes, scores,
                                                                         scores, None, nms_threshold=0.5,
                                                                         sort_detections=False)

            keys = list(detectors_class_scores.keys())
            if len(keys) > 0 and len(detectors_class_scores[keys[0]]) > 0:
                class_scores = np.zeros((len(scores), len(detectors_class_scores[keys[0]][0])))
                for i in range(len(labels)):
                    class_scores[i][labels[i] + 1] = scores[i]
            else:
                labels = np.array([])
                class_scores = np.array([])
                bounding_boxes = np.array([])

        else:
            labels = np.array([])
            class_scores = np.array([])
            bounding_boxes = np.array([])

        return bounding_boxes, labels, class_scores