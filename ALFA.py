# (c) Evgeny Razinkov, Kazan Federal University, 2017
import numpy as np

import bbox_clustering as bbox_clustering
from NMS import bboxes_nms


class Object:
    def __init__(self, all_detectors_names, detectors_names, bounding_boxes, class_scores, bounding_box_fusion_method,
                 class_scores_fusion_method, add_empty_detections, empty_epsilon, confidence_style):

        self.bounding_box_fusion_method = bounding_box_fusion_method
        self.class_scores_fusion_method = class_scores_fusion_method
        self.add_empty_detections = add_empty_detections
        self.empty_epsilon = empty_epsilon
        self.confidence_style = confidence_style
        self.detectors_names = list(all_detectors_names)

        self.number_of_classes = len(class_scores[0]) if len(class_scores) > 0 else 0
        self.bounding_boxes = bounding_boxes
        self.class_scores = list(class_scores)
        self.detected_by = detectors_names
        self.epsilon = 0.0
        self.finalized = False


    def get_final_bounding_box(self):
        if self.bounding_box_fusion_method == 'MAX':
            return self.max_bounding_box()
        elif self.bounding_box_fusion_method == 'MIN':
            return self.min_bounding_box()
        elif self.bounding_box_fusion_method == 'AVERAGE':
            return self.average_bounding_box()
        elif self.bounding_box_fusion_method == 'WEIGHTED AVERAGE':
            return self.weighted_average_bounding_box()
        elif self.bounding_box_fusion_method == 'WEIGHTED AVERAGE FINAL LABEL':
            return self.weighted_average_final_label_bounding_box()
        elif self.bounding_box_fusion_method == 'MOST CONFIDENT':
            return self.np_bounding_boxes[np.argmax(self.np_scores)]
        else:
            print('Unknown value for bounding_box_fusion_method ' + self.bounding_box_fusion_method + '. Using AVERAGE')
            return self.average_bounding_box()


    def average_scores(self):
        return np.average(self.np_class_scores[:self.effective_scores], axis=0)


    def multiply_scores(self):
        # print(self.np_class_scores[:self.effective_scores][:, 0::5])
        temp = np.prod(np.clip(self.np_class_scores[:self.effective_scores], a_min=self.epsilon, a_max=None), axis=0)
        return temp/np.sum(temp)


    def get_final_class_scores(self):
        if self.class_scores_fusion_method == 'AVERAGE':
            return self.average_scores()
        elif self.class_scores_fusion_method == 'MULTIPLY':
            return self.multiply_scores()
        elif self.class_scores_fusion_method == 'MOST CONFIDENT':
            return self.np_class_scores[np.argmax(self.np_scores)]
        else:
            print('Unknown value for class_scores_fusion_method ' + self.class_scores_fusion_method + '. Using AVERAGE')
            return self.average_scores()


    def finalize(self, detectors_names):
        self.detected_by_all = True
        for detector in detectors_names:
            if detector not in self.detected_by:
                self.detected_by_all = False
                self.class_scores.append(
                    [1.0 - self.empty_epsilon] + [float(self.empty_epsilon)/(self.number_of_classes - 1)] *
                    (self.number_of_classes - 1))

        self.np_bounding_boxes = np.array(self.bounding_boxes)
        self.np_bounding_boxes = np.reshape(self.np_bounding_boxes,
                                            (len(self.bounding_boxes), len(self.bounding_boxes[0])))

        self.np_class_scores = np.array(self.class_scores)
        if self.confidence_style == 'ONE MINUS NO OBJECT':
            self.np_scores = np.sum(self.np_class_scores[:, 1:], axis=1)
        elif self.confidence_style == 'LABEL':
            self.np_scores = np.amax(self.np_class_scores[:, 1:], axis=1)
        else:
            print('Unknown value for confidence_style ' + self.confidence_style + '. Using LABEL')
            self.np_scores = np.amax(self.np_class_scores[:, 1:], axis=1)

        self.finalized = True


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


    def get_object(self):
        if not self.finalized:
            self.finalize(self.detectors_names)
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


class ALFA:
    def __init__(self):
        self.bc = bbox_clustering.BoxClustering()


    def ALFA_result(self, all_detectors_names, detectors_bounding_boxes, detectors_class_scores, tau, gamma, bounding_box_fusion_method,
                        class_scores_fusion_method, add_empty_detections, empty_epsilon, same_labels_only,
                        confidence_style, use_BC, max_1_box_per_detector, single):
        """
        ALFA algorithm

        ----------
        all_detectors_names : list
            Detectors names, that sholud have taken or have taken part in fusion. For e.g. ['ssd', 'denet', 'frcnn']
            even if 'ssd' didn't detect object.

        detectors_bounding_boxes : dict
            Dictionary, where keys are detector's names and values are numpy arrays of detector's bounding boxes.

            Example: {
            'ssd': [[10, 28, 128, 250],
                    ...
                    [55, 120, 506, 709]],
            ...
            'denet': [[55, 169, 350, 790],
                      ...
                      [20, 19, 890, 620]],
            }

        detectors_class_scores : dict
            Dictionary, where keys are detector's names and values are numpy arrays of detector's class scores vectors,
            corresponding to bounding boxes.

            Example: {
            'ssd': [[0.8, 0.004, ..., 0.000009],
                    ...
                    [0.0, 0.9, ..., 0.0001]],
            ...
            'denet': [[0.4, 0.4, ..., 0.1],
                      ...
                      [0.000001, 0.0000005, ..., 0.9]],
            }

        tau : float
            Parameter tau in the paper, between [0.0, 1.0]

        gamma : float
            Parameter gamma in the paper, between [0.0, 1.0]

        bounding_box_fusion_method : str
            Bounding box fusion method ["MIN", "MAX", "MOST CONFIDENT", "AVERAGE", "WEIGHTED AVERAGE",
            "WEIGHTED AVERAGE FINAL LABEL"]

        class_scores_fusion_method : str
            Bounding box fusion method ["MOST CONFIDENT", "AVERAGE", "MULTIPLY"]

        add_empty_detections : boolean
            True - low confidence class scores tuple will be added to cluster for each detector, that missed
            False - low confidence class scores tuple won't be added to cluster for each detector, that missed

        empty_epsilon : float
            Parameter epsilon in the paper, between [0.0, 1.0]

        same_labels_only : boolean
            True - only detections with same class label will be added into same cluster
            False - detections labels won't be taken into account while clustering

        confidence_style : str
            How to compute score for object proposal ["LABEL", "ONE MINUS NO OBJECT"]
            We used "LABEL" in every experiment

        use_BC : boolean
            True - Bhattacharyya and Jaccard coefficient will be used to compute detections similarity score
            False - only Jaccard coefficient will be used to compute detections similarity score

        max_1_box_per_detector : boolean
            True - only one detection form detector could be added to cluster
            False - multiple detections from same detector could be added to cluster

        single : boolean
            True - computes ALFA prediction for mAP-s computation refered in paper
            False - computes ALFA prediction for mAP-m computation refered in paper


        Returns
        -------
        bounding_boxes : list
            Bounding boxes result of ALFA

        labels : list
            Labels result of ALFA

        class_scores : list
            Class scores result of ALFA
        """

        objects_boxes, objects_detector_names, objects_class_scores = self.bc.get_raw_candidate_objects(detectors_bounding_boxes,
                                                                                                detectors_class_scores,
                                                                                                tau, gamma,
                                                                                                same_labels_only, use_BC,
                                                                                                max_1_box_per_detector)

        objects = []
        for i in range(0, len(objects_boxes)):
            objects.append(Object(all_detectors_names, objects_detector_names[i],
                       objects_boxes[i],
                       objects_class_scores[i], bounding_box_fusion_method, class_scores_fusion_method,
                       add_empty_detections, empty_epsilon, confidence_style))

        bounding_boxes = []
        class_scores = []
        labels = []
        for detected_object in objects:
            object_bounding_box, object_class_scores, object_label = \
                detected_object.get_object()
            bounding_boxes.append(object_bounding_box)
            class_scores.append(object_class_scores)
            labels.append(object_label)
        bounding_boxes = np.array(bounding_boxes)
        class_scores = np.array(class_scores)
        labels = np.array(labels)

        if not single:
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

        labels, scores, bounding_boxes, class_scores, _ = bboxes_nms(
            labels, scores, bounding_boxes, class_scores,
            class_scores, None,
            nms_threshold=0.5)

        return bounding_boxes, labels, class_scores


