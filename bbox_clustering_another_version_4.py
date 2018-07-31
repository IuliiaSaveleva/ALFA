# (c) Evgeny Razinkov, Kazan Federal University, 2017
import os
import pickle
import datetime

import numpy as np
# import tensorflow as tf
#import matplotlib.pyplot as plt
#import objekt


ADD_FAKE_LOGITS = False

# ENSEMBLE_BOUNDING_BOX = 'MAX'
# ENSEMBLE_BOUNDING_BOX = 'MIN'
# ENSEMBLE_BOUNDING_BOX = 'AVERAGE'
# ENSEMBLE_BOUNDING_BOX = 'WEIGHTED_AVERAGE'
# ENSEMBLE_SCORES = 'MULTIPLY'
# ENSEMBLE_SCORES = 'AVERAGE'
# ADD_EMPTY_DETECTIONS = False
# ADD_EMPTY_DETECTIONS = True

MAX_1_BOX_PER_DETECTOR = True
USE_BC = True

def iou_box_area(box):
    return (box[2] - box[0])*(box[3] - box[1])


def get_iou(box1, box2):
    l1x, l1y, r1x, r1y = box1
    l2x, l2y, r2x, r2y = box2
    if l2x >= r1x: 
        return 0
    if l1x >= r2x: 
        return 0
    if l2y >= r1y: 
        return 0
    if l1y >= r2y: 
        return 0
    xs = np.sort(np.array([l1x, r1x, l2x, r2x])).astype(float)
    ys = np.sort(np.array([l1y, r1y, l2y, r2y])).astype(float)
    intersection = (xs[2] - xs[1])*(ys[2] - ys[1])
    union = iou_box_area(box1) + iou_box_area(box2) - intersection
    return intersection/union

def BC(distribution1, distribution2):
    bc = 0.0
    for i in range(0, len(distribution1)):
        bc += np.sqrt(distribution1[i] * distribution2[i])
    return bc 

def cut_no_object_array(class_scores):
    without_no_object = class_scores[:, 1:]
    class_scores_sum = np.sum(without_no_object, axis = 1)
    class_scores_sum = class_scores_sum + np.equal(class_scores_sum, 0.0).astype(float)
    new_class_scores = (without_no_object.T / class_scores_sum).T
    return new_class_scores

def cut_no_object(class_scores_item):
    without_no_object = class_scores[1:]
    class_scores_sum = np.sum(without_no_object)
    if class_scores_sum == 0:
        class_scores_sum = 1
    new_class_scores = without_no_object/class_scores_sum
    return new_class_scores

def fastIoU(boxes):
    number_of_boxes = len(boxes)
    m0 = np.zeros((number_of_boxes, number_of_boxes, 4))
    m0[:] = boxes
    m0T = np.transpose(m0, axes = (1, 0, 2))
    m_up_left = np.maximum(m0[:, :, :2], m0T[:, :, :2])
    m_down_right = np.minimum(m0[:, :, 2:], m0T[:, :, 2:])
    m_diff = np.clip(m_down_right - m_up_left, a_min = 0.0, a_max = None)
    m_intersection = m_diff[:, :, 0] * m_diff[:, :, 1]
    m_area = (m0[:, :, 2] - m0[:, :, 0]) * (m0[:, :, 3] - m0[:, :, 1])
    m_T_area = (m0T[:, :, 2] - m0T[:, :, 0]) * (m0T[:, :, 3] - m0T[:, :, 1])
    iou = m_intersection / (m_area + m_T_area - m_intersection)
    return iou

def fastSame(detector_indices):
    number_of_boxes = len(detector_indices)
    s0 = np.zeros((number_of_boxes, number_of_boxes))
    s0[:] = detector_indices
    s_diff = np.not_equal(s0 - s0.T, 0.0).astype(float)
    return s_diff #+ np.eye(number_of_boxes)

def fastLabels(only_object_scores):
    number_of_boxes = len(only_object_scores)
    labels = np.argmax(only_object_scores, axis = 1)
    L0 = np.zeros((number_of_boxes, number_of_boxes))
    L0[:] = labels
    l_same = np.equal(L0 - L0.T, 0.0).astype(float)
    return l_same

def fastBC(only_object_scores):
    number_of_boxes = len(only_object_scores)
    number_of_actual_classes = 0
    if len(only_object_scores) > 0:
        number_of_actual_classes = len(only_object_scores[0])
    b0 = np.zeros((number_of_boxes, number_of_boxes, number_of_actual_classes))
    b0[:] = only_object_scores
    b0mul = b0 * np.transpose(b0, axes = (1, 0, 2))
    bc = np.sum(np.sqrt(b0mul), axis = 2)
    return bc


class BoxClustering:
    def __init__(self, bounding_boxes, class_scores, hard_threshold=0.5, power_iou=0.5, same_labels_only=True, silent=True):
        self.same_labels_only = same_labels_only
        self.power_iou = power_iou
        self.detector_names = bounding_boxes.keys()
        number_of_classes = 1
        if len(self.detector_names) > 0:
            number_of_classes = len(class_scores[self.detector_names[0]][0])
        #number_of_boxes = 0
        #for boxes in bounding_boxes.values():
        #    number_of_boxes += len(boxes)

        self.n_boxes = 0
        self.bounding_boxes = bounding_boxes
        #self.boxes = []
        self.boxes = np.zeros((0, 4))
        #self.names = []
        self.names = np.zeros(0)
        #self.class_scores = []
        self.class_scores = np.zeros((0, number_of_classes))
        #self.only_object_scores = []
        self.only_object_scores = np.zeros((0, number_of_classes - 1))
        
        self.hard_threshold = hard_threshold
        detector_index = 0
        self.actual_names = []
        for detector_name in self.detector_names:

            detector_boxes = len(bounding_boxes[detector_name])
            if detector_boxes > 0:
                self.n_boxes += detector_boxes
                #self.boxes += bounding_boxes[detector_name]
                #print(self.boxes.shape, bounding_boxes[detector_name].shape)
                self.boxes = np.vstack((self.boxes, bounding_boxes[detector_name]))
                #self.names +=  [detector_index] * len(bounding_boxes[detector_name])
                self.actual_names += [detector_name] * detector_boxes
                self.names = np.hstack((self.names, np.ones(detector_boxes) * detector_index ))
                #self.class_scores += class_scores[detector_name]
                self.class_scores = np.vstack((self.class_scores, class_scores[detector_name]))

                #self.only_object_scores += cut_no_object_array(class_scores[detector_name])
                #self.only_object_scores = np.vstack((self.only_object_scores, cut_no_object_array(class_scores[detector_name])))
            detector_index += 1
        self.only_object_scores = cut_no_object_array(self.class_scores)
        self.silent = silent
        self.status(self.boxes)
        self.status(self.class_scores)
        self.status(self.names)
        



    def status(self, message):
        if not self.silent:
            print(message)

    def prepare_matrix_(self):
        self.status('Preparing matrices...')
        self.iou_matrix = np.zeros((self.n_boxes, self.n_boxes))

        self.path_matrix = np.zeros((self.n_boxes, self.n_boxes)).astype(int)
        self.bc_matrix = np.zeros((self.n_boxes, self.n_boxes))

        for b1 in range(0, self.n_boxes):
            self.status(self.boxes[b1])
            self.path_matrix[b1, b1] = 1
            self.iou_matrix[b1, b1] = 1.0
            for b2 in range(0, b1):
                self.status(self.boxes[b2])
                if MAX_1_BOX_PER_DETECTOR and (self.names[b1] == self.names[b2]):
                    iou = 0.0
                elif self.same_labels_only and (np.argmax(self.class_scores[b1][1:])!=np.argmax(self.class_scores[b2][1:])):
                    iou = 0.0
                else:
                    iou = get_iou(self.boxes[b1], self.boxes[b2])
                self.status('IoU: ' + str(iou))
                self.iou_matrix[b1, b2] = iou
                self.iou_matrix[b2, b1] = iou
                #bc = BC()

                #self.bc_matrix[b1, b2] = 
                if iou > self.hard_threshold:
                    self.path_matrix[b1, b2] = 1
                    self.path_matrix[b2, b1] = 1
        self.status('Matrix:')
        self.status(self.path_matrix)
        self.status(self.iou_matrix)
        self.status('Matrices ready')

    def prepare_matrix(self):
        self.iou_matrix = fastIoU(self.boxes)
        # for i in range(len(self.iou_matrix)):
            # if self.iou_matrix[i, i] < 1:
            #     print(self.iou_matrix[i, i])
            #     print(self.boxes[i])
            #     print(self.actual_names[i])
        if MAX_1_BOX_PER_DETECTOR:
            s_diff = fastSame(self.names)
            self.iou_matrix = self.iou_matrix * (s_diff + np.eye(self.n_boxes))
        if self.same_labels_only:
            l_same = fastLabels(self.only_object_scores)
            self.iou_matrix *= l_same
        if USE_BC:
            bc = fastBC(self.only_object_scores)
            self.iou_matrix = np.power(self.iou_matrix, self.power_iou)* np.power(bc, 1.0 - self.power_iou)
        
        self.path_matrix = np.greater_equal(self.iou_matrix, self.hard_threshold).astype(int)
        self.status(self.iou_matrix)
        self.status(self.path_matrix)

    def get_paths(self):
        self.status('Getting paths...')
        self.whole_path_matrix = self.path_matrix
        new_paths_might_exist = True
        while new_paths_might_exist:
            self.new_whole_path_matrix = np.matmul(self.whole_path_matrix, np.transpose(self.whole_path_matrix))
            self.new_whole_path_matrix = np.greater_equal(self.new_whole_path_matrix, 0.5).astype(int)
            if np.array_equal(self.whole_path_matrix, self.new_whole_path_matrix):
                new_paths_might_exist = False
            self.whole_path_matrix = self.new_whole_path_matrix
        self.status('All paths:')
        self.status(self.whole_path_matrix)
        self.status('All paths are ready')

    def cluster_indices(self, indices):
        #clusters = [np.array([a]) for a in indices]
        clusters = [[a] for a in indices]
        if len(clusters) == 1:
            return clusters
        ci1, ci2, iou = self.find_clusters_to_merge(clusters)
        while iou > self.hard_threshold:
            self.status('IoU of merging clusters: ' + str(iou))
            clusters = self.merge_clusters(ci1, ci2, clusters)
            if len(clusters) == 1:
                return clusters
            else:
                ci1, ci2, iou = self.find_clusters_to_merge(clusters)
        return clusters

        
    def find_clusters_to_merge(self, clusters):
        max_cluster_iou = 0.0
        ci1 = None
        ci2 = None
        if len(clusters) < 2:
            return 0, 0, 1.0
        for i in range(1, len(clusters)):
            for j in range(0, i):
                cl_iou = self.cluster_distance(clusters[i], clusters[j])
                if max_cluster_iou < cl_iou:
                    ci1 = i
                    ci2 = j
                    max_cluster_iou = cl_iou
        return ci1, ci2, max_cluster_iou

    def merge_clusters(self, ci1, ci2, clusters):
        if ci1 == ci2:
            return clusters
        #clusters[ci1] = np.hstack((clusters[ci1], clusters[ci2]))
        clusters[ci1] = clusters[ci1] + clusters[ci2]
        del clusters[ci2]
        return clusters
    
    def cluster_distance(self, c1, c2):
        #return np.min(self.iou_matrix[c1, c2])
        min_iou_init = False
        min_iou = 0.0


        for x1 in c1:
            for x2 in c2:
                if not min_iou_init:
                    min_iou = self.iou_matrix[x1, x2]
                    min_iou_init = True
                else:
                    if min_iou > self.iou_matrix[x1, x2]:
                        min_iou = self.iou_matrix[x1, x2]
        return min_iou


    def get_raw_candidate_objects(self):

        self.prepare_matrix()
        # for i in range(len(self.path_matrix)):
        #     if np.sum(self.path_matrix[i]) == 0:
        #         print('ppc0')

        self.get_paths()
        # for i in range(len(self.whole_path_matrix)):
        #     if np.sum(self.whole_path_matrix[i]) == 0:
        #         print('ppc')
        objects_boxes = []
        objects_detectors = []
        objects_logits = []
        objects_I = []
        result_boxes = []
        np_boxes = np.array(self.boxes)
        np_detectors = np.array(self.actual_names)
        np_logits = np.array(self.class_scores)
        if len(self.whole_path_matrix):
            unique_whole_path_matrix = np.vstack({tuple(row) for row in self.whole_path_matrix})
            for i in range(0, len(unique_whole_path_matrix)):
                indices, = np.where(unique_whole_path_matrix[i] > 0)
                self.status('Number of bounding boxes to cluster: ' + str(len(indices)))
                if len(indices) > 0:
                    clusters = self.cluster_indices(list(indices))
                    self.status('Clusters created: ' + str(len(clusters)))
                    for c in clusters:
                        boxes_0 = np_boxes[c]
                        objects_boxes.append(boxes_0)
                        result_boxes.append(np.average(boxes_0, axis = 0))
                        detectors_0 = list(np_detectors[c])


                        logits_0 = list(np_logits[c])


                        objects_logits.append(logits_0)
                        objects_detectors.append(detectors_0)

        return objects_boxes, objects_detectors, objects_logits
        
    def process_candidate_objects(self, objects_boxes, objects_detectors, objects_logits):
        result_boxes = []
        objects_I = []
        for i in range(0, len(objects_boxes)):
            boxes_0 = objects_boxes[i]
            logits_0 = objects_logits[i]
            detectors_0 = objects_detectors[i]
            result_boxes.append(np.average(boxes_0, axis = 0))
            I_0 = [1] * len(boxes_0)
            if ADD_FAKE_LOGITS:
                for detector_name in self.detector_names:
                    if detector_name not in detectors_0:
                        logits_0.append(np.zeros_like(self.class_scores[0]))
                        detectors_0.append(detector_name)
                        I_0.append(0)
            objects_I.append(I_0)    
        return objects_boxes, objects_detectors, objects_logits, objects_I, result_boxes

    def get_candidate_objects(self):
        objects_boxes, objects_detectors, objects_logits = self.get_raw_candidate_objects()
        return self.process_candidate_objects(objects_boxes, objects_detectors, objects_logits)






if __name__ == '__main__':


    bboxes_ssd = np.array([[1., 1., 11., 11.],
                       [1.1, 1.1, 10, 10],
                       [2., 2., 9., 9.],
                       [5., 11., 30., 30.]])
    bboxes_frcnn = np.array([[1.5, 0.7, 10.1, 9.],
                       [2.1, 3.1, 15, 11],
                       [1., 2.2, 9.1, 9.9],
                       [5.7, 11.8, 28., 31.],
                       [3.7, 10.8, 27., 28.]])
    logits_ssd = np.array([[0.1, 0.1, 0.8], 
                               [0.2, 0.1, 0.7],
                               [0.3, 0.1, 0.6],
                               [0.0, 0.7, 0.3]])
    logits_frcnn = np.array([[0.1, 0.1, 0.8], 
                                 [0.2, 0.1, 0.7],
                                 [0.3, 0.1, 0.6],
                                 [0.5, 0.2, 0.1],
                                 [0.0, 0.7, 0.3]])


    detectors = ['ssd', 'frcnn']
    bounding_boxes = {}
    bounding_boxes['ssd'] = bboxes_ssd #np.vstack((bboxes_ssd, bboxes_ssd, bboxes_ssd, bboxes_ssd))
    bounding_boxes['frcnn'] = bboxes_frcnn #np.vstack((bboxes_frcnn, bboxes_frcnn, bboxes_frcnn, bboxes_frcnn))
    class_scores = {}
    class_scores['ssd'] = logits_ssd #np.vstack((logits_ssd, logits_ssd, logits_ssd, logits_ssd))
    class_scores['frcnn'] = logits_frcnn #np.vstack((logits_frcnn, logits_frcnn, logits_frcnn, logits_frcnn))

    for i in range(1):
        box_cl = BoxClustering(bounding_boxes = bounding_boxes, class_scores = class_scores, hard_threshold = 0.3, power_iou= 0.5, silent = False)
        object_boxes, object_detector_names, object_logits, object_I, result_boxes = box_cl.get_candidate_objects()

    number_of_objects = len(object_boxes)

    print(str(number_of_objects) + ' objects detected')
    for i in range(0, number_of_objects):
        print('Object ' + str(i) +'. Resulting bounding box: ' + str(result_boxes[i]))
        number_of_boxes = len(object_boxes[i])
        number_of_logits = len(object_logits[i])
        for j in range(0, number_of_logits):
            if j < number_of_boxes:
                print('Bounding box: ' + str(object_boxes[i][j]) + '. class_scores: ' + str(object_logits[i][j]) + ' (detected by ' + str(object_detector_names[i][j]) + ')')
            else:
                print('No bounding box. Fake class_scores: ' + str(object_logits[i][j]) + ' (NOT detected by ' + str(object_detector_names[i][j]) + ')')




