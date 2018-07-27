# (c) Evgeny Razinkov, Kazan Federal University, 2017

import numpy as np


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


def cut_no_object_score(class_scores):
    without_no_object = class_scores[:, 1:]
    class_scores_sum = np.sum(without_no_object, axis = 1)
    class_scores_sum = class_scores_sum + np.equal(class_scores_sum, 0.0).astype(float)
    new_class_scores = (without_no_object.T / class_scores_sum).T
    return new_class_scores


def fastIoU(boxes):
    number_of_boxes = len(boxes)
    m0 = np.zeros((number_of_boxes, number_of_boxes, 4))
    m0[:] = boxes
    m0T = np.transpose(m0, axes = (1, 0, 2))
    m_up_left = np.maximum(m0[:, :, :2], m0T[:, :, :2])
    m_down_right = np.minimum(m0[:, :, 2:], m0T[:, :, 2:])
    m_diff = np.clip(m_down_right - m_up_left, a_min = 0.0, a_max=None)
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
    return s_diff


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
    b0mul = b0 * np.transpose(b0, axes=(1, 0, 2))
    bc = np.sum(np.sqrt(b0mul), axis=2)
    return bc


class BoxClustering:
    def __init__(self):
        pass

    def prepare_matrix(self):
        self.sim_matrix = fastIoU(self.boxes)

        if self.max_1_box_per_detector:
            s_diff = fastSame(self.names)
            self.sim_matrix = self.sim_matrix * (s_diff + np.eye(self.n_boxes))
        if self.same_labels_only:
            l_same = fastLabels(self.only_object_scores)
            self.sim_matrix *= l_same
        if self.use_BC:
            bc = fastBC(self.only_object_scores)
            self.sim_matrix = np.power(self.sim_matrix, self.power_iou) * np.power(bc, 1.0 - self.power_iou)
        
        self.path_matrix = np.greater_equal(self.sim_matrix, self.hard_threshold).astype(int)


    def get_paths(self):
        self.whole_path_matrix = self.path_matrix
        new_paths_might_exist = True
        while new_paths_might_exist:
            self.new_whole_path_matrix = np.matmul(self.whole_path_matrix, np.transpose(self.whole_path_matrix))
            self.new_whole_path_matrix = np.greater_equal(self.new_whole_path_matrix, 0.5).astype(int)
            if np.array_equal(self.whole_path_matrix, self.new_whole_path_matrix):
                new_paths_might_exist = False
            self.whole_path_matrix = self.new_whole_path_matrix


    def cluster_indices(self, indices):
        clusters = [[a] for a in indices]
        lc = len(indices)
        if lc == 1:
            return clusters
        elif lc == 2:
            return [clusters[0] + clusters[1]]
        else:
            ci1, ci2, sim = self.find_clusters_to_merge(clusters)
            while sim > self.hard_threshold:
                clusters = self.merge_clusters(ci1, ci2, clusters)
                if len(clusters) == 1:
                    return clusters
                else:
                    ci1, ci2, sim = self.find_clusters_to_merge(clusters)
            return clusters

        
    def find_clusters_to_merge(self, clusters):
        max_cluster_sim = 0.0
        ci1 = None
        ci2 = None
        if len(clusters) < 2:
            return 0, 0, 1.0
        for i in range(1, len(clusters)):
            for j in range(0, i):
                cl_sim = self.cluster_distance(clusters[i], clusters[j])
                if max_cluster_sim < cl_sim:
                    ci1 = i
                    ci2 = j
                    max_cluster_sim = cl_sim
        return ci1, ci2, max_cluster_sim


    def merge_clusters(self, ci1, ci2, clusters):
        if ci1 == ci2:
            return clusters
        clusters[ci1] = clusters[ci1] + clusters[ci2]
        del clusters[ci2]
        return clusters


    def cluster_distance(self, c1, c2):
        min_sim_init = False
        min_sim = 0.0
        for x1 in c1:
            for x2 in c2:
                if not min_sim_init:
                    min_sim = self.sim_matrix[x1, x2]
                    min_sim_init = True
                else:
                    if min_sim > self.sim_matrix[x1, x2]:
                        min_sim = self.sim_matrix[x1, x2]
        return min_sim


    def get_raw_candidate_objects(self, bounding_boxes, class_scores, tau, gamma, same_labels_only, use_BC,
                 max_1_box_per_detector):
        """
        Clusters detections from different detectors.

        Parameters
        ----------
        bounding_boxes : dict
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

        class_scores : dict
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

        same_labels_only : boolean
            True - only detections with same class label will be added into same cluster
            False - detections labels won't be taken into account while clustering

        use_BC : boolean
            True - Bhattacharyya and Jaccard coefficient will be used to compute detections similarity score
            False - only Jaccard coefficient will be used to compute detections similarity score

        max_1_box_per_detector : boolean
            True - only one detection form detector could be added to cluster
            False - multiple detections from same detector could be added to cluster

        Returns
        -------
        objects_boxes : list
            List containing [clusters, cluster bounding boxes, box coordinates]
            cluster bounding boxes - bounding boxes added to cluster

        objects_detectors_names : list
            List containing [clusters, cluster detectors names]
            cluster detectors names - names of detectors, corresponding to bounding boxes added to cluster

        objects_class_scores : list
            List containing [clusters, cluster class scores]
            cluster class scores - class scores, corresponding to bounding boxes added to cluster

        """

        self.hard_threshold = tau
        self.power_iou = gamma
        self.same_labels_only = same_labels_only
        self.use_BC = use_BC
        self.max_1_box_per_detector = max_1_box_per_detector

        self.detector_names = list(bounding_boxes.keys())
        number_of_classes = 1
        if len(self.detector_names) > 0:
            number_of_classes = len(class_scores[self.detector_names[0]][0])

        self.n_boxes = 0
        self.bounding_boxes = bounding_boxes
        self.boxes = np.zeros((0, 4))
        self.names = np.zeros(0)
        self.class_scores = np.zeros((0, number_of_classes))
        self.only_object_scores = np.zeros((0, number_of_classes - 1))

        detector_index = 0
        self.actual_names = []
        for detector_name in self.detector_names:
            detector_boxes = len(bounding_boxes[detector_name])
            if detector_boxes > 0:
                self.n_boxes += detector_boxes
                self.boxes = np.vstack((self.boxes, bounding_boxes[detector_name]))
                self.actual_names += [detector_name] * detector_boxes
                self.names = np.hstack((self.names, np.ones(detector_boxes) * detector_index))
                self.class_scores = np.vstack((self.class_scores, class_scores[detector_name]))
            detector_index += 1
        self.only_object_scores = cut_no_object_score(self.class_scores)

        self.prepare_matrix()

        self.get_paths()

        objects_boxes = []
        objects_detectors_names = []
        objects_class_scores = []
        np_boxes = np.array(self.boxes)
        np_detectors = np.array(self.actual_names)
        np_class_scores = np.array(self.class_scores)
        # max_objects = len(self.boxes)
        # num_objects = 0
        # main_result_boxes = np.zeros((max_objects, len(self.detector_names), 4))
        # main_result_names = np.zeros((max_objects, len(self.detector_names)))
        # main_result_class_scores = np.zeros((max_objects, len(self.detector_names), number_of_classes))
        if len(self.whole_path_matrix):
            unique_whole_path_matrix = np.vstack({tuple(row) for row in self.whole_path_matrix})
            for i in range(0, len(unique_whole_path_matrix)):
                indices, = np.where(unique_whole_path_matrix[i] > 0)
                if len(indices) > 0:
                    clusters = self.cluster_indices(list(indices))
                    for c in clusters:
                        # nn = len(c)
                        # main_result_boxes[num_objects][:nn] = np_boxes[c]
                        # main_result_names[num_objects][:nn] = np_detectors[c]
                        # main_result_class_scores[num_objects][:nn] = np_class_scores[c]
                        # num_objects += 1

                        boxes_0 = np_boxes[c]
                        objects_boxes.append(boxes_0)
                        detectors_0 = list(np_detectors[c])
                        class_scores_0 = list(np_class_scores[c])
                        objects_class_scores.append(class_scores_0)
                        objects_detectors_names.append(detectors_0)

        return objects_boxes, objects_detectors_names, objects_class_scores
        # return main_result_boxes[:num_objects], main_result_names[:num_objects], main_result_class_scores[:num_objects]


