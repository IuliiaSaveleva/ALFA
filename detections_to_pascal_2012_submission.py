import pickle
import os
import shutil

from reading_methods import read_detectors_full_detections

dataset_classnames = {
    'PASCAL VOC': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
}

full_detections_filename = './ssd_denet_frcnn_0.015_2012.pkl'
submission_folder = './ssd_denet_frcnn_0.015_2012'

if os.path.exists(submission_folder):
    shutil.rmtree(submission_folder)
os.mkdir(submission_folder)
os.mkdir(submission_folder + '/results/')
submission_folder = submission_folder + '/results/'
os.mkdir(submission_folder + '/VOC2012/')
submission_folder = submission_folder + '/VOC2012/'
os.mkdir(submission_folder + '/Main/')
submission_folder = submission_folder + '/Main/'

classes_imagenames_scores_bboxes = {}

full_detections = list(read_detectors_full_detections([full_detections_filename]).values())[0]

for j in range(len(full_detections)):
    imagename = full_detections[j][0].split('.')[0]
    bounding_boxes, labels, class_scores = full_detections[j][1], full_detections[j][2], full_detections[j][3]
    if len(class_scores) > 0:
        rscores = [class_scores[i, labels[i] + 1] for i in range(len(labels))]
        for i in range(len(labels)):
            label_name = dataset_classnames['PASCAL VOC'][labels[i]]
            score = rscores[i]
            bounding_box = bounding_boxes[i]
            if label_name not in classes_imagenames_scores_bboxes:
                classes_imagenames_scores_bboxes[label_name] = ([imagename], [score], [bounding_box])
            else:
                imagenames, scores, bounding_boxes_ = classes_imagenames_scores_bboxes[label_name]
                imagenames.append(imagename)
                scores.append(score)
                bounding_boxes_.append(bounding_box)
                classes_imagenames_scores_bboxes[label_name] = (imagenames, scores, bounding_boxes_)

for class_name in classes_imagenames_scores_bboxes.keys():
    result_file = submission_folder + 'comp4_det_test_' + class_name + '.txt'
    with open(result_file, 'w') as f:
        imagenames, class_scores, bounding_boxes = classes_imagenames_scores_bboxes[class_name]
        if len(imagenames) > 0:
            for i in range(0, len(imagenames)):
                bounding_box = bounding_boxes[i]
                xmin = bounding_box[0]
                ymin = bounding_box[1]
                xmax = bounding_box[2]
                ymax = bounding_box[3]
                bbox_result = "%.6f" % xmin + ' ' + "%.6f" % ymin + ' ' + "%.6f" % xmax + ' ' + "%.6f" % ymax
                result = imagenames[i] + ' ' + "%.6f" % class_scores[i] + ' ' + bbox_result + '\n'
                f.write(result)