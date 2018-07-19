import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
#import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(script_dir)

from read_image import read_one_image


def parse_pascal_voc_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    imagename = tree.find('filename').text
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        try:
            obj_struct['difficult'] = int(obj.find('difficult').text)
        except:
            obj_struct['difficult'] = 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects, imagename

def parse_coco_annotations(filename):
    pass

dataset_parse_functions = {
    'PASCAL VOC': parse_pascal_voc_rec,
    'COCO': parse_coco_annotations
}

dataset_classnames = {
    'PASCAL VOC': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
}

class Computation_mAP:
    def __init__(self, detector):
        self.detector = detector

    def compute_ap(self, rec, prec, use_11_point_metric=False):
        """Compute AP given precision and recall."""
        if use_11_point_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def compute_map(self, dataset_name, images_dir, annotations_dir, cache_dir, full_imagenames_filename, imagenames_filename, annotations_filename,
                    pickled_annotations_filename, detections_filename, full_detecions_filename, ovthresh=0.5, use_11_point_metric=False):

        classnames = dataset_classnames[dataset_name]

        imagenames_file = os.path.join(cache_dir, imagenames_filename) if cache_dir not in imagenames_filename else imagenames_filename
        full_imagenames_file = os.path.join(cache_dir, full_imagenames_filename) if cache_dir not in full_imagenames_filename else full_imagenames_filename
        annotations_file = os.path.join(cache_dir, annotations_filename) if cache_dir not in annotations_filename else annotations_filename
        pickled_annotations = os.path.join(cache_dir, pickled_annotations_filename) if cache_dir not in pickled_annotations_filename else pickled_annotations_filename
        detections_file = os.path.join(cache_dir, detections_filename) if cache_dir not in detections_filename else detections_filename
        full_detections_file = os.path.join(cache_dir, full_detecions_filename) if cache_dir not in full_detecions_filename else full_detecions_filename

        dir = images_dir
        filename = imagenames_file

        if not os.path.isfile(filename):
            imagenames = []
            for (dir, subdirs, files) in os.walk(dir):
                for file in files:
                    if not file.startswith('.'):
                        imagenames.append(file)
            f = open(filename, 'w')
            imagenames = np.array(imagenames)[0:min(len(imagenames), 5000)]
            for imagename in imagenames:
                f.write(imagename + '\n')
            f.close()
        else:
            # read list of images
            with open(imagenames_file, 'r') as f:
                lines = f.readlines()
            imagenames = [x.strip() for x in lines]

        with open(full_imagenames_file, 'r') as f:
            lines = f.readlines()
        full_imagenames = [x.strip() for x in lines]

        dir = annotations_dir
        filename = annotations_file

        if not os.path.isfile(filename):
            f = open(filename, 'w')
            for (dir, subdirs, files) in os.walk(dir):
                for file in files:
                    if not file.startswith('.') and file.split('.')[0] + '.jpg' in imagenames:
                        f.write(file + '\n')
            f.close()

        annotated_existing_images = []
        if not os.path.isfile(pickled_annotations):
            with open(annotations_file, 'r') as f:
                lines = f.readlines()
            annotation_names = [x.strip() for x in lines]
            # load annots
            recs = {}
            for i, annotation_name in enumerate(annotation_names):
                rec, imagename = dataset_parse_functions[dataset_name](os.path.join(annotations_dir, annotation_name))
                if imagename in imagenames:
                    annotated_existing_images.append(imagename)
                    recs[imagename] = rec
                    if i % 100 == 0:
                        print('Reading annotation for {:d}/{:d}'.format(
                            i + 1, len(annotation_names)))
            # save
            print('Saving cached annotations to {:s}'.format(pickled_annotations))
            with open(pickled_annotations, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(pickled_annotations, 'rb') as f:
                recs = pickle.load(f)
                for imagename in recs.keys():
                    if imagename in imagenames:
                        annotated_existing_images.append(imagename)

        imagenames = annotated_existing_images

        not_matching = False
        # if len(imagenames) != len(recs):
        #     not_matching = True
        for imagename in imagenames:
            if imagename not in recs.keys():
                not_matching = True
                break
        if not_matching:
            print('Images and annotations are not matchig each other!')
            exit()

        # if os.path.isfile(detections_file):
        #     os.remove(detections_file)
        #     with open(detections_file, 'r') as f:
        #         content = f.read()
        #         lines = content.split('\n')
        #         last_line = lines[len(lines) - 2]
        #         start_imagename = last_line.split(' ')[0]
        #
        # start_index = imagenames.index(start_imagename) + 1

        # if os.path.isfile(detections_file):
        #     os.remove(detections_file)
        # if os.path.isfile(full_detections_file):
        #     os.remove(full_detections_file)

        start_index = 0

        full_detections = []
        if not os.path.isfile(detections_file):
            f = open(detections_file, 'w')
            for j in range(start_index, len(imagenames)):
                imagename = imagenames[j]
                print(imagename)
                img = read_one_image(os.path.join(images_dir, imagename))
                bounding_boxes, labels, class_scores = self.detector.predict(img)
                print(bounding_boxes.shape, labels, class_scores.shape)
                full_detections.append((imagename, bounding_boxes, labels, class_scores))
                if len(class_scores) > 0:
                    rscores = [class_scores[i, labels[i] + 1] for i in range(len(labels))]
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
                        print(str(j) + '/' + str(len(imagenames)), result)
                        f.write(result)
                else:
                    print('No detections for this image!')
            f.close()

        if not os.path.exists(full_detections_file):
            with open(full_detections_file, 'wb') as f:
                pickle.dump(full_detections, f)

        dbf_info = {}

        aps = []

        # read dets
        with open(detections_file, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]

        joined_confidences = []
        joined_tp = []
        joined_fp = []
        joined_nd = 0

        for classname in classnames:

            all_class_objects_count = 0
            for imagename in full_imagenames:
                R = [obj for obj in recs[imagename] if obj['name'] == classname]
                all_class_objects_count += len(R)

            # extract gt objects for this class
            class_recs = {}
            npos = 0
            current_test_set_class_objects_count = 0
            for imagename in imagenames:
                R = [obj for obj in recs[imagename] if obj['name'] == classname]
                current_test_set_class_objects_count += len(R)
                bbox = np.array([x['bbox'] for x in R])
                difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
                det = [False] * len(R)
                npos = npos + sum(~difficult)
                class_recs[imagename] = {'bbox': bbox,
                                         'difficult': difficult,
                                         'det': det}

            if current_test_set_class_objects_count == 0:
                aps.append(0.0)
                continue

            class_weight = float(current_test_set_class_objects_count) / float(all_class_objects_count)

            class_indices = np.array([x[1] for x in splitlines]) == classname
            image_ids = np.array([x[0] for x in splitlines])[class_indices]
            confidence = np.array(np.array([float(x[2]) for x in splitlines]))[class_indices]
            BB = [np.array([float(z) for z in x[3:]]) for x in splitlines]
            BB = np.array(BB)
            BB = BB[class_indices]

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / np.maximum(uni, np.finfo(np.float64).eps)
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.compute_ap(rec, prec, use_11_point_metric)
            aps.append(ap * class_weight)

            joined_confidences.extend(list(sorted_scores))
            joined_tp.extend(list(tp))
            joined_fp.extend(list(fp))
            joined_nd += nd

            # print(classname, ap)
            # if len(sorted_scores) > 0:
            #     print(np.min(-sorted_scores), np.max(-sorted_scores))

            #plt.plot(rec, prec)
            #plt.title('Other')
            #plt.xlabel('recall')
            #plt.ylabel('precision')
            #plt.show()

            dict = {}
            dict['prec'] = prec
            dict['rec'] = rec
            dict['thresholds'] = -sorted_scores

            dbf_info[classname] = dict

        joined_confidences = np.array(joined_confidences)
        joined_tp = np.array(joined_tp)
        joined_fp = np.array(joined_fp)

        sorted_joined_indices = np.argsort(-joined_confidences)
        sorted_joined_scores = np.sort(-joined_confidences)
        sorted_joined_tp = joined_tp[sorted_joined_indices]
        sorted_joined_fp = joined_fp[sorted_joined_indices]

        joined_prec = sorted_joined_tp / (sorted_joined_tp + sorted_joined_fp)
        joined_rec = sorted_joined_tp / joined_nd

        #dbf_info = {
        #    'prec': joined_prec,
        #    'rec': joined_rec,
        #    'thresholds': -sorted_joined_scores
        #}

        #with open('./frcnn_best_result_prec_rec.pkl', 'wb') as f:
        #    pickle.dump(dbf_info, f)

        # mAP = np.mean(aps)
        # print('mAP:', mAP)
        # print(classnames)
        # print(aps)
        return aps


if __name__ == '__main__':

    dataset_dir = '/home/yulia/PycharmProjects/PASCAL VOC/VOC2007+2012 trainval/'
    annotations_dir = os.path.join(dataset_dir, 'Annotations/')
    images_dir = os.path.join(dataset_dir, 'JPEGImages/')
    cache_dir = './'

    map_computation = Computation_mAP(None)
    map_computation.compute_map('PASCAL VOC', images_dir, annotations_dir, cache_dir, 'ssd_trainval_imagesnames.txt', 'ssd_trainval_annotations.txt',
                                'ssd_trainval_annots.pkl', 'frcnn_trainval_detections.txt', 'frcnn_trainval_full_detections.pkl')