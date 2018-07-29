import os
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import argparse

script_dir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(script_dir)



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


dataset_parse_functions = {
    'PASCAL VOC': parse_pascal_voc_rec,
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


    def compute_map(self, dataset_name, dataset_dir, imagenames, annotations,
                    images_detections, map_iou_threshold, weighted_map=False,
                    full_imagenames=None):
        """
        Computes mean Average Precision.

        ----------
        dataset_name : string
            Dataset name, e.g. 'PASCAL VOC'

        dataset_dir : string
            Path to images dir, e.g.'.../PASCAL VOC/VOC2007 test/VOC2007/'

        imagenames : list
            List contains all or a part of dataset filenames

        annotations : dict
            Dict of annotations

        detections_filename : string
            Pickle to store detections for mAP computation
            File contains a list of detections for one detector or fusion result, kept in format:

            (image filename: '00000.png', bounding boxes: [[23, 45, 180, 790], ..., [100, 39, 705, 98]],
                labels: [0, ..., 19], class_scores: [[0.0, 0.01, ...., 0.98], ..., [0.9, 0.0, ..., 0.001]]),

            Means all detections found on image "image_filename", where bouning box, label, class score
                correspond elementwise

        map_iou_threshold : float
            Jaccard coefficient value to compute mAP, between [0, 1]

        weighted_map : boolean
            True - compute weighted mAP by part class samples count to all class samples count in dataset
            False - compute ordinary mAP

        full_imagenames: list
            List contains all of dataset filenames, if compute
            weighted map on a part of dataset


        Returns
        -------
        aps : list
            Average Precision value for each class

        mAP : float
            mean Average Precision value

        pr_curves : dict
            Precision-Recall curves for each class stored in format:
                pr_curves[
                    'aeroplane': {
                        'prec'
                        'rec'
                        'thresholds'
                    },
                    ...
                    'train': {
                        'prec'
                        'rec'
                        'thresholds'
                    }
        """

        if not os.path.exists(dataset_dir):
            print('Dataset dir not found!')
            exit(1)

        if dataset_name not in dataset_classnames:
            print('Invalid dataset name!')
            exit(1)

        classnames = dataset_classnames[dataset_name]

        detections = []
        for image_detections in images_detections:
            image_detections_count = len(image_detections[1])
            for i in range(image_detections_count):
                detections.append((image_detections[0], image_detections[1][i], image_detections[2][i],
                                   image_detections[3][i]))

        recs = annotations

        pr_curves = {}

        aps = []

        for i in range(len(classnames)):

            classname = classnames[i]

            if weighted_map:
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

            class_weight = 1.0

            if weighted_map:
                if current_test_set_class_objects_count == 0:
                    aps.append(0.0)
                    continue

                class_weight = float(current_test_set_class_objects_count) / float(all_class_objects_count)

            class_indices = [j for j in range(len(detections)) if detections[j][2] == i]
            image_ids = np.array([detections[j][0] for j in range(len(detections))])[class_indices]
            confidence = np.array([detections[j][3][detections[j][2] + 1] for j in range(len(detections))])[class_indices]
            BB = np.array([detections[j][1] for j in range(len(detections))])[class_indices]

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

                if ovmax > map_iou_threshold:
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
            ap = self.compute_ap(rec, prec, False)
            aps.append(ap * class_weight)
            print(classname + ' AP:', ap)

            # plt.plot(rec, prec)
            # plt.title(classname)
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.show()

            dict = {}
            dict['prec'] = prec
            dict['rec'] = rec
            dict['thresholds'] = -sorted_scores

            pr_curves[classname] = dict

        mAP = np.mean(aps)
        print('mAP:', mAP)
        return aps, mAP, pr_curves


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Only \"PASCAL VOC\" is supported, default=\"PASCAL VOC\"',
                        default='PASCAL VOC')
    parser.add_argument('--dataset_dir', required=True, type=str,
        help='\"(Your path)/PASCAL VOC/VOC2007 test/VOC2007\" where \"Annotations\" and \"JPEGImages\" folders are stored')
    parser.add_argument('--detections_filename', required=True, type=str,
        help='Path to detections pickle, e.g. \"./SSD_detections/SSD_ovthresh_0.015_single_detections_PASCAL_VOC_2007_test.pkl\"')
    parser.add_argument('--imagenames_filename', required=True, type=str,
        help='File to store images filenames of dataset, if compute weighted map, this file contains part of '
            'dataset filenames, e.g. \"./PASCAL_VOC_files/imagenames_2007_test.txt\"')
    parser.add_argument('--pickled_annots_filename', required=True, type=str,
        help='Pickle where annotations to compute mAP are stored, e.g. \"./PASCAL_VOC_files/annots_2007_test.pkl\"')
    parser.add_argument('--map_iou_threshold', type=float,
                        help='Jaccard coefficient value to compute mAP, default=0.5', default=0.5)
    return parser.parse_args(argv)


def read_imagenames(filename, dir):
    if not os.path.isfile(filename):
        print('Can not find file:', filename)
        imagenames = []
        for (dir, subdirs, files) in os.walk(dir):
            for file in files:
                if not file.startswith('.'):
                    imagenames.append(file)
        f = open(filename, 'w')
        for imagename in imagenames:
            f.write(imagename + '\n')
        f.close()
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
    return imagenames


def read_annotations(filename, dir, imagenames, dataset_name):

    if not os.path.isfile(filename):
        print('Can not find file:', filename)
        annotation_names = []
        for (dir, subdirs, files) in os.walk(dir):
            for file in files:
                if not file.startswith('.') and file.split('.')[0] + '.jpg' in imagenames:
                    annotation_names.append(file)
        # load annots
        recs = {}
        for i, annotation_name in enumerate(annotation_names):
            rec, imagename = dataset_parse_functions[dataset_name](os.path.join(dir, annotation_name))
            if imagename in imagenames:
                recs[imagename] = rec
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(annotation_names)))
        # save
        print('Saving cached annotations to {:s}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(recs, f, protocol=2)
    else:
        # load
        with open(filename, 'rb') as f:
            recs = pickle.load(f)
    return recs


def main(args):

    if os.path.isfile(args.detections_filename):
        with open(args.detections_filename, 'rb') as f:
            if sys.version_info[0] == 3:
                images_detections = pickle.load(f, encoding='latin1')
            else:
                images_detections = pickle.load(f)
    else:
        print('Detections filename was not found!')
        exit(1)

    annotations_dir = os.path.join(args.dataset_dir, 'Annotations/')
    images_dir = os.path.join(args.dataset_dir, 'JPEGImages/')

    imagenames = read_imagenames(args.imagenames_filename, images_dir)
    annotations = read_annotations(args.pickled_annots_filename, annotations_dir, imagenames, args.dataset_name)

    map_computation = Computation_mAP(None)
    map_computation.compute_map(args.dataset_name, args.dataset_dir, imagenames, annotations, images_detections,
                                args.map_iou_threshold)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))