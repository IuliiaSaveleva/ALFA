# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib.pyplot as plt


def list_image_sets():
    """
    List all the image sets from Pascal VOC. Don't bother computing
    this on the fly, just remember it. It's faster.
    """
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


def plt_bboxes_different_detectors(img, classes_dict, scores_dict, iou_dict, bboxes_dict, figsize=(10,10), linewidth=3.0):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    class_names = list_image_sets()
    # fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')

    colors = {'gt': (255, 255, 255),
              'SSD': (0, 255, 0),
              'DeNet': (255, 0, 0),
              'ALFA': (0, 0, 255)
              }

    for detector_name in ['gt', 'SSD', 'DeNet', 'ALFA']:
        classes = classes_dict[detector_name]
        scores = scores_dict[detector_name]
        ious = iou_dict[detector_name]
        bboxes = bboxes_dict[detector_name]
        color = colors[detector_name]
        color = (color[0] / 255., color[1] / 255., color[2] / 255.)
        for i in range(classes.shape[0]):
            cls_id = int(classes[i])
            if cls_id >= 0:
                score = scores[i][1:][cls_id]
                iou = ious[i]
                xmin = bboxes[i, 0]
                ymin = bboxes[i, 1]
                xmax = bboxes[i, 2]
                ymax = bboxes[i, 3]
                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=color,
                                     linewidth=linewidth)
                plt.gca().add_patch(rect)
                class_name = class_names[cls_id]
                if detector_name != 'gt':
                    plt.gca().text(xmin, ymin - 2,
                                   'IoU={:.2f} | {:s}'.format(iou, detector_name),
                                   bbox=dict(facecolor=color, alpha=0.5),
                                   fontsize=12, color='white')
    plt.show()
