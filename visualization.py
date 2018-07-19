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

import random
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm


def list_image_sets():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


def plt_bboxes(img, classes, scores, bboxes, linewidth=1.5):

    class_names = list_image_sets()
    plt.imshow(img)
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = bboxes[i, 0]
            ymin = bboxes[i, 1]
            xmax = bboxes[i, 2]
            ymax = bboxes[i, 3]
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = class_names[cls_id]
            score = max(score[1:])
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f} |'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()


def plt_bboxes_from_different_sources(img, labels_dict, scores_dict, bboxes_dict, linewidth=3.0):

    class_names = list_image_sets()
    color_num = 0
    plt.imshow(img)
    plt.axis('off')
    for detector_name in ['gt', 'SSD_unique']:
        labels = labels_dict[detector_name]
        scores = scores_dict[detector_name]
        bboxes = bboxes_dict[detector_name]
        color = colors_tableau[color_num]
        color = (color[0] / 255., color[1] / 255., color[2] / 255.)
        color_num += 1
        for i in range(len(labels)):
            cls_id = int(labels[i])
            if cls_id >= 0:
                score = scores[i][1:][cls_id]
                xmin = bboxes[i][0]
                ymin = bboxes[i][1]
                xmax = bboxes[i][2]
                ymax = bboxes[i][3]
                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=color,
                                     linewidth=linewidth)
                plt.gca().add_patch(rect)
                class_name = class_names[cls_id]
                if detector_name != 'gt':
                    plt.gca().text(xmin, ymin - 2,
                                   '{:s} {:.2f} | {:s}'.format(class_name, score, detector_name),
                                   bbox=dict(facecolor=color, alpha=0.5),
                                   fontsize=12, color='white')
                else:
                    plt.gca().text(xmin, ymin - 2,
                                   '{:s} | {:s}'.format(class_name, detector_name),
                                   bbox=dict(facecolor=color, alpha=0.5),
                                   fontsize=12, color='white')
    plt.show()