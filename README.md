# ALFA: Agglomerative Late Fusion Algorithm for Object Detection

By Evgenii Razinkov, Iuliia Saveleva, Jiří Matas

This is the original implementation of "ALFA: Agglomerative Late Fusion Algorithm for Object Detection" https://arxiv.org/pdf/1907.06067.pdf.

![](paper_image/ALFA_result.png "ALFA result")

Image from PASCAL VOC 2007 test set.
Bounding boxes and IoU with ground truth:
DeNet – red (IoU = 0.75);
SSD – green (IoU = 0.77);
ALFA – blue (IoU = 0.93).
Ground truth bounding box is in white.


<table>
    <thead>
        <tr>
            <th rowspan=2>Detector</th>
            <th rowspan=2>fps(Hz)</th>
            <th colspan=2>PASCAL VOC 2007</th>
            <th colspan=2>PASCAL VOC 2012</th>
        </tr>
        <tr>
            <th>mAP-s(%)</th>
            <th>mAP(%)</th>
            <th>mAP-s(%)</th>
            <th>mAP(%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Faster R-CNN</th>
            <th><sub>7</sub></th>
            <th><sub>77.95</sub></th>
            <th><sub>78.83</sub></th>
            <th><sub>72.72</sub></th>
            <th><sub>73.59</sub></th>
        </tr>
        <tr>
            <th>SSD</th>
            <th><sub>59</sub></th>
            <th><sub>79.26</sub></th>
            <th><sub>80.37</sub></th>
            <th><sub>72.89</sub></th>
            <th><sub>74.17</sub></th>
        </tr>
        <tr>
            <th>DeNet</th>
            <th><sub>33</sub></th>
            <th><sub>78.09</sub></th>
            <th><sub>79.26</sub></th>
            <th><sub>70.73</sub></th>
            <th><sub>72.10</sub></th>
        </tr>
        <tr>
            <th colspan=6>SSD + DeNet</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>20.3</sub></th>
            <th><sub>83.12</sub></th>
            <th><sub>83.53</sub></th>
            <th><sub>76.80</sub></th>
            <th><sub>77.37</sub></th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>16.9</sub></th>
            <th><sub>83.29</sub></th>
            <th><sub>83.88</sub></th>
            <th><sub>75.74</sub></th>
            <th><sub>76.38</sub></th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>20.6</sub></th>
            <th><sub>83.87</sub></th>
            <th><sub>84.32</sub></th>
            <th><sub>76.97</sub></th>
            <th><sub>77.82</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>18.1</sub></th>
            <th><sub>84.16</sub></th>
            <th><sub>84.41</sub></th>
            <th><sub>77.52</sub></th>
            <th><sub>77.98</sub></th>
        </tr>
                <tr>
            <th colspan=6>SSD + DeNet + Faster R-CNN</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>5.2</sub></th>
            <th><sub>84.31</sub></th>
            <th><sub>84.43</sub></th>
            <th><sub>78.11</sub></th>
            <th><sub>78.34</sub></th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>4.7</sub></th>
            <th><sub>84.97</sub></th>
            <th><sub>85.24</sub></th>
            <th><sub>75.71</sub></th>
            <th><sub>75.69</sub></th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>5.2</sub></th>
            <th><sub>85.78</sub></th>
            <th><sub>85.67</sub></th>
            <th><sub>79.16</sub></th>
            <th><sub>79.42</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>5.0</sub></th>
            <th><sub>85.90</sub></th>
            <th><sub>85.72</sub></th>
            <th><sub>79.41</sub></th>
            <th><sub>79.47</sub></th>
        </tr>
    </tbody>
</table>

This repository was tested on python 2.7 and 3.5, platforms Linux and Mac OS.


## Before you start

Download this project and unarchive files in "./SSD_Detections", "./DeNet_detections", "./Faster_R-CNN_detections" to
run the scripts

Download PASCAL VOC 2007 and PASCAL VOC 2012 dataset and change "path/to/VOC2007 test/VOC2007",
 and "path/to/VOC2012 test/VOC2012" in bash commands according to location of datasets on your computer.


## Draw image from paper

To draw image from paper:
```bash
python ./draw_paper_pic.py \
--alfa_parameters_json="./Algorithm_parameters/ALFA/SSD_DeNet_0.015_single_cross_validation_parameters_2007.json"
```


## Reproduce fps and PASCAL VOC 2007 results

Results on PASCAL VOC 2007 differ slightly from paper results due to randomness of a cross-validation procedure.

### Base Detectors

* Fps values for base detectors were taken from their papers.

* Evaluate on PASCAL VOC 2007

```bash
python ./cross_validate_base_detector.py \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --dataset_dir value to real path /VOC2007 test/VOC2007 then type:
```bash
--detections_filename="./SSD_detections/SSD_ovthresh_0.015_single_detections_PASCAL_VOC_2007_test.pkl"
```
Change --detections_filename to path from table below.

<table>
    <thead>
        <tr>
            <th>Detector</th>
            <th>Detections</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>SSD</th>
            <th><sub>./SSD_detections/SSD_ovthresh_0.015_single_detections_PASCAL_VOC_2007_test.pkl</sub></th>
        </tr>
        <tr>
            <th>DeNet</th>
            <th><sub>./DeNet_detections/DeNet_ovthresh_0.015_single_detections_PASCAL_VOC_2007_test.pkl</sub></th>
        </tr>
        <tr>
            <th>Faster R-CNN</th>
            <th><sub>./Faster_R-CNN_detections/Faster_R-CNN_ovthresh_0.015_single_detections_PASCAL_VOC_2007_test.pkl</sub></th>
        </tr>
    </tbody>
</table>

### ALFA

* Compute fps

```bash
python ./validate_ALFA.py \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --dataset_dir value to real path /VOC2007 test/VOC2007 then type:
```bash
--alfa_parameters_json="./Algorithm_parameters/ALFA/SSD_DeNet_0.05_single_cross_validation_parameters_2007.json"
```
Change last part of the path --alfa_parameters_json to value from table below.

* Evaluate on PASCAL VOC 2007

```bash
python ./cross_validate_ALFA.py \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --dataset_dir value to real path /VOC2007 test/VOC2007 then type:
```bash
--alfa_parameters_json="./Algorithm_parameters/ALFA/SSD_DeNet_0.05_single_cross_validation_parameters_2007.json"
```
Change last part of the path --alfa_parameters_json to value from table below.

<table>
    <thead>
        <tr>
            <th>Algorithm</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th colspan=2>SSD + DeNet, mAP-s(%)</th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>SSD_DeNet_0.05_single_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>SSD_DeNet_0.015_single_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th colspan=5>SSD + DeNet, mAP(%)</th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>SSD_DeNet_0.05_multiple_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>SSD_DeNet_0.015_multiple_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP-s(%)</th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.05_single_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_single_cross_validation_parameters_2007.json</sub></th>
        </tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP(%)</th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.05_multiple_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_multiple_cross_validation_parameters_2007.json</sub></th>
        </tr>
    </tbody>
</table>

### DBF

* Compute fps

```bash
python ./validate_DBF.py \
--validation_imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--validation_pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--test_imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--test_pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--validation_dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --validation_dataset_dir value to real path /VOC2007 test/VOC2007 then type:
```bash
--test_dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --test_dataset_dir value to real path /VOC2007 test/VOC2007 then type:
```bash
--dbf_parameters_json="./Algorithm_parameters/DBF/SSD_DeNet_0.015_single_cross_validation_parameters_2007.json"
```
Change last part of the path --dbf_parameters_json to value from table below.

* Evaluate on PASCAL VOC 2007

```bash
python ./cross_validate_DBF.py \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --dataset_dir value to real path /VOC2007 test/VOC2007 then type:
```bash
--dbf_parameters_json="./Algorithm_parameters/DBF/SSD_DeNet_0.015_single_cross_validation_parameters_2007.json"
```
Change last part of the path --dbf_parameters_json to value from table below.

<table>
    <thead>
        <tr>
            <th>Algorithm</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th colspan=2>SSD + DeNet, mAP-s(%)</th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>SSD_DeNet_0.015_single_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th colspan=5>SSD + DeNet, mAP(%)</th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>SSD_DeNet_0.015_multiple_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP-s(%)</th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_single_cross_validation_parameters_2007.json</sub></th>
        </tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP(%)</th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_multiple_cross_validation_parameters_2007.json</sub></th>
        </tr>
    </tbody>
</table>

### NMS

* Compute fps

```bash
python ./validate_NMS.py \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --dataset_dir value to real path /VOC2007 test/VOC2007 then type:
```bash
--nms_parameters_json="./Algorithm_parameters/NMS/SSD_DeNet_0.015_single_cross_validation_parameters_2007.json"
```
Change last part of the path --nms_parameters_json to value from table below.

* Evaluate on PASCAL VOC 2007

```bash
python ./cross_validate_NMS.py \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --dataset_dir value to real path /VOC2007 test/VOC2007 then type:
```bash
--nms_parameters_json="./Algorithm_parameters/NMS/SSD_DeNet_0.015_single_cross_validation_parameters_2007.json"
```
Change last part of the path --nms_parameters_json to value from table below.

<table>
    <thead>
        <tr>
            <th>Algorithm</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th colspan=2>SSD + DeNet, mAP-s(%)</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>SSD_DeNet_0.015_single_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th colspan=5>SSD + DeNet, mAP(%)</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>SSD_DeNet_0.015_multiple_cross_validation_parameters_2007.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP-s(%)</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_single_cross_validation_parameters_2007.json</sub></th>
        </tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP(%)</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_multiple_cross_validation_parameters_2007.json</sub></th>
        </tr>
    </tbody>
</table>


## Reproduce PASCAL VOC 2012 results

### Base Detectors

* Convert "detections_path.pkl" to PASCAL VOC 2012 submission format by running:
```bash
python ./detections_to_PASCAL_VOC_2012_submission.py \
--detections_filename="detections_path.pkl" \
```
Change --detections_filename value from table below then type:
```bash
--submission_folder="path/to/submission_folder"
```
Change --submission_folder value to real path on your computer

<table>
    <thead>
        <tr>
            <th>Detector</th>
            <th>Detections</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>SSD</th>
            <th><sub>./SSD_detections/SSD_ovthresh_0.015_single_detections_PASCAL_VOC_2012_test.pkl</sub></th>
        </tr>
        <tr>
            <th>DeNet</th>
            <th><sub>./DeNet_detections/DeNet_ovthresh_0.015_single_detections_PASCAL_VOC_2012_test.pkl</sub></th>
        </tr>
        <tr>
            <th>Faster R-CNN</th>
            <th><sub>./Faster_R-CNN_detections/Faster_R-CNN_ovthresh_0.015_single_detections_PASCAL_VOC_2012_test.pkl</sub></th>
        </tr>
    </tbody>
</table>

* Archive "path/to/submission_folder/results" as .tar.gz

* Upload "path/to/submission_folder/results.tar.gz" to PASCAL VOC 2012 evaluation server

### ALFA

* Get ALFA detections running the following command:
```bash
python ./validate_ALFA.py \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2012_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2012_test.pkl" \
--dataset_dir="path/to/VOC2012 test/VOC2012" \
```
Change --dataset_dir value to real path to /VOC2012 test/VOC2012 then type:
```bash
--alfa_parameters_json="./Algorithm_parameters/ALFA/SSD_DeNet_0.05_single_cross_validation_parameters_2012.json" \
```
Change --alfa_parameters_json to value value from table below then type:
```bash
--output_filename="path/to/output_filename.pkl"
```
Change --output_filename to real path on your computer

<table>
    <thead>
        <tr>
            <th>Algorithm</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th colspan=2>SSD + DeNet, mAP-s(%)</th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>SSD_DeNet_0.05_single_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>SSD_DeNet_0.015_single_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet, mAP(%)</th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>SSD_DeNet_0.05_multiple_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>SSD_DeNet_0.015_multiple_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP-s(%)</th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.05_single_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_single_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP(%)</th>
        </tr>
        <tr>
            <th>Fast ALFA</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.05_multiple_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th>ALFA</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_multiple_cross_validation_parameters_2012.json</sub></th>
        </tr>
    </tbody>
</table>


* Convert "path/to/output_filename.pkl" to PASCAL VOC 2012 submission format by running:
```bash
python ./detections_to_PASCAL_VOC_2012_submission.py \
--detections_filename="detections_path.pkl" \
```
Change --detections_filename value from table below then type:
```bash
--submission_folder="path/to/submission_folder"
```
Change --submission_folder value to real path on your computer

* Archive "path/to/submission_folder/results" as .tar.gz

* Upload "path/to/submission_folder/results.tar.gz" to PASCAL VOC 2012 evaluation server

### DBF

* Get DBF detections running the following command:
```bash
python ./validate_DBF.py \
--validation_imagenames_filename="./PASCAL_VOC_files/imagenames_2007_test.txt" \
--validation_pickled_annots_filename="./PASCAL_VOC_files/annots_2007_test.pkl" \
--test_imagenames_filename="./PASCAL_VOC_files/imagenames_2012_test.txt" \
--test_pickled_annots_filename="./PASCAL_VOC_files/annots_2012_test.pkl" \
--validation_dataset_dir="path/to/VOC2007 test/VOC2007" \
```
Change --validation_dataset_dir value to real path to /VOC2007 test/VOC2007 then type:
```bash
--test_dataset_dir="path/to/VOC2012 test/VOC2012" \
```
Change --test_dataset_dir value to real path to /VOC2012 test/VOC2012 then type:
```bash
--dbf_parameters_json="./Algorithm_parameters/DBF/SSD_DeNet_0.015_single_cross_validation_parameters_2012.json" \
```
Change --dbf_parameters_json to value value from table below then type:
```bash
--output_filename="path/to/output_filename.pkl"
```
Change --output_filename to real path on your computer

<table>
    <thead>
        <tr>
            <th>Algorithm</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th colspan=2>SSD + DeNet, mAP-s(%)</th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>SSD_DeNet_0.015_single_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet, mAP(%)</th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>SSD_DeNet_0.015_multiple_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP-s(%)</th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_single_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP(%)</th>
        </tr>
        <tr>
            <th>DBF</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_multiple_cross_validation_parameters_2012.json</sub></th>
        </tr>
    </tbody>
</table>

* Convert "path/to/output_filename.pkl" to PASCAL VOC 2012 submission format by running:
```bash
python ./detections_to_PASCAL_VOC_2012_submission.py \
--detections_filename="detections_path.pkl" \
```
Change --detections_filename value from table below then type:
```bash
--submission_folder="path/to/submission_folder"
```
Change --submission_folder value to real path on your computer

* Archive "path/to/submission_folder/results" as .tar.gz

* Upload "path/to/submission_folder/results.tar.gz" to PASCAL VOC 2012 evaluation server


### NMS

* Get NMS detections running the following command:
```bash
python ./validate_NMS.py \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2012_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2012_test.pkl" \
--dataset_dir="path/to/VOC2012 test/VOC2012" \
```
Change --dataset_dir value to real path to /VOC2012 test/VOC2012 then type:
```bash
--nms_parameters_json="./Algorithm_parameters/NMS/SSD_DeNet_0.015_single_cross_validation_parameters_2012.json" \
```
Change --nms_parameters_json to value value from table below then type:
```bash
--output_filename="path/to/output_filename.pkl"
```
Change --output_filename to real path on your computer

To get different detectors combinations results in NMS use parameters from the table:


<table>
    <thead>
        <tr>
            <th>Algorithm</th>
            <th>Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th colspan=2>SSD + DeNet, mAP-s(%)</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>SSD_DeNet_0.015_single_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet, mAP(%)</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>SSD_DeNet_0.015_multiple_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP-s(%)</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_single_cross_validation_parameters_2012.json</sub></th>
        </tr>
        <tr>
            <th colspan=2>SSD + DeNet + Faster R-CNN, mAP(%)</th>
        </tr>
        <tr>
            <th>NMS</th>
            <th><sub>SSD_DeNet_Faster_R-CNN_0.015_multiple_cross_validation_parameters_2012.json</sub></th>
        </tr>
    </tbody>
</table>

* Convert "path/to/output_filename.pkl" to PASCAL VOC 2012 submission format by running:
```bash
python ./detections_to_PASCAL_VOC_2012_submission.py \
--detections_filename="detections_path.pkl" \
```
Change --detections_filename value from table below then type:
```bash
--submission_folder="path/to/submission_folder"
```
Change --submission_folder value to real path on your computer

* Archive "path/to/submission_folder/results" as .tar.gz

* Upload "path/to/submission_folder/results.tar.gz" to PASCAL VOC 2012 evaluation server
