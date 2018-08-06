# ALFA: Agglomerative Late Fusion Algorithm for Object Detection

By Evgenii Razinkov, Iuliia Saveleva, Jiří Matas

The paper "ALFA: Agglomerative Late Fusion Algorithm for Object Detection" is accepted 
to ICPR 2018, the link and description to the method will be added here as soon as 
the paper become available.

Results on PASCAL VOC 2007 are not reproducible due to randomness of a cross-validation procedure.

## Evaluation on Pascal VOC 2012

To reproduce the results of ALFA algorithm on PASCAL VOC 2012 run the following command:
```bash
python ./validate_ALFA.py \
--dataset_dir="path/to/VOC2012 test/VOC2012" \
--imagenames_filename="./PASCAL_VOC_files/imagenames_2012_test.txt" \
--pickled_annots_filename="./PASCAL_VOC_files/annots_2012_test.pkl" \
--alfa_parameters_json="./Cross_validation_ALFA_parameters/SSD_DeNet_ALFA_0.05_single_cross_validation_parameters_2012.json" \
--output_filename="path/to/output_filename.pkl"
```

| Algorithm | Detectors  | single detection | Parameters |
|--------|:---------:|:------:| :------:|
| Fast ALFA | SSD + DeNet | Yes  | SSD_DeNet_ALFA_0.05_single_cross_validation_parameters_2012.json |
| ALFA | SSD + DeNet | Yes |  SSD_DeNet_ALFA_0.015_single_cross_validation_parameters_2012.json |
| Fast ALFA | SSD + DeNet + Faster R-CNN | Yes | SSD_DeNet_Faster_R-CNN_ALFA_0.05_single_cross_validation_parameters_2012.json |
| ALFA | SSD + DeNet + Faster R-CNN | Yes | SSD_DeNet_Faster_R-CNN_ALFA_0.015_single_cross_validation_parameters_2012.json |
| Fast ALFA | SSD + DeNet | No | SSD_DeNet_ALFA_0.05_multiple_cross_validation_parameters_2012.json |
| ALFA | SSD + DeNet | No |  SSD_DeNet_ALFA_0.015_multiple_cross_validation_parameters_2012.json |
| Fast ALFA | SSD + DeNet + Faster R-CNN | No | SSD_DeNet_Faster_R-CNN_ALFA_0.05_multiple_cross_validation_parameters_2012.json |
| ALFA | SSD + DeNet + Faster R-CNN | No | SSD_DeNet_Faster_R-CNN_ALFA_0.015_multiple_cross_validation_parameters_2012.json |