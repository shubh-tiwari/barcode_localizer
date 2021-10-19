# barcode_localizer

This repo contains scripts to train model for barcode code classification and localization. Selective search algorithm is used for region proposal and barcode is identified from proposed regions using trained classification model.

### Sample training data
This is a very small sample of the actual dataset. Dataset contains two classes - barcode regions and non-barcode regions. Non-barcode region can be anything like random text, apmty space, packets, documents. Data contains both actual and augmented data

![image](https://github.com/shubh-tiwari/barcode_localizer/blob/main/images/training_data.png)

### Sample output
The pipeline is predicting good results on barcode present on packets, cans and documents. A small piece of result is added below :

![image](https://github.com/shubh-tiwari/barcode_localizer/blob/main/images/result1.JPG)
![image](https://github.com/shubh-tiwari/barcode_localizer/blob/main/images/result2.JPG)

### Inference on video
To inference on video, the following script can be used : [Video inference script](https://github.com/shubh-tiwari/barcode_localizer/blob/main/scripts/video_infer.py)
This script will take model path and original video as input and will return video with bounding box.

### Comment
This task is done to explore the selective search based region proposal. The same task can be accomplished much more effiently using yolo, SSD or any other newer object detection method.
