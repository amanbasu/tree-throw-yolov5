## Train

1. `config-train.json`: contains the training configurations. 
    test_images - Provide the images that should belong to the test set. The rest of the images are randomly split into training and validation set.
    size - size of each crop that the model can handle.
    stride - stride size of the crops.
    image_path - folder path that stores all the images
    dataset_path - folder path that stores the files for YOLO
    (Note: please make sure that the `dataset_path` doesn't contain 'images' or 'labels' in its absolute path, or it would interfere with `dataloaders.py`)
2. Label images using Roboflow.
    a. generate high-pass crops from your images and label them using roboflow.
    `python generate_images.py --config config-train.json --save_jpg`
    (save slope or msrm instead of hpass by simply changing the channel index in the code)
    b. upload these images to https://roboflow.com and label them.
    c. export images after labeling and put them in the `labels/` folder under `dataset_path`
    d. rename label filenames to maintain consistency
    `python rename_labels.py --config config-train.json`
    (filename changes from `in2017_01251380_12_0_1200_jpg.rf.744492ae1ec8a75db8501c71d139e30d.txt` to `in2017_01251380_12_0_1200.txt`)
3. Prepare data for YOLO.
    a. generate labels for overlapping crops by combining adjacent labels
    `python generate_strided_labels.py --config config-train.json`
    b. generate tif images to be used as input by the model
    `python generate_images.py --config config-train.json --train`
    c. convert to YOLO format
    `python to_yolo_format.py --config config-train.json --train`
    (splits the data into train/valid/test sets and prepares them to be used by YOLO)
    This code also prints out the path for the yaml file that you should copy for the next step
4. Train YOLO.
    a. copy the yaml file path from the previous step
    b. train YOLO
    `python train.py --img 400 --batch 64 --epochs 600 --data <yaml-path> --name yolov5 --augment --cache`
    (change `--data` to the yaml path you copied and `--name` to the folder where you want to save the weights)
5. Test YOLO.
    Run this once the training ends.
    `python val.py --weights runs/train/yolov5/weights/best.pt --data <yaml-path> --img 400 --name yolov5 --save-txt --task test --conf-thres 0.1 --single-cls --save-conf`
    (change `--data` to the yaml path you copied earlier)

## Inference
Considering that now you have got a trained model, you can use it to predict the tree throws in the images.

1. `config-inference.json`: contains the inference configurations. 
    size - size of each crop that the model can handle.
    image_path - folder path that stores all the inference images
    dataset_path - folder path that stores the files for YOLO
2. Prepare data for YOLO.
    b. generate tif images to be used as input by the model
    `python generate_images.py --config config-inference.json`
    c. convert to YOLO format
    `python to_yolo_format.py --config config-inference.json`
    (prepares test images to be used by YOLO)
    This code also prints out the path for the yaml file that you should copy for the next step
3. Infer through YOLO.
    `python val.py --weights <model-weight> --data <yaml-path> --img 400 --name <folder-name> --save-txt --task test --conf-thres 0.1 --single-cls --save-conf`
    (change `--data` to the yaml path you copied earlier, `--weights` to the path of the model weights and `--name` to the folder where you want to save the labels)
    The labels will be saved at `runs/val/<folder-name>/labels/`
4. Post-process the labels.
    Consolidate the labels for all the crops and convert relative coordinates to absolute coordinates by using the geo-tagged source file.
    `python box_to_coords.py --config config-inference.json`
    (use argument `--save_hpass` to save a geo-referenced hpass image of the source file)

## Ultralytics framework insights

### `train.py`
Important script arguments
    `--img`: input image size
    `--batch`: batch size
    `--epochs`: number of epochs
    `--data`: path to the yaml file used for reading the data
    `--hyp`: path to the hyperparameter file, default to `data/hyps/hyp.tree-throw.yaml`
    `--resume`: this flag resumes training from the last checkpoint if stopped before the specified number of epochs
    `--evolve`: used for hyperparameter optimization/tuning
    `--cache`: increases the training speed by caching data
    `--optimizer`: select between 'SGD', 'Adam', and 'AdamW'
    `--augment`: use data augmentation while training

### `val.py`
Important script arguments
    `--weights`: path to the best model weights
    `--data`: path to the yaml file used for reading the data
    `--img`: input image size
    `--save-txt`: use this argument to save predicted labels
    `--conf-thres`: confidence threshold for predictions, predictions < conf-thres would be ignored
    `--iou-thres`: iou threshold for non-max supression. If two predictions have iou > iou-thres (overlap), the one with lower confidence score would be ignored
    `--single-cls`: treat the problem as a single class problem
    `--save-conf`: save confidence scores with boxes
    `--max-det`: maximum number of detections per image, defaults to 300

### `utils/augmentations.py`
Augmentations used for training
    Blur: prob 0.1
    ToGray: prob 0.1 (merges all channel data)
    CLAHE: prob 0.1 (applies Contrast Limited Adaptive Histogram Equalization to the image)
    RandomBrightnessContrast: prob 0.1
    RandomGamma: prob 0.1
    HorizontalFlip: prob 0.25
    VerticalFlip: prob 0.25
    PixelDropout: prob 0.1 (zeros out random pixels)
    RandomRotate90: prob 0.25
    Sharpen: prob 0.1
    GaussNoise: prob 0.1
    ISONoise: prob 0.1
*Note: no image normalization is done in YOLOv5*

### `utils/dataloaders.py`
1. `read_tif()`: function added to read the tif files.
2. `img2label_paths()`: reads the images and then tries to find the corresponding label file. If the label file is not found, it is assumed that there are no boxes.
3. `class LoadImagesAndLabels`: loads the images and labels and checks if they are properly formatted. Some of the code has been removed to support tif files.
    `check_cache_ram()`: commented out the code that loads images and checks for the cache size. A constant size of n * 480000 is given. Doesn't affect anything, just housekeeping stuff.
    `__getitem__()`: commented out `augment_hsv()` as we dont have RGB channels. Flip up-down and left-right is controlled in `utils/augmentations.py`.
    `load_image()`: removed the code that relates to the RGB images. Just read tif files and normalize them between 0 and 255.
4. `verify_image_label()`: comment out the code that verifies RGB images.

### `utils/metrics.py`
Contains the `fitness()` function that is used for evaluating the model's performance. The default one works fine for us.

### `utils/loss.py`
Contains code for all the loss functions, interesting to see how they are implemented.
