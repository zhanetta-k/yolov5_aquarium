## Marine Creatures Detection with YOLOv5

The Aquarium dataset has been collected for object detection with YOLOv5 from the sources:

 1. [Aquarium Dataset](https://public.roboflow.com/object-detection/aquarium)

 It consists of images collected from two aquariums in the US:
- The Henry Doorly Zoo in Omaha (October 16, 2020);
- National Aquarium in Baltimore (November 14, 2020).

 2. [Flickr](https://www.flickr.com)

The dataset is available on [Roboflow](https://universe.roboflow.com/aquariumdataset/aquarium-dataset-78xdt).
It includes 1,054 images with 6,284 bounding box annotations and encompasses 7 classes (fish, jellyfish, penguin, puffin, shark, starfish, stingray).

Two pre-trained YOLOv5 models are applied to the dataset: YOLOv5s and YOLOv5m.

### Documentation

**Install dependencies**

For a start, install YOLOv5 dependencies.

```bash
pip install -r requirements.txt

git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

**Redefine paths**

Adjust the paths to input train/val/test data in the .yaml file.

```bash
cd src
python train_infer.py --path_yaml ../input/data/
```

**Training**

`train.py` and model's weights are in the `yolov5` directory. When launching training, also indicate:
- path to data.yaml
- image size
- number of epochs
- batch size
- folder name to store results  # the default is 'exp'
- path to a project folder for saving the results folder

```bash
python yolov5/train.py --data input/data/data.yaml --weights yolov5/yolov5m.pt --img 640 \
     --epochs 40 --batch-size 16 --name results_yolo5m --cache --project models/runs/train
```

Keep the _best model(s)_ in a separate folder by moving them to the `best_models_storage` folder.

```bash
python src/train_infer.py dir_move_from results_yolo5s
```

**Inference**

`detect.py` runs inference on a variety of sources.
Store results under the project folder `models/runs/detect`.

```bash
python yolov5/detect.py --weights models/best_models_storage/best_yolo5m.pt \
    --source input/data/test/images --img 640 --conf 0.4 --name results_yolo5m \
    --project models/runs/detect
```

### Fast API

`main.py` is the entry point of the application with 3 endpoints defined in there.
`segmentation.py` defines the methods on getting the best model and resizing the image.

1. Start the project.
```bash
uvicorn main:app --reload
```
2. Navigate to [http://0.0.0.0:8000/docs#/](http://0.0.0.0:8000/docs#/) in your browser.
3. Execute the endpoint by clicking the `try it out`.
