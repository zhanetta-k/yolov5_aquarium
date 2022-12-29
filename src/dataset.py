import os
import yaml
import glob
from collections import Counter
import argparse
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from roboflow import Roboflow


DATA_DIR = 'input/data/'


def download_data(api_key: str, workspace: str,
                  project_name: str, version: int,
                  data_dir='input/data/'):
    """
      Download dataset for YOLOv5 PyTorch models
      to data directory from Roboflow.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    project.version(version).download("yolov5", data_dir)


def define_paths2imgs(data_dir=DATA_DIR):
    """Define paths to data folders with images."""
    train_dir = os.path.join(data_dir, "train/images/")
    val_dir = os.path.join(data_dir, "valid/images/")
    test_dir = os.path.join(data_dir, "test/images/")
    return train_dir, val_dir, test_dir


def define_paths2labels(data_dir=DATA_DIR):
    """Define paths to data folders with labels."""
    train_dir = os.path.join(data_dir, "train/labels/")
    val_dir = os.path.join(data_dir, "valid/labels/")
    test_dir = os.path.join(data_dir, "test/labels/")
    return train_dir, val_dir, test_dir


# YAML file
def get_classes(data_dir=DATA_DIR, yaml_file='data.yaml'):
    """Return the list of class names."""

    with open(os.path.join(data_dir, f'{yaml_file}'), 'r') as f:
        file = yaml.load(f, Loader=yaml.FullLoader)

    num_classes = file["nc"]
    class_names = file["names"]
    return num_classes, class_names


def count_images(data_dir):
    """Count number of images in a set."""

    num_imgs = len(os.listdir(data_dir))
    return num_imgs


def count_total(train_dir, valid_dir, test_dir):
    """Count total number of images in a dataset."""

    train_imgs = count_images(train_dir)
    valid_imgs = count_images(valid_dir)
    test_imgs = count_images(test_dir)
    total = np.sum([train_imgs, valid_imgs, test_imgs])

    print(f"The dataset consists of {total} images.")
    print("-"*19)
    print(f"Train: {train_imgs} images.")
    print(f"Validation: {valid_imgs} images.")
    print(f"Test: {test_imgs} images.")


def get_class_breakdown(class_names: list, imgs_dir: str) -> dict:
    """Create a dictionary with class breakdown
    by ground truth bounding boxes."""

    mask = f"{imgs_dir}*.jpg"

    hist = [0] * len(class_names)
    num_images = 0

    for img_file in glob.glob(mask):
        label_file = img_file.replace('images', 'labels').replace(
                                      '.jpg', '.txt')
        label_file = label_file.replace('.jpg', '.txt')
        num_images = num_images + 1

        with open(label_file, "r") as f:
            label_lines = f.readlines()
            for line in label_lines:
                label = int(line[0])
                hist[label] = hist[label]+1

    class_dict = {}
    for c in enumerate(class_names):
        class_dict[f"{c[1]}"] = hist[c[0]]
    return class_dict


def get_stats(class_dict: dict, title: str = "Stats"):
    """Show info on class breakdown in a set."""

    print(f"{title}", "-"*20, sep="\n")
    print("Class breakdown (by ground truth bboxes):")

    for k, v in class_dict.items():
        print(f"{k}: {v}")


def count_annotations(train_classes: dict,
                      val_classes: dict,
                      test_classes: dict):
    """Create dataframe with the total number
    of annotations for each class."""

    class_breakdown = np.sum([Counter(train_classes),
                              Counter(val_classes),
                              Counter(test_classes)])

    data = {"label": [k for k in class_breakdown.keys()],
            "amount": [i for i in class_breakdown.values()]}
    df = pd.DataFrame(data)
    df.sort_values(by="amount", ascending=True, inplace=True)
    return df


def is_img_empty(labels_dir: str):
    """Check if images with no objects exist
       by confirming if its size is 0 bytes."""

    empty_files = []
    files = glob.glob(os.path.join(labels_dir, '*.txt'))
    for file in files:
        empty_file = os.stat(file).st_size == 0
        if empty_file:
            empty_files.append(file)

    empty_file_name = []
    for file in empty_files:
        name = [file[i:-4]for i, s in enumerate(file) if s == 'I']
        empty_file_name.append(name)
    return [f[0] for f in empty_file_name]


def visualize_classes(x, y, name: str, color: str = None,
                      dir_save: str = "../notebooks/imgs"):
    """Visualize classes distribution with a horizontal plot."""
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    if color is None:
        color = ['#0641c8']*len(x)

    fig = go.Figure(go.Bar(x=x,
                           y=y,
                           marker=dict(color=color),
                           text=x,
                           textposition='inside',
                           orientation='h'))

    fig.update_layout(width=800,
                      height=370,
                      title=name)
    fig.show('svg')
    fig.write_image(f'{dir_save}classes_annotated.png')


# Construct the argument parser.
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for downloading dataset \
                             from Roboflow.')
    parser.add_argument('--workspace', type=str, required=True,
                        help='Workspace name of the project on Roboflow.')
    parser.add_argument('--project_name', type=str, required=True,
                        help='Project name on Roboflow.')
    parser.add_argument('--version', type=int, required=True,
                        help='Project version on Roboflow.')
    parser.add_argument('--data_dir', default=str(DATA_DIR), type=str,
                        help='Path to the folder for downloading \
                             the dataset.')
    return parser.parse_args()


def main(args):
    data_dir = args.data_dir
    download_data(api_key=args.api_key,
                  workspace=args.workspace,
                  project_name=args.project_name,
                  version=args.version,
                  data_dir=data_dir)
    print("\nDataset downloaded.\n")

    train_imgs_dir, val_imgs_dir, test_imgs_dir = define_paths2imgs(data_dir)
    train_labels_dir, _, _ = define_paths2labels(data_dir)

    # Number of images, train/test split info.
    count_total(train_imgs_dir,
                val_imgs_dir,
                test_imgs_dir)

    num_classes, classes = get_classes(data_dir=data_dir)
    print(f"\nThe dataset encompasses {num_classes} classes:",
          classes, sep="\n")

    # Images with no objects.
    emp_imgs = is_img_empty(train_labels_dir)
    print(f"\nThere are {len(emp_imgs)} background images with no objects.")

    # Dicts with class breakdown by ground truth bboxes.
    train_classes = get_class_breakdown(classes, train_imgs_dir)
    val_classes = get_class_breakdown(classes, val_imgs_dir)
    test_classes = get_class_breakdown(classes, test_imgs_dir)

    # Dataframe with the total number of annotations for each class.
    df = count_annotations(train_classes, val_classes, test_classes)
    print(f"\nNumber of annotations: {df.amount.sum()}\n")

    # Stats for train and valid sets.
    get_stats(train_classes, "Training Set Stats")
    print()
    get_stats(val_classes, "Validation Set Stats")


if __name__ == "__main__":
    args = set_args()
    main(args)
