import os
import shutil
import yaml
import glob
import argparse
from base64 import b64encode

import numpy as np
import pandas as pd
from IPython.display import Image, HTML, display


# YAML file
def change_paths(path_yaml="input/data/",
                 train_dir="../input/data/train/images",
                 val_dir="../input/data/valid/images",
                 test_dir="../input/data/test/images"):
    """Tweak paths to images directories in the file `data.yaml`."""

    # Load YAML file and adjust it.
    with open(f"{path_yaml}data.yaml", "r") as f_yaml:
        file = yaml.load(f_yaml, Loader=yaml.FullLoader)

    file.update({'train': train_dir,
                 'val': val_dir,
                 'test': test_dir})

    # Write adjustments to a file.
    with open(f"{path_yaml}data.yaml", "w") as f:
        yaml.dump(file, f)


def set_results_dir(dir_name: str):
    """Set the results directory name."""
    res_dir = f"results_{dir_name}"

    print(f"Outputs are saved to directory: {res_dir} ...")
    return res_dir


def move_best_model(res_dir):
    """Move the best model to 'best_models' folder."""

    best_models_folder = "models/best_models_storage"
    if not os.path.exists(best_models_folder):
        os.makedirs(best_models_folder)

    current_f = f"models/runs/train/{res_dir}/weights/best.pt"
    new_f = f"{best_models_folder}/best_{res_dir[8:]}.pt"
    shutil.move(current_f, new_f)
    return new_f


def visualize(model_dir, batch_num, val=True, labels=True, width=900):
    """Visualize images from training and validation stages.

       batch_num: range(0, 3)"""

    train_val_dir = "models/runs/train/"
    path = os.path.join(train_val_dir, model_dir)

    if val:
        if labels:
            print("Validation: ground truth data\n")
            display(Image(filename=f"{path}/val_batch{batch_num}_labels.jpg",
                    width=width))
        else:
            print("Validation: predicted data\n")
            display(Image(filename=f"{path}/val_batch{batch_num}_pred.jpg",
                    width=width))

    else:
        print("Ground truth augmented training data\n")
        display(Image(filename=f"{path}/train_batch{batch_num}.jpg",
                width=width))


def visualize_test_imgs(test_res_dir, num_imgs=5):
    """Display test images with inferences."""

    images = glob.glob(f"{test_res_dir}/*.jpg")
    total = len(images)

    if num_imgs > total:
        print("The number exceeds the amount of images. \
              Total number of images: {total}")
    else:
        random_nums = np.random.randint(0, total, size=num_imgs)
        for num in random_nums:
            display(Image(filename=images[num]))
            print("\n")


def get_max(csv_file):
    """Get the max mAP values from results."""
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.replace(' ', '')

    mAP_1 = np.round(df['metrics/mAP_0.5'].max(), 2)
    mAP_2 = np.round(df['metrics/mAP_0.5:0.95'].max(), 2)
    return mAP_1, mAP_2


def visualize_video(video_dir, filename):
    """Display video in jupyter notebook."""

    save_path = os.path.join(video_dir, f"{filename}.mp4")
    # compressed video path
    compressed_path = os.path.join(video_dir, f"{filename}_compressed.mp4")

    os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")
    # show video
    mp4 = open(compressed_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=400 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)


# Construct the argument parser.
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_yaml', help='Redefine paths \
                        to train/val/test in the .yaml')
    parser.add_argument('--dir_move_from', type=str,
                        help='Folder name where results stored: \
                              e.g."results_yolo5s". It will be moved \
                              to "models/best_models_storage" folder.')
    return parser.parse_args()


if __name__ == "__main__":

    args = set_args()
    for k, v in args.__dict__.items():
        if k == 'path_yaml' and v is not None:
            # Since YOLOv5 model (by Ultralytics) searches
            # for dataset inside the root folder 'yolov5',
            # tweak paths to image folders in the .yaml file.
            change_paths(v)
            print("\nThe paths to image folders adjusted in the .yaml file:")
            with open(os.path.join(args.path_yaml, "data.yaml"), 'r') as file:
                f = yaml.load(file, Loader=yaml.FullLoader)
            print(f"Train: {f['train']}\nVal: {f['val']}\nTest: {f['test']}\n")

        elif k == 'dir_move_from' and v is not None:
            moved_to = move_best_model(v)
            print("The best model was moved:")
            print(f"{v} --> {moved_to}\n")
