#!/usr/bin/env python3
"""This script is used to extract frames from videos and save them to a PDF.

It accepts various command line arguments.

    Example of how to run:

    python extract_slides.py <path_to_input_data_dir> <out_path>.pdf
    python extract_slides.py "/Users/tvdaal/Dropbox/Tom/CS/Data science/Coursera - MLOps specialization (2021)/Course 1 - Introduction to Machine Learning in Production/Week 1 - Overview of the ML Lifecycle and Deployment" slides.pdf

    For more information on the various optional parameters run:

    python extract_slides.py --help
"""


# Standard Python libraries:
from argparse import ArgumentParser
import glob
import os
from typing import List

# Third-party libraries:
import cv2
from icecream import ic
from PIL import Image


def frames_to_pdf(im_paths: List[str], out_path: str) -> bool:
    """Combines images into a PDF file.

    Args:
        im_paths: Collection of images that need to be combined into a PDF.
        out_path: Path to the output PDF file.

    Returns:
        True if PDF generation successful, otherwise False.
    """

    # Save images to list:
    for im_path in im_paths:
        im = Image.open(im_path)
        im_list.append(im)

    # Save frames to PDF:
    im_list[0].save(
        out_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=im_list[1:],
    )
    success = os.path.isfile(out_path)

    return success


def video_to_frames(iteration: int, path: str, sec: int):
    """Extracts frames from a single video file into "frames" directory.

    The iteration variable is simply a number that is used in the naming of
    the saved frames.

    Args:
        iteration: Iteration number.
        path: Path to video.
        sec: Number of seconds between saved frames.
    """

    # Convert video to OpenCV object and obtain frame rate:
    video = cv2.VideoCapture(path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_thr = fps * sec

    # Empty directory for frames or create it if it does not exist yet:
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
      os.makedirs(frames_dir)
    else:
        pattern = frames_dir + "/*"
        files = glob.glob(pattern)
        for file in files:
            os.remove(file)

    # Extract frames from video, one by one:
    frames_count = 0
    success = 1
    im_list = []
    while success:
        success, frame = video.read()
        if (frames_count % frames_thr == 0) and (frame is not None):
            frame_name = "frame_{}_{}.jpg".format(iteration, frames_count)
            im_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(im_path, frame)
        frames_count += 1


def get_file_paths(input_dir: str, ext: str) -> List[str]:
    """Obtain a list of all files in the given directory tree.

    Args:
        input_dir: The path to the input directory.
        ext: Extension of files to save.

    Returns:
        List of full paths to all files that are contained in the input
        directory, including subdirectories.
    """

    list_of_files = os.listdir(input_dir)
    all_files = []
    for entry in list_of_files:
        full_path = os.path.join(input_dir, entry)

        # If entry is a directory then get the list of files in this directory:
        if os.path.isdir(full_path):
            all_files = all_files + get_file_paths(full_path)
        elif full_path[-3:] == ext[-3:]:
            all_files.append(full_path)
        else:
            continue

    return all_files


def dir_to_frames(input_dir: str, ext: str, sec: int):
    """Extracts frames from video files in input_dir.

    Args:
        input_dir: Path to the input directory.
        ext: Extension of video files.
        sec: Number of seconds between saved frames.
    """

    video_paths = get_file_paths(input_dir, ext)
    for i, path in enumerate(video_paths):
        video_to_frames(i, path, sec)


def dir_to_pdf(input_dir: str, out_path: str, ext: str, sec: int) -> bool:
    """Extracts frames from video files in input_dir and saves them to a PDF.

    Args:
        input_dir: Path to the input directory.
        out_path: Path to the output PDF file.
        ext: Extension of video files.
        sec: Number of seconds between saved frames.

    Returns:
        True if PDF generation successful, otherwise False.
    """

    dir_to_frames(input_dir, ext, sec)
    temp_dir = "frames"
    im_paths = get_file_paths(temp_dir, "jpg")
    success = frames_to_pdf(im_paths, out_path)

    return success


def construct_argparser() -> ArgumentParser:
    """Constructs an argument parser for command line use.

    Returns:
        An argument parser.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Specify the path to the input data directory.",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Specify the path to the output PDF file.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        help="Specify the extension of the video files.",
    )
    parser.add_argument(
        "--sec",
        type=int,
        default=10,
        help="Specify the time in seconds between frame captures",
    )

    return parser


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    # Generate the PDF from the collection of video files:
    success = dir_to_pdf(args.input_dir, args.out_path, args.ext, args.sec)
    if not success:
        print("-\n> Generation failed :(")
    else:
        print("\n-> Generation successful :)")
