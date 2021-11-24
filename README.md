# Description

This Python application extracts distinct frames from a directory of video
recordings and saves them to a single PDF file.

The script `extract_slides.py` automatically removes frames that only show the
speaker, as well as duplicates. Furthermore, for any sequence of slides
(defined by slides that share the same title), it only keeps the last one.
The reason for this is that the last slide in a sequence typically provides
most information. Finally, the algorithms are robust to pixel noise and rely
on the comparison of pixel statistics across different areas and color channels
of the videos' frames.

Note that some functionalities are application specific. This dependence has
been isolated in the function `select_frames` as much as possible.

# Background

For an online course that I took, I wanted to extract (PowerPoint) slides from
a large collection of recordings (ca. 20 hours of video material). Since I
could not find any good tooling for this online, I decided to one myself.
With this custom solution in Python, I was able to extract all relevant
frames with ease. From a few million video frames, I successfully extracted
close to a thousand slides in PDF format.

# Installation

This repository has been tested for Python 3.5. To get started, run the
following commands:

```
git clone https://github.com/tvdaal/slide-extractor.git
cd slide-extractor
conda env create -f environment.yml
conda activate py35-slides
```

The virtual conda environment called 'py35-slides' contains all necessary
packages and dependencies.
