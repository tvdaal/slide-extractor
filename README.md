# Description

This application extracts distinct frames from a directory of video recordings and saves them to a PDF file.

I needed to extract (PowerPoint) slides from a large collection of
recordings that also contain parts where only the speaker is showing. Since
I could not find any good tooling for this online, I decided to build a Python script for this myself. With this custom solution, I was able to extract all relevant frames with ease. The script automatically removes duplicates and for any given title on the slide, it only keep the last one (as that one typically has most information on it). These operations rely on the comparison of pixel values across different areas of the videos' frames.

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
