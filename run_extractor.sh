#!/usr/bin/bash

# This script is specific to the application and can be used to process
# different folders with video content in a particular way, employing the
# Python application to extract distinct slides from a collection of video
# recordings. This bash script takes one argument, namely the input directory.

script="$(pwd)/extract_slides.py"
cd "$1"

for folder in */
do
  echo -e "\nInvestigating recordings in ${folder}..."
  out_path="${folder}${folder%/}.pdf"
  python "${script}" "${folder}" "${out_path}"
done

rm -r frames
