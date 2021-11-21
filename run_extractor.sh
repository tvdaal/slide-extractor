#!/usr/bin/bash

script="$(pwd)/extract_slides.py"
cd "$1"

for folder in */
do
  echo -e "\nInvestigating recordings in ${folder}..."
  out_path="${folder}${folder%/}.pdf"
  python "${script}" "${folder}" "${out_path}"
done
