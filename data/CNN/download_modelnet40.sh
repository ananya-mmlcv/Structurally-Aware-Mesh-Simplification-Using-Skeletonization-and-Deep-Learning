#!/bin/bash

# Define the URL of the zip file
url="http://modelnet.cs.princeton.edu/ModelNet40.zip"

# Define the filename (extracted from the URL)
filename=$(basename "$url")

# Download the zip file
wget "$url"

# Check if download was successful (exit code 0 indicates success)
if [[ $? -eq 0 ]]; then
  echo "Download complete: $filename"
  # Extract the zip file
  unzip "$filename"
  echo "Extracted: ModelNet40"
else
  echo "Download failed!"
fi
