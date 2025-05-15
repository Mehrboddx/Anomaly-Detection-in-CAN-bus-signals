#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME="$1"

declare -A DATASET_URLS
DATASET_URLS["syncan"]="https://github.com/etas/SynCAN/archive/refs/heads/master.zip"
DATASET_URLS["road"]="https://zenodo.org/records/10462796/files/road.zip"

if [[ -z "${DATASET_URLS[$DATASET_NAME]}" ]]; then
    echo "Error: Dataset '$DATASET_NAME' not found."
    echo -n "Available datasets: "
    printf "%s " "${!DATASET_URLS[@]}"
    echo
    exit 1
fi

# Create Datasets folder if it doesn't exist
mkdir -p Datasets

DATASET_URL=${DATASET_URLS[$DATASET_NAME]}
OUTPUT_FILE="${DATASET_NAME}.zip"
EXTRACT_FOLDER="Datasets/${DATASET_NAME}"

echo "Downloading $DATASET_NAME..."
wget -O $OUTPUT_FILE $DATASET_URL

echo "Extracting $DATASET_NAME..."
unzip -o $OUTPUT_FILE -d Datasets/
rm $OUTPUT_FILE

# Extraction logic based on dataset name
if [ "$DATASET_NAME" == "syncan" ]; then
    # Find the folder that was unzipped and rename it to "syncan"
    FOLDER_NAME=$(find Datasets/ -maxdepth 1 -type d -name "*SynCAN*" | head -n 1)
    
    if [ -z "$FOLDER_NAME" ]; then
        echo "Error: Could not find the extracted SynCAN folder."
        exit 1
    fi
    
    mv "$FOLDER_NAME" "$EXTRACT_FOLDER"
    
    cd $EXTRACT_FOLDER
    echo "Unzipped training dataset in syncan/ambient"
    unzip 'train_*.zip' -d ambient
    echo "Unzipped testing dataset in syncan/attack"
    unzip 'test_*.zip' -d attack

elif [ "$DATASET_NAME" == "road" ]; then
    # Find the unzipped folder and move it
    FOLDER_NAME=$(find Datasets/ -maxdepth 1 -type d -name "*road*" | head -n 1)
    
    if [ -z "$FOLDER_NAME" ]; then
        echo "Error: Could not find the extracted ROAD folder."
        exit 1
    fi
    
    mv "$FOLDER_NAME" "$EXTRACT_FOLDER"
    cd $EXTRACT_FOLDER
    echo "ROAD dataset is ready in the '$EXTRACT_FOLDER' folder!"

    # Remove any zip files if they exist
    find . -maxdepth 1 -name "*.zip" -exec rm -f {} \;

    # Check if "signal_extractions/ambient/" exists before removing
    if [ -d "signal_extractions/ambient/" ]; then
        echo "Cleaning up 'signal_extractions/ambient/'..."
        find signal_extractions/ambient/ -type f ! -name 'ambient_*' -exec rm {} +
    fi
    
    echo "ROAD Data Downloaded!"

else
    echo "Dataset extraction logic not implemented for '$DATASET_NAME'."
    exit 1
fi

echo "Dataset '$DATASET_NAME' is ready in the '$EXTRACT_FOLDER' folder!"

