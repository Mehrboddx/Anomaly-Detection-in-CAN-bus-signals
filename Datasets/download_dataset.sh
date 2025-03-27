#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME="$1"

declare -A DATASET_URLS
DATASET_URLS["SynCAN"]="https://github.com/etas/SynCAN/archive/refs/heads/main.zip"
DATASET_URLS["OTIDS"]="https://www.dropbox.com/scl/fo/8kll7yvbgogkp0vahowvm/ADhDIC8LRFL8wHUexib3C3w?rlkey=43n570cnodtq6yls139r4yvn7&e=1&st=em61xu2d&dl=0"

if [[ -z "${DATASET_URLS[$DATASET_NAME]}" ]]; then
    echo "Error: Dataset '$DATASET_NAME' not found."
    echo -n "Available datasets: "
    printf "%s " "${!DATASET_URLS[@]}"
    echo
    exit 1
fi

DATASET_URL=${DATASET_URLS[$DATASET_NAME]}
OUTPUT_FILE="${DATASET_NAME}.zip"
EXTRACT_FOLDER="${DATASET_NAME}"

echo "Downloading $DATASET_NAME..."
wget -O $OUTPUT_FILE $DATASET_URL

echo "Extracting $DATASET_NAME..."
unzip -o $OUTPUT_FILE -d $EXTRACT_FOLDER

rm $OUTPUT_FILE

echo "Dataset '$DATASET_NAME' is ready in the '$EXTRACT_FOLDER' folder!"