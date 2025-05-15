# Anomaly-Detection-in-CAN-bus-signals
This is the github repository for my thesis project regarding Anomaly Detection in can bus signals focusing mostly on masquerade attacks and usage of LLMs in this field. 

The datasets are to be downloaded using the sh file in Datasets. In case of interest in using these datasets follow below instructions:
## Clone Project
```
git clone https://github.com/Mehrboddx/Anomaly-Detection-in-CAN-bus-signals.git
cd Anomaly-Detection-in-CAN-bus-signals
```
## Download SynCAN

Syncan is pretty straightforward just do:

```
bash download_dataset.sh "syncan"
```
It will be ready to use.

## Download ROAD

```
bash download_dataset.sh "road"
python road_labeler.py --logs-folder "road/signal_extractions/attack"
```
And the dataset will be ready to use.