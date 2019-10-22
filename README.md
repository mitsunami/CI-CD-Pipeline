# CI-CD-Pipeline

This application run Mask-RCNN Deep Learning model on a clusters. If you call a make_prediction request with image ID, then the object detection model runs and return labels detected in the imaged specified.

### Usage
- Once repository is pushed, jenkins runs lint source codes, build docker image and push it to Docker Hub
- Rolling update on EKS kubernetes clusters

### Run locally
To test locally, execute `./run_docker.sh`
