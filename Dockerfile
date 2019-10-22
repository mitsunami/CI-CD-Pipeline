FROM python:3.7.3-stretch

## Step 1:
# Create a working directory
WORKDIR /app

## Step 2:
# Copy source code to working directory
COPY . app.py /app/

## Step 3:
# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --upgrade pip && pip install --trusted-host pypi.python.org -r requirements.txt
#RUN apt-get update; apt-get install nginx; 
RUN git clone https://github.com/matterport/Mask_RCNN
RUN git clone https://github.com/cocodataset/cocoapi
WORKDIR /app/cocoapi/PythonAPI
RUN python3 setup.py install
WORKDIR /app

## Step 4:
# Expose port 80
EXPOSE 80

## Step 5:
# Run app.py at container launch
CMD ["python3", "app.py"]
