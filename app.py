from flask import Flask, request, jsonify, render_template
from flask.logging import create_logger
import logging

import os
import sys
import random
import numpy as np
import skimage.io
import cv2
import colorsys

# Import Mask RCNN
ROOT_DIR = os.path.abspath("./Mask_RCNN/")
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
#from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/")) # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
            


app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, scores=None):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
 
    colors = random_colors(N)
 
    masked_image = image.copy()
    for i in range(N):
        color = colors[i]
 
        # Bounding box
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        camera_color = (color[0] * 255, color[1] * 255, color[2] * 255)
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), camera_color , 1)
 
        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        #x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        camera_font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(masked_image,caption,(x1, y1),camera_font, 1, camera_color)
 
        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)
 
    return masked_image.astype(np.uint8)

"""
def scale(payload):
    # Scales Payload
    
    #LOG.info(f"Scaling Payload: \n{payload}")
    LOG.info("Scaling Payload: \n%s", payload)
    scaler = StandardScaler().fit(payload.astype(float))
    scaled_adhoc_predict = scaler.transform(payload.astype(float))
    return scaled_adhoc_predict
"""

@app.route("/")
def home():
    html = f"<h3>DeepLearning Object Detection</h3>"
    return html.format(format)

@app.route('/showresult', methods = ['GET'])
def showresult():
    """/ Show result: /showresult"""
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    """Performs an sklearn prediction
        
        input looks like:
        {
        "IMAGE":{"0":1}
        }
        
        result looks like:
        { "prediction": [ <val> ] }
        
        """
    
    """
    # Logging the input payload
    json_payload = request.json
    #LOG.info(f"JSON payload: \n{json_payload}")
    LOG.info("JSON payload: \n%s", json_payload)
    inference_payload = pd.DataFrame(json_payload)
    #LOG.info(f"Inference payload DataFrame: \n{inference_payload}")
    LOG.info("Inference payload DataFrame: \n%s", inference_payload)
    # scale the input
    scaled_payload = scale(inference_payload)
    # get an output prediction from the pretrained model, clf
    prediction = list(clf.predict(scaled_payload))
    # Log the output prediction value
    LOG.info("Optput prediction value: %s", prediction)
    return jsonify({'prediction': prediction})
    """

    # Logging the input payload
    json_payload = request.json
    #LOG.info(f"JSON payload: \n{json_payload}")
    LOG.info("JSON payload: \n%s", json_payload)

    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    #LOG.info("Input image list: \n%s", file_names)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                            class_names, r['scores'])
    display = display_instances(image, r['rois'], r['masks'], r['class_ids'], r['scores'])
    cv2.imwrite("src/img/mrcnn.jpg", display)

    labels = ''
    for class_id in r['class_ids']:
        labels = labels + ', ' + class_names[class_id]
    LOG.info("Output class value: %s", labels)

    #return jsonify({'prediction': prediction})
    return jsonify({'labels': labels})

if __name__ == "__main__":


    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    model.keras_model._make_predict_function()

    app.run(host='0.0.0.0', port=80, debug=True) # specify port=80
