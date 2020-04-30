import numpy as np
import cv2
import requests
import json
from pathlib import Path 

def start_video():
    '''
    starts capturing the video from main camera 
    '''
    cap = cv2.VideoCapture(0)
    # configure camera for 720p @ 60 FPS
    height, width = 720, 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH ,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def download_tfjs_model(n_modelpath=''):
    '''
    downlads a tfjs model to models folder

    Parameters:
        n_modelpath (str): path to the tjfs model based on https://storage.googleapis.com/tfjs-models
    '''
    # set base url to tfjs models path
    base_url = "https://storage.googleapis.com/tfjs-models/savedmodel/"

    if not n_modelpath:
        raise ValueError('request_url should not be an empty string')
 
    # get build request url and call get request
    request_url = base_url + n_modelpath + '.json'
    r = requests.get(request_url)
    # load json from response result
    model_json = r.json()
    # set directory
    dir = 'models/'+n_modelpath.replace('-','_')
    # create directory, if it not exists
    Path(dir).mkdir(parents=True, exist_ok=True)
    # finaly save or overwrite the downloaded model
    json_file = open(dir+'/model.json', 'w')
    json.dump(model_json, json_file)
    json_file.close()


#download_tfjs_model(n_modelpath='bodypix/resnet50/float/model-stride16')
