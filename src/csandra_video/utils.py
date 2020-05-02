from PIL import Image
import numpy as np
import cv2
import requests
import json
import csandra_video.cv_camera as cvc
import numpy as np
import tfjs_graph_converter as tfjs
import tensorflow as tf
from pathlib import Path 

def start_video(n_graph):
    '''
    starts capturing the video from main camera 
    
    Parameters:
        n_graph (tf frozen graph): path to the model.json file
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
        # todo: remove to function
        # store data
        cv2.imwrite("test.jpg", frame)
        #results = eval_image(n_graph, data.tobytes())

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def start_stream():
    '''
    starts a video stream in background and returns the frames
    '''
    camera = cvc.CVCamera()
    frame = camera.get_frame()
    return replace_background(camera, frame)

def replace_background(n_camera,n_frame):
    img_buffer = np.frombuffer(n_frame, dtype=np.uint8)
    img = cv2.imdecode(img_buffer, flags=1)
    mask = get_segementation_mask(n_graph=n_camera.graph, n_image=img)
    # read in a "virtual background" (should be in 16:9 ratio)
    replacement_bg_raw = cv2.imread('backgroundtest.jpg')

    # resize to match the frame (width & height from before)
    height, width = 720, 1280
    replacement_bg = cv2.resize(replacement_bg_raw, (width, height))
    # combine the background and foreground, using the mask and its inverse
    inv_mask = np.bitwise_not(mask)
    f = np.bitwise_and(img, mask)
    inv_f = np.bitwise_and(replacement_bg, inv_mask)
    f = f + inv_f
    return cv2.imencode('.jpg', f)[1].tobytes()




def download_tfjs_model(n_modelpath=''):
    '''
    downlads a tfjs model to models folder

    Parameters:
        n_modelpath (str): path to the tjfs model based on https://storage.googleapis.com/tfjs-models
    
    Returns:
        Path to json file of the model
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
    # load all the weights
    for item in model_json['weightsManifest'][0]['paths']:
        # build the weight path string, based on the given modelpath
        modelpath = str.join('/',n_modelpath.split('/')[0:-1])
        weights_path = modelpath +'/'+item
        # do the get requests to download
        request_url = base_url+weights_path
        r = requests.get(request_url)
        data = r.content
        # make file
        newFile = open(dir + '/'+item, "wb")
        # write to file
        newFile.write(data)

    return dir+'/model.json'

def load_graph_model(n_modelpath=''):
    '''
    loads the graph model from json model

    Parameters:
        n_modelpath (str): path to the model.json file
    
    Returns:
        graph model

    '''
    if not n_modelpath:
        raise ValueError('request_url should not be an empty string')
    
    # convert model.json into graph model
    graph = tfjs.api.load_graph_model(n_modelpath)
    return graph

def eval_image(n_graph, n_image):
    '''
    evaluates an image and returns the tensorflow result array
    
    Parameters:
        n_graph (tf frozen graph): path to the model.json file
        n_image (numpy array): image to be observed

    Returns:
        Array length of 8:
            [displacement_bwd, displacement_fwd, heatmaps, longoffsets, offsets, partHeatmaps, segments, partOffsets]
    '''
    if not n_graph:
        raise ValueError('n_graph must be a graph model')
    if not n_image.any():
        raise ValueError('n_image must be an image')
    
    # load image into array
    height = n_image.shape[0]
    width = n_image.shape[1]
    # Create copy of image for processing
    x = np.copy(n_image)
    # add imagenet mean - extracted from body-pix source
    m = np.array([-123.15, -115.90, -103.06])
    x = np.add(x, m)
    n_image = x[tf.newaxis, ...]

    with tf.compat.v1.Session(graph=n_graph) as sess:
        input_tensor_names = tfjs.util.get_input_tensors(n_graph)
        #print(input_tensor_names)
        output_tensor_names = tfjs.util.get_output_tensors(n_graph)
        #print(output_tensor_names)
        input_tensor = n_graph.get_tensor_by_name(input_tensor_names[0])
        results = sess.run(output_tensor_names, feed_dict={
                        input_tensor: n_image})
    return results

def get_segementation_mask(n_graph=None, n_image=None):
    '''
    applies a segementation mask to an image
    
    Parameters:
        n_graph (tf frozen graph): path to the model.json file
        n_image (numpy array): image to be observed

    Returns:
        numpy array with mask applied
    '''
    results = eval_image(n_graph,n_image)
    
    segments = np.squeeze(results[6], 0)
    # Segmentation MASk
    segmentation_threshold = 0.7
    scores = tf.sigmoid(segments)
    mask = tf.math.greater(scores, tf.constant(segmentation_threshold))
    #print('maskshape', mask.shape)
    mask = tf.dtypes.cast(mask, tf.int32)
    mask = np.reshape(
        mask, (mask.shape[0], mask.shape[1]))
    #print('maskValue', mask[:][:])

    # Get image from mask
    mask_img = Image.fromarray(mask * 255)
    img = np.copy(n_image)
    
    # Apply resize to new image
    mask_img = mask_img.resize(
         (img.shape[1], img.shape[0]), Image.LANCZOS).convert("RGB")
    
    # store info to np.array
    mask_img = tf.keras.preprocessing.image.img_to_array(
         mask_img, dtype=np.uint8)

    # Apply mask with original input
    f =  np.array(mask_img)
    # f = np.bitwise_and(img, np.array(mask_img))
    # do some post processing on the result
    f = cv2.dilate(f, np.ones((1,1), np.uint8) , iterations=1)
    f = cv2.erode(f, np.ones((1,1), np.uint8) , iterations=1)
    #f = cv2.blur(f.astype(float), (30,30))
    # cv2.imwrite('color_img.jpg', f)
    return f



