import cv2
import csandra_video.utils as utils
import time
import threading
from _thread import get_ident

class NewFrameEvent(object):
    '''
    inform and sends new frame info, when send from camera frame
    '''
    def __init__(self):
        # dictionary of events
        self.events = {}
    
    def wait_next_frame(self):
        '''
        invoked by client thread and signal to wait
        '''
        id = get_ident()
        if id not in self.events:
            # enter new client connection
            self.events[id] = [threading.Event(), time.time()]
        return self.events[id][0].wait()
    
    def set_frame(self):
        '''
        invoked by the camera
        '''
        now = time.time()
        id_remove = None
        for id, event in self.events.items():
            if not event[0].isSet():
                # if event is not set, set it and update timestamp
                event[0].set()
                event[1] = now
            elif now - event[1] > 5:
                # remove id, if last update is older than 5 seconds
                id_remove = id
        if id_remove:
            del self.events[id_remove]

    def clear(self):
        '''
        clear after processing the frame
        '''
        self.events[get_ident()][0].clear()

class CVCamera(object):
    '''
    class for camera actions
    '''
    thread = None
    frame = None
    last_client_access = 0
    event = NewFrameEvent()
    graph = None

    def __init__(self):
        '''
        start background thread for camera processing
        '''
        # load ml model
        CVCamera.graph = graph = utils.load_graph_model(n_modelpath='models/bodypix/resnet50/float/model_stride16/model.json')
        if CVCamera.thread is None:
            CVCamera.last_client_access = time.time()
            
            CVCamera.thread = threading.Thread(target=self._start_video)
            CVCamera.thread.start()

            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        '''
        Get and returns the current frame from camera

        Returns:
            Frame (bytes)
        '''
        CVCamera.last_client_access = time.time()
        CVCamera.event.wait_next_frame()
        CVCamera.event.clear()

        return CVCamera.frame 

    def _start_video(self):
        '''
        starts the camera action in background
        '''
        print('start method')
        frames_iterator = self._capture_frames()
        for frame in frames_iterator:
            CVCamera.frame = frame
            CVCamera.event.set_frame()
            time.sleep(0)

            if time.time() - CVCamera.last_client_access > 10:
                frames_iterator.close()
                break
        CVCamera.thread = None

    def _capture_frames(self):
        '''
        return the image frame capture from camera
        as an iterator

        Returns:
            Frame (bytes)
        '''
        cap = cv2.VideoCapture(0)        
        if not cap.isOpened():
            raise RuntimeError('Could not start camera.')
        # configure camera for 720p @ 60 FPS
        height, width = 720, 1280
        cap.set(cv2.CAP_PROP_FRAME_WIDTH ,width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        cap.set(cv2.CAP_PROP_FPS, 60)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            yield cv2.imencode('.jpg', frame)[1].tobytes()