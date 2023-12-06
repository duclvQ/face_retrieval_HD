from crop_face import CropFace
from face_encoder import FaceEncoder
from pymongo import MongoClient
import threading
import cv2
import time
import os

# import queue 
import queue

import sys
sys.path.append('./')
sys.path.append('../')
# Create a connection to the MongoDB server
# Create a connection to the MongoDB server
client = MongoClient('mongodb://localhost:27017/')

db_name = "faceDB"
collection_all_faces = "allFaceRecords"
collection_video_urls = "videoURLs"



class ProcessVideo:
    def __init__(self, stride=5, \
                    device=0, \
                    pretrained='vggface2',\
                    batch_size=32,
                    
                        ):
        self.video_path = None
        self.video_URL = None
        self.total_frames = 0
        self.video_FPS = None
        self.face_saving_path = None
        self.saving_folder = None
        
        self.client = None
        self.db_name = None
        #self.collection_name = None
        self._collection_all_faces = None
        self._collection_video_urls = None
        
        
        if self.client is None:
            Exception('MongoDB client is not set')
        if self.db_name is None:
            Exception('MongoDB database name is not set')

        if self._collection_all_faces is None:
            Exception('MongoDB collection is not set')
        
        
        
        
        self.capture_done = threading.Event()
        self.stride = stride
        self.device = device
        #self.pretrained = pretrained
        self.detector = CropFace()
        self.encoder = FaceEncoder()
        
    @staticmethod
    def get_first_two_digits(num: int) -> str:
        if num>=100:
            num/=10
        num_str = str(num)
        
        if '.' in num_str:
            num_str = num_str.replace('.', '')
        return num_str[:2].zfill(2)
    
    def frame_to_timecode(self, frame_num, fps) -> str:
        total_seconds = frame_num / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        miliseconds = self.get_first_two_digits((total_seconds - int(total_seconds))*1000)  
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{miliseconds}"    
    
    def read_video_thread(self, video_path, frame_queue):
        if not os.path.exists(video_path):
            Exception('Video path is not exist')
        if not os.path.exists(self.saving_folder):
            os.makedirs(self.saving_folder)
        self.video_path = video_path
        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.total_frames)
        self.video_FPS = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        temp_list = []
        previous_progress = 0
        while True:
            ret = cap.grab()
            #print(ret)
            frame_count +=1
            #print(frame_count)
            if ret==True:
                progress = int(frame_count/self.total_frames*100)
                if progress!=previous_progress:
                    #print(f'progress: {progress}%')
                    previous_progress = progress
                if frame_count % self.stride == 0:
                    #if frame_queue.qsize()>32: # Wait for 10ms to prevent running out of memory.
                    #    time.sleep(0.030)
                    ret, frame = cap.retrieve()
                    frame_queue.put((frame_count,frame))
                
               
            else:
                cap.release()
                # sleep for 100ms to ensure that all frames are processed
                time.sleep(0.1)
                break   
        self.capture_done.set()   
    
    def face_detection_thread(self, frame_queue, face_queue):
        frame_collection = []
        frame_num_collection = []
        while True:
            if frame_queue.qsize()>0:
                frame_num, frame = frame_queue.get()
                frame_collection.append(frame)
                frame_num_collection.append(frame_num)
                if len(frame_collection) == self.encoder.batch_size:
                    faces, faces_org = self.detector.crop_face_from_batch(frame_collection)
                   
                    for frame_num_, face, face_org in zip(frame_num_collection, faces, faces_org):
                        if face ==None:
                            continue
                        for face_num in range(face.shape[0]):
                            
                            #_face = np.expand_dims(face[face_num], axis=0)
                            _face = face[face_num]
                            _face_org = face_org[face_num]
                            #print(f'face shape of {face_num}',_face.shape)
                            face_queue.put((frame_num_, _face, _face_org))
                         # Print the progress in a single line
                        print(f'\rProcessing frame {frame_num_} out of {self.total_frames} ({(frame_num_ / self.total_frames) * 100:.2f}%)', end='')

                    frame_collection = []
                    frame_num_collection = []

            else:
                if self.capture_done.is_set():
                    break
                else:
                    # if the queue is empty, sleep for 10ms to prevent running out of memory
                    time.sleep(0.01)
        face_queue.put((None, None))
    
    def face_encoding_thread(self, face_queue):
        face_collection = []
        frame_num_collection = []
        face_org_collection = []
        while True:
            if face_queue.qsize()>0:
                frame_num, face, face_org = face_queue.get()
                if frame_num is None:
                    # None is the signal that the face detection thread is done
                    break
                face_collection.append(face)
                frame_num_collection.append(frame_num)
                face_org_collection.append(face_org)    
                if len(face_collection) == self.encoder.batch_size:
                    face_encoding = self.encoder(face_collection)
                    for idx, (frame_num_, face_encoding_, face_, face_org_) in enumerate(zip(frame_num_collection, face_encoding, face_collection, face_org_collection)):
                        # get face number in collection
                        face_saving_path = f'{self.saving_folder}/{frame_num_}_{idx}.jpg'
                        # save face_ to face_saving_path
                        ## convert torch tensor to numpy array
                        try:
                            face_org_ = face_org_.detach().cpu()
                        except:
                            pass
                        face_org_ = face_org_.numpy()
                        
                        # convert size to hwc
                        face_org_ = face_org_.transpose(1,2,0)
                        cv2.imwrite(face_saving_path, face_org_)
                        # save face_encoding_ to all_faces database
                        self._collection_all_faces.insert_one({'video_name': self.video_path, \
                                                                'video_URL': self.video_URL,\
                                                                'video_FPS': self.video_FPS, \
                                                                'face_path': face_saving_path, \
                                                                'frame_num': frame_num_, \
                                                                'face_num':  idx, \
                                                                'face_encoding': face_encoding_.tolist(), \
                                                                'timecode': self.frame_to_timecode(frame_num_, self.video_FPS)
                                               })
                        
                    face_collection = []
                    frame_num_collection = []
            else:
                # if the queue is empty, sleep for 10ms to prevent running out of memory
                time.sleep(0.01)
                
        #print('Done')
    
    def main_thread(self):
        
        frame_queue = queue.Queue()
        face_queue = queue.Queue()
        t1 = threading.Thread(target=self.read_video_thread, args=(self.video_path, frame_queue))
        t2 = threading.Thread(target=self.face_detection_thread, args=(frame_queue, face_queue))
        t3 = threading.Thread(target=self.face_encoding_thread, args=(face_queue,))
        t1.start()
        #print('Capture thread started')
        t2.start()
        #print('Detection thread started')
        t3.start()
        #print('Encoding thread started')
        t1.join()
        t2.join()
        t3.join()
        #print('Done')

if __name__ == "__main__":
 
    P = ProcessVideo()
    P.video_path = '/home/dev/face_retrieval/videos/video_1.mp4'
    P.video_URL = 'http://example.com/' + P.video_path

    P.saving_folder = os.path.join('home/dev/face_retriveal/cropped_faces', P.video_path.split('/')[-1].split('.')[0])
    P.client = client
    P.db_name = db_name
    #P.collection_name = collection_name
    collection_all_faces = "allFaceRecords"
    collection_video_urls = "videoURLs"
    _collection_all_faces = P.client[P.db_name][collection_all_faces]
    _collection_video_urls = P.client[P.db_name][collection_video_urls]
    P._collection_all_faces = _collection_all_faces
    P._collection_video_urls = _collection_video_urls
    
    
    P.main_thread()           
                    
                