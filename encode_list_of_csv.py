# read a csv file, each line is a link to a video, download the video, and then save it to a folder
# process the downloaded video, after that, delete the video and move on to the next video

from face_encoder_from_video_path import ProcessVideo
from pytube import YouTube
from pytube.exceptions import AgeRestrictedError
import pandas as pd
from pytube import YouTube
import unicodedata
import os
import gc
import time
from pymongo import MongoClient
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def download_video_from_youtube(link, path):
    try:
        yt = YouTube(link)
        video = yt.streams.get_highest_resolution()
        # save the name of the video as unicode and unsigned
        video_name = unicodedata.normalize('NFKD', video.title).encode('ascii', 'ignore').decode('utf-8')
        video_name = video_name.replace(' ', '_')
        video_name = video_name.replace('\"', '_')
        video_name = video_name.replace('\'', '_')
        # download the video
        video.download(path, filename=video_name+'.mp4')
        return os.path.join(path, video_name+'.mp4')
    except AgeRestrictedError:
        print(f"{bcolors.WARNING}Skipping age-restricted video: {link}")
        # download the video
        #video.download(path)
        return None
  
csv_file = 'vtv24_url.csv'
save_path = '/home/dev/face_retrieval/videos'

# read the csv file
lines = pd.read_csv(csv_file)
lines = lines['urls']

os.makedirs(save_path, exist_ok=True)
# download the video, process the video, delete the video
for line in lines:
    
    
    url = line.split(' ')[-1]
    print(f"{bcolors.OKGREEN}Downloading video...{line}")
    file_path = download_video_from_youtube(url, save_path)
    if file_path == None:
        print(f'{bcolors.FAIL}Download failed, skipping...')
        continue
    print(f'{bcolors.OKGREEN}Downloaded video to: ', file_path)
    print('Processing video...')
    P = ProcessVideo()
    P.video_path = file_path
    P.video_URL = url

    P.saving_folder = os.path.join('/home/dev/face_retrieval/cropped_faces/', P.video_path.split('/')[-1].split('.')[0])
    os.makedirs('/home/dev/face_retrieval/cropped_faces/', exist_ok=True)
    P.client = MongoClient('mongodb://localhost:27017/')
    P.db_name = "faceDB"
    #P.collection_name = collection_name
    collection_all_faces = "allFaceRecords"
    collection_video_urls = "videoURLs"
    _collection_all_faces = P.client[P.db_name][collection_all_faces]
    _collection_video_urls = P.client[P.db_name][collection_video_urls]
    P._collection_all_faces = _collection_all_faces
    P._collection_video_urls = _collection_video_urls
    
    P.main_thread()
    del P
    gc.set_threshold(1000, 15, 15)
    print("Processing video done, deleting video...")
    os.remove(file_path)
    print(f"{bcolors.FAIL}Deleted video")
    print(f"{bcolors.HEADER}Processing next video...")
    time.sleep(1)
    
    

        
        
        
        
    
    