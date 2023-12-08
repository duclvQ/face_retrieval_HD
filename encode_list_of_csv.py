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
from memory_profiler import profile
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
    
from pytube.exceptions import AgeRestrictedError
from http.client import IncompleteRead
from pytube.exceptions import VideoUnavailable
def download_video_from_youtube(link, path, retries=3):
    for _ in range(retries):
        try:
            yt = YouTube(link)
            video = yt.streams.get_highest_resolution()
            # save the name of the video as unicode and unsigned
            video_name = unicodedata.normalize('NFKD', video.title).encode('ascii', 'ignore').decode('utf-8')
            video_name = video_name.replace(' ', '_')
            video_name = video_name.replace('\"', '_')
            video_name = video_name.replace('\'', '_')
            video_name = video_name.replace('/', '_')
            video_name = video_name.replace(',', '_')
            # download the video
            video.download(path, filename=video_name+'.mp4')
            return os.path.join(path, video_name+'.mp4')
        except AgeRestrictedError:
            print(f"{bcolors.WARNING}Skipping age-restricted video: {link}")
            return None
        except IncompleteRead:
            print('Incomplete read, retrying download...')
        except VideoUnavailable:
            print(f"Video {link} is unavailable")
            return None
    print('Failed to download video after multiple attempts')
    return None

csv_folder = '/home/dev/face_retrieval/video_url_csv_files'
save_path = '/home/dev/face_retrieval/videos'

def get_all_available_csv_files(csv_folder):
    csv_files = list()
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(csv_folder, file))
    return csv_files
def check_if_url_is_in_database(url, collection_name):
    client = MongoClient('mongodb://localhost:27017/')
    db_name = "faceDB"
    collection_name = collection_name
    collection = client[db_name][collection_name]
    if collection.find_one({'url':url}) is None:
        return False
    return True

while True:
    
    list_of_csv_files = get_all_available_csv_files(csv_folder)
    if len(list_of_csv_files)==0:
        print('No csv file found, waiting for 100 seconds...')
        time.sleep(100)
        continue
    
    csv_file = list_of_csv_files[0]
    

    # read the csv file
    lines = pd.read_csv(csv_file)
    lines = lines['urls']


    os.makedirs(save_path, exist_ok=True)
    # download the video, process the video, delete the video
    P = ProcessVideo()
    for line in lines[:]:
        if ' ' in line:
            url = line.split(' ')[-1]
        else:
            url = line
        if check_if_url_is_in_database(url, 'allFaceRecords'):
            print(f"{bcolors.WARNING}Video {url} is already in the database, skipping...{line}")
            continue
        
        
        print(f"{bcolors.OKGREEN}Downloading video...{line}")
        file_path = download_video_from_youtube(url, save_path)
        if file_path == None:
            print(f'{bcolors.FAIL}Download failed, skipping...')
            continue
        print(f'{bcolors.OKGREEN}Downloaded video to: ', file_path)
        print('Processing video...')
        
        
        
        P.video_path = file_path
        P.video_URL = url
        P.capture_done.clear()
        # add time to save folder
        current_time = time.time()
        
        P.saving_folder = os.path.join('/home/dev/face_retrieval/cropped_faces/', P.video_path.split('/')[-1].split('.')[0]+'_'+str(current_time)
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
        
        
        
        print("Processing video done, deleting video...")
        os.remove(file_path)
        print(f"{bcolors.FAIL}Deleted video")
        print(f"{bcolors.HEADER}Processing next video...")
        gc.collect()
        # remove csv file after processing
        os.remove(csv_file)
    
    

        
        
        
        
    
    