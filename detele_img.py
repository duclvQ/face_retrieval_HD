

from pymongo import MongoClient
import os
client = MongoClient('mongodb://localhost:27017/')
collection_all_faces = client["faceDB"]["allFaceRecords"]
# 
for document in collection_all_faces.find():
    image_path = document['face_path']
    frame_num = document['frame_num']
   
    if int(frame_num)%30!=0:
        # delete image with image_path
            try:
              os.remove(image_path)
              # delete this record
            except:
              print("Not found: ",image_path)
            collection_all_faces.delete_one({r"face_path":image_path})
            print("\rdelete %s"%image_path, end="")
        
        
    