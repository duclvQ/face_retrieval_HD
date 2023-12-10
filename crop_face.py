from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
import cv2
import time
import os

class CropFace:
    def __init__(self):
        self.total_number_frames = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.get_available_size_of_gpu() < 1000:
            print('GPU memory is not enough for face detection.')
            device = 'cpu'
        print('Running on device: {}'.format(device))
        
        self.mtcnn = MTCNN(image_size=182, margin=10, min_face_size=130, thresholds=[0.8, 0.9, 0.9],keep_all=True, factor=0.709, post_process=False,device=device)  # initializing mtcnn for face detection
        self.save_dir = None
     
    
    def get_available_size_of_gpu(self):
        return torch.cuda.get_device_properties(0).total_memory
    
    def similarity_mesure(self, img1, img2, inference_time_visible=False):
        if inference_time_visible:
            start_time = time.time()
            # histogram comparison
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            end_time = time.time()
            print(f"Similarity Inference time: {end_time - start_time}")
        else:
            # histogram comparison
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity
    
    def crop_face(self, img,save_path,  inference_time_visible=False):
            if inference_time_visible:
                start_time = time.time()
                face = self.mtcnn(img, save_path)
                end_time = time.time()
                print(f"Inference time: {end_time - start_time}")
            else:
                face = self.mtcnn(img, save_path)
            return face
    def get_face_bbox(self, img): 
            boxes, _ = self.mtcnn.detect(img)
            return boxes
           
    def crop_face_from_batch(self, batch):
        # Detecting face in the image
        faces = self.mtcnn(batch)
        return faces

    def process_video(self, video_path, show_video=False, skip_similar_frames=True, stride=1, inference_time_visible=False, simily_threshold=0.9995, save_face=True):
        self.save_dir = os.path.join(video_path[:-4])
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        cap = cv2.VideoCapture(video_path)        
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        self.total_number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total_number_frames:",self.total_number_frames)
        
        _, previous_frame = cap.read()
        frame_count = 0
        start_total_time = time.time()
        start_time = time.time()
        # Read until video is completed
        image_list = list()
        while(cap.isOpened()):
            # Capture frame-by-frame
            tmp = time.time()
            ret = cap.grab()
            
            frame_count+=1
            if ret == True:
                if frame_count%stride!=0: continue
                # Display the resulting frame
                #tmp = time.time()
                status, frame = cap.retrieve()  # Decode processing frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                total_grab_time = (time.time() - tmp)
                if inference_time_visible:
                    print("decode_time:",total_grab_time)
                face_path = os.path.join(self.save_dir, f"face_{frame_count}.jpg")
                start_time = time.time()
                faces = self.crop_face(img=frame, save_path=face_path, inference_time_visible=inference_time_visible)
                inference_time = time.time() - start_time  # Compute the inference time
                #print(f"Inference time: {inference_time} seconds")
                image_list.append(frame)
                if len(image_list)==16:
                    #start_time = time.time()
                    faces = self.crop_face_from_batch(image_list)
                    #inference_time = time.time() - start_time
                    #print(f"Inference time: {inference_time} seconds")
                    image_list = list()
                if skip_similar_frames:
                    #start_time = time.time()
                    similarity_score = self.similarity_mesure(previous_frame, frame, inference_time_visible=inference_time_visible)
                    #print("similarity:",similarity_score)
                    if similarity_score>simily_threshold: continue
                previous_frame = frame
                if show_video:
                    if face is not None:
                        boxes = self.get_face_bbox(img=frame)
                        # Draw bounding boxes on the frame
                        for box in boxes:
                            width = box[2] - box[0]
                            height = box[3] - box[1]
                            if width <= 120 or height <= 130:
                                continue
                            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                            # Save detected face into a folder
            else:
                
                break
                   
                
                  

 
#creater = CropFace()
#video_dir = "/home/dev/face_retrieval/vtc_video_tinmoi/Con_trai_nan_nhan_vu_tai_nan_o_Lang_Son:__Thay_nguoi_than_minh_nam_the_rat_bang_hoang_va_bat_ngo_.mp4"
#creater.process_video(video_dir, show_video=False, skip_similar_frames=False, stride=5, inference_time_visible=False, simily_threshold=0.9995, save_face=True)
