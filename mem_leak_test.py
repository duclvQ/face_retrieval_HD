from facenet_pytorch import MTCNN, InceptionResnetV1
import threading
import numpy as np
for _ in range(100):
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=182, margin=10, min_face_size=50, thresholds=[0.8, 0.9, 0.9],keep_all=True, factor=0.709, post_process=False,device='cuda:0')  # initializing mtcnn for face detection
    def crop():       
        # Create an inception resnet (in eval mode):
        #resnet = InceptionResnetV1(pretrained='vggface2').eval()
        from PIL import Image
        img_list = []
<<<<<<< HEAD
        for i in range(10000):
            img = Image.open('./my_source/test2.jpg')
=======
        img = Image.open('./my_source/test2.jpg')
        for i in range(10000):
            
>>>>>>> ff8c2f3526c1c71dc69c724f8e03b94fa4d23c9b
            img_list.append(img)
            if len(img_list)==16:
                # Get cropped and prewhitened image tensor
                img_cropped = mtcnn(img_list)
                img_list.clear()
                print(i)
    t1 = threading.Thread(target=crop, args=())
    t1.start()
    t1.join()
