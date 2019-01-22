
import face_recognition
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-dd", "--dataset_dir",
    help="path to dataset ")
ap.add_argument("-o", "--encodings_output", default='encodings',
    help="path to output encodings")
ap.add_argument("-t", "--test_dir", default='examples',
    help="path to test directory")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

def get_encodings(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width= 400)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("[INFO] recognizing faces..."+ image_path)
    boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    return encodings

def load_images(image_paths):
    images=[]
    for f in os.listdir(image_paths):
        if not f.startswith('.'):
            images.append(f)
    return images

def save_encodings(encodings, filename, encodings_path):
    print('[INFO] saving endodings')    
    for (i, encoding) in enumerate(encodings):
        if i==0:
            filepath=encodings_path+"/"+os.path.splitext(filename)[0]+'.npy'
        else:
            filepath=encodings_path+os.path.splitext(filename)[0]+'_{}'.format(i)+'.npy'
        np.save(filepath, encoding)

def load_encodings_from_directory(encodings_path):
    print("[INFO] loading encodings")
    encodings=[]
    encoding_names=[]
    for f in os.listdir(encodings_path):
        if not (f.startswith('.')):
            encoding=np.load(encodings_path+ '/' +f)
            encodings.append(encoding)
            encoding_names.append(f)
    return encodings, encoding_names

def create_labels(dataset_dir):
    labels=[]
    for f in os.listdir(dataset_dir):
        if not f.startswith('.'):
            labels.append(f)
    return labels

def make_label_dirs(labels, encodings_output):
    for label in labels:
        path =encodings_output+"/"+label
        if not os.path.exists(path):
            os.mkdir(path)

def create_encodings(dataset_dir , encodings_output):
    labels= create_labels(dataset_dir)
    make_label_dirs(labels, encodings_output)
    for label in labels:
        path=dataset_dir+"/"+label
        encodings_path= encodings_output+"/"+label
        images= load_images(path)
        for image in images:
            print(image)
            encodings= get_encodings(path+"/"+image)
            
            save_encodings(encodings, image, encodings_path)

def create_model_using_encodings(encodings_dir):
    labels= create_labels(encodings_dir)
    train_data=[]
    train_labels=[]
    for label in labels:
        path= encodings_dir+"/"+label
        encodings, encoding_names= load_encodings_from_directory(path)
        for encoding in encodings:
            train_data.append(encoding)
            train_labels.append(label)
    model= LinearSVC()
    model.fit(train_data, train_labels)
    model_file='model_file.sav'
    pickle.dump(model, open(model_file, 'wb'))

def test_the_model(test_dir):
    print("[INFO] evaluating classifier...")
    filename = 'model_file.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    images= load_images(test_dir)
    for image in images:
        print(image)
        path=test_dir+"/"+image
        encodings= get_encodings(path)
        predictions=loaded_model.predict(encodings)
        print(predictions)


create_encodings(args['dataset_dir'], args['encodings_output'])
create_model_using_encodings(args['encodings_output'])
test_the_model(args['test_dir'])

    

