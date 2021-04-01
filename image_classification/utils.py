import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pyautogui
from PIL import Image


plural_to_sg = {
    "bicycles":"bicycle",
    "bridges":"bridge",
    "buses":"bus",
    "cars":"car",
    "chimneys":"chimney",
    "crosswalks":"crosswalk",
    "fire hydrants":"a fire hydrant",
    "motorcycles":"motorcycle",
    "mountains or hills":"mountain or hill",
    "palm trees":"palm tree",
    "stairs":"stair",
    "traffic lights":"traffic light"
}

def process(description):
    """
    process captcha description :
    - singularize class name
    - ensures cars/boats/motorcycles/bicycles/buses are recognized as vehicles
    - remove end point in the last setence
    
    Arguments:
    description -- string, captcha description
    Returns:
    valid_labels -- list of correct output labels for this captcha
    challenge_type -- last sentence of the description (indicates click/select captcha)
    """
    
    _ , challenge_label, challenge_type = description.split('\n')
    challenge_type = challenge_type.rstrip('.') # if challenge type ends with a point, remove it
    
    print(challenge_label)
    if challenge_label in plural_to_sg:
        challenge_label = plural_to_sg[challenge_label]
    
    if challenge_label == 'vehicle':
        valid_labels = ['car', 'boat', 'motorcycle', 'bicycle', 'bus']
    else:
        valid_labels = []
        valid_labels.append(challenge_label)
        
    return (valid_labels,challenge_type)
    
# coordinates of each image if the grid is 3x3
coordinates_3x3 = [
    [(125,325),(250,325),(390,325)],
    [(125,455),(250,455),(390,455)],
    [(125,585), (250,585), (390,585)]
]

# coordinates of each image if the grid is 4x4
coordinates_4x4 = [
    [(110,306),(214,306),(305,306),(400,306)],
    [(110,403),(214,403),(305,403),(400,403)],
    [(110,503),(214,503),(305,503),(400,503)],
    [(110,600),(214,600),(305,600),(400,600)]
]

coordinates_verify = (404,686)

categories = {
    "0":"bicycle",
    "1":"bridge",
    "2":"bus",
    "3":"car",
    "4":"chimney",
    "5":"crosswalk",
    "6":"a fire hydrant",
    "7":"motorcycle",
    "8":"mountain or hill",
    "9":"other",
    "10":"palm tree",
    "11":"stair",
    "12":"traffic light",
}

def pred(model, labels, grid_size, folder_path = './', threshold = 1):
    """
    outputs prediction 0/1 for each image in the directory
    
    Arguments:
    labels -- list, labels of interest for the current captcha (e.g. ['traffic lights'] or ['fire hydrants'])
    grid_size -- number of images in a single line/column of the current captcha (3 or 4)
    folder_path -- path to the folder containing the images
    threshold -- degree of flexibility (e.g. threshold = 3, output is 1 if one of the top 3 predictions is of the right label)
    Returns:
    predictions -- numpy array of shape (grid_size, grid_size)
    """


    files_list = os.listdir(folder_path)
    files_list.sort()
    #print("files in current folder :",files_list)

    predictions = np.zeros((grid_size,grid_size))
    # Load every image in the directory
    for filename in files_list:
        #print(folder_path + filename)
        data = np.asarray(Image.open(folder_path + filename)) # loading images as numpy array
        data = data[:,:,:3] # removing alpha channel (if any)

        plt.imshow(data, interpolation='nearest')
        plt.show()

        #data = data / 255.0
        data = np.expand_dims(data, axis=0)

        prediction = model.predict(data)
        print("prediction :", prediction)
        print("prediction.shape :", prediction.shape)
        prediction = prediction[0].argmax()
        print("argsort :", prediction)

        # prediction = 1 if one of the top predictions is of the right label
        
        print(f"{categories[str(prediction)]}", end=' ')
        if categories[str(prediction)] in labels :

            # Retrieve position in the grid with filename
            x_grid = filename[9]
            y_grid = filename[12]
            predictions[int(x_grid)-1, int(y_grid) - 1] = 1
            os.remove(filename)
        print()
        
    return predictions

def submit_images(predictions):
    """
    click on all images predicted by the model
    
    Arguments:
    predictions -- numpy array, predictions[k] = 1 if the image k should be clicked
    Returns:
    img_clicked -- list of (row,column) of the images img_clicked
    """
    
    clicked = []
    coordinates = coordinates_3x3 if len(predictions) == 3 else coordinates_4x4
    
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i,j] == 1:
                x, y = coordinates[i][j]
                pyautogui.moveTo(x,y,0.5, pyautogui.easeOutQuad)
                pyautogui.click()
                clicked.append((i + 1,j + 1))
            
    return clicked

def validate():
    x, y = coordinates_verify
    pyautogui.moveTo(x,y,0.5, pyautogui.easeOutQuad)
    pyautogui.click()


def select_all_matching(model, labels, grid_size, folder_path = './', threshold = 1):
    """
    Click on the most likely image among the first set of images
    
    Arguments:
    labels -- list, labels of interest for the current captcha (e.g. ['traffic lights'] or ['fire hydrants'])
    grid_size -- number of images in a single line/column of the current captcha (3 or 4)
    folder_path -- path to the folder containing the images
    threshold -- degree of flexibility (e.g. threshold = 3, output is 1 if one of the top 3 predictions is of the right label)
    Returns:
    predictions -- numpy array of shape (grid_size, grid_size)
    """


    files_list = os.listdir(folder_path)
    files_list.sort()
    #print("files in current folder :",files_list)

    predictions = np.zeros((grid_size,grid_size))
    # Load every image in the directory
    for filename in files_list:
        #print(folder_path + filename)
        data = np.asarray(Image.open(folder_path + filename)) # loading images as numpy array
        data = data[:,:,:3] # removing alpha channel (if any)

        plt.imshow(data, interpolation='nearest')
        plt.show()

        #data = data / 255.0
        data = np.expand_dims(data, axis=0)

        prediction = model.predict(data)

        prediction = prediction[0].argsort()[-threshold:][::-1]
        #print(prediction)

        # prediction = 1 if one of the top predictions is of the right label
        for pred in prediction:
            print(f"{categories[str(pred)]}", end=' ')
            if categories[str(pred)] in labels :
                
                # Retrieve position in the grid with filename
                x_grid = filename[9]
                y_grid = filename[12]
                predictions[int(x_grid)-1, int(y_grid) - 1] = 1
                os.remove(filename)
        print()
        
    return predictions

def delay(i = 2):
    time.sleep(random.randint(i,i+1))
    

def clear_directory(dir_path = './'):
    
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
