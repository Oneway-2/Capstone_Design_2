#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tkinter  # 인터페이스를 위한 import

import os
import shutil
import cv2          # openCV
import numpy as np  # 이미지 배열화
from random import randint      # 추천 클래스 출력

import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils  # one hot encoding을 위해

from PIL import ImageTk, Image # tkinter 에서 jpg 파일을 열기 위해

import matplotlib.pyplot as plt # 원그래프 만들기 위해
from matplotlib.figure import Figure # 원그래프를 tkinter 에 적용시키기 위해
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 원그래프를 tkinter 에 적용시키기 위해

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 12})


# In[2]:


mnist_model = keras.models.load_model('./mnist-42-0.0000.hdf5')  
quickdraw_model = keras.models.load_model('./quickdraw.h5')  


# In[3]:


mnist_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
closest_prediction = "temp_string"

quickdraw_classes = [
    [0, 'airplane', '비행기'],
    [1, 'alarm_clock', '알람시계'],
    [2, 'anvil', '모루'],
    [3, 'apple', '사과'],
    [4, 'axe', '도끼'],
    [5, 'baseball', '야구공'],
    [6, 'baseball_bat', '야구배트'],
    [7, 'basketball', '농구공'],
    [8, 'beard', '콧수염'],
    [9, 'bed', '침대'],
    [10, 'bench', '벤치'],
    [11, 'bycicle', '자전거'],
    [12, 'bird', '새'],
    [13, 'book', '책'],
    [14, 'beard', '턱수염'],
    [15, 'bridge', '구름다리'],
    [16, 'broom', '빗자루'],
    [17, 'butterfly', '나비'],
    [18, 'camera', '카메라'],
    [19, 'candle', '양초'],
    [20, 'car', '자동차'],
    [21, 'cat', '고양이'],
    [22, 'ceiling_fan', '천장선풍기'],
    [23, 'cell_phone', '휴대전화'],
    [24, 'chair', '의자'],
    [25, 'circle', '원'],
    [26, 'clock', '시계'],
    [27, 'cloud', '구름'],
    [28, 'coffee_cup', '커피컵'],
    [29, 'cookie', '쿠키'],
    [30, 'cup', '컵'],
    [31, 'diving_board', '다이빙보드'],
    [32, 'donut', '도넛'],
    [33, 'door', '문'],
    [34, 'drums', '드럼'],
    [35, 'dumbbell', '덤벨'],
    [36, 'envelope', '봉지'],
    [37, 'eye', '눈'],
    [38, 'eyeglasses', '안경'],
    [39, 'face', '얼굴'],
    [40, 'fan', '선풍기'],
    [41, 'flower',' 꽃'],
    [42, 'frying_pan', '프라이팬'],
    [43, 'grapes', '포도'],
    [44, 'hammer', '망치'],
    [45, 'hat', '모자'],
    [46, 'headphones', '헤드폰'],
    [47, 'helmet', '헬멧'],
    [48, 'hot_dog', '핫도그'],
    [49, 'ice_cream', '아이스크림'],
    [50, 'key', '열쇠'],
    [51, 'knife', '칼'],
    [52, 'ladder', '사다리'],
    [53, 'laptop', '노트북'],
    [54, 'lighting', '번개'],
    [55, 'light_bulb', '전구'],
    [56, 'line', '선'],
    [57, 'lolipop', '막대사탕'],
    [58, 'microphone', '마이크'],
    [59, 'moon', '달'],
    [60, 'mountain', '산'],
    [61, 'moustache', '콧수염'],
    [62, 'mushroom', '버섯'],
    [63, 'pants', '바지'],
    [64, 'paper_clip', '클립'],
    [65, 'pencil', '연필'],
    [66, 'pillow', '배게'],
    [67, 'pizza', '피자'],
    [68, 'power_outlet', '콘센트'],
    [69, 'radio', '라디오'],
    [70, 'rainbow', '무지개'],
    [71, 'rifle', '소총'],
    [72, 'saw', '톱'],
    [73, 'scissors', '가위'],
    [74, 'screwdriver', '드라이버'],
    [75, 'shorts', '반바지'],
    [76, 'shovel', '삽'],
    [77, 'smiley_face', '웃는얼굴'],
    [78, 'snake', '뱀'],
    [79, 'sock', '양말'],
    [80, 'spider', '거미'],
    [81, 'spoon', '숟가락'],
    [82, 'square', '정사각형'],
    [83, 'star', '별'],
    [84, 'stop_sign', '정지신호'],
    [85, 'suitcase', '여행가방'],
    [86, 'sun', '태양'],
    [87, 'sword', '검'],
    [88, 'syringe', '주사기'],
    [89, 't-shirt', '티셔츠'],
    [90, 'table', '책상'],
    [91, 'tennis_racquet', '테니스라켓'],
    [92, 'tent', '텐트'],
    [93, 'tooth', '치아'],
    [94, 'traffic_light', '신호등'],
    [95, 'tree', '나무'],
    [96, 'triangle', '삼각형'],
    [97, 'umbrella', '우산'],
    [98, 'wheel', '바퀴'],
    [99, 'wristwatch', '손목시계'] 
]


# In[4]:


########################################## 그림판 열리는 함수 ##########################################
drawing = False

def drawing_start():
    
    def draw_circle(event, x, y, flags, param):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(img,(x,y),thickness,(255,255,255),-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    img = np.zeros((300,300,1), np.uint8)
    global thickness

    cv2.namedWindow('Canvas')
    cv2.setMouseCallback('Canvas' ,draw_circle)

    while(1):
        cv2.imshow('Canvas' ,img)
        type_key = cv2.waitKey(1)
        if type_key == 27: # esc 를 누르면 저장없이 퇴장
            break
        elif type_key == ord('r'):  # r 을 누르면 다시그리기
            img = np.zeros((300,300,1), np.uint8)
        elif type_key == ord('s'):  # s 를 누르면 저장하고 퇴장.
            resize_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            cv2.imwrite('./drawing_canvas.jpg', resize_img)
            break
        elif type_key == 91: # 얇게
            thickness -=1  
            if (thickness <= 0):
                thickness = 1
            print("thickness is " + str(thickness))
        elif type_key == 93: # 굵게
            thickness +=1     
            if (thickness > 16):
                thickness = 15
            print("thickness is " + str(thickness))
            
    change_image()     
    cv2.destroyAllWindows()
    
########################################## 그림판 열리는 함수 ##########################################


# In[5]:


########################################## 그린 그림 배열로 저장하는 과정 ##########################################

def predicting_start(X_madeVal):
    
    def readimg(path):
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #img = plt.imread(path, cv2.IMREAD_GRAYSCALE)
        #img=np.reshape(img, [-1, 784]) # 이거 하면 1차원으로 늘려주는건데, 일단 안썼다. mnist.data_load 예제랑 똑같이 하려고.
        return img

    X_madeVal.append(readimg('./drawing_canvas.jpg')) # X_madeVal에 이미지 저장    
    X_madeVal = np.array(X_madeVal)                         

    X_pitMadeVal = X_madeVal
    X_madeVal = X_madeVal.reshape(X_madeVal.shape[0], 28, 28, 1).astype('float32') / 255
    
    #print(type(X_madeVal))
    
    return X_madeVal

########################################## 그린 그림 배열로 저장하는 과정 ##########################################


# In[6]:


########################################## 원그래프 그리는 함수 ##########################################

def showing_pie_graph(sorted_predictions, predictions, is_it_mnist):
    
    one_to_five = [int(sorted_predictions[0][0]) , int(sorted_predictions[0][1]) , 
                   int(sorted_predictions[0][2]) , int(sorted_predictions[0][3]) , 
                   int(sorted_predictions[0][4])]     # 가장 정확한 숫자가 들은 인덱스를 차례대로 저장.
    
    ratio = [predictions[0][one_to_five[0]] , predictions[0][one_to_five[1]] , 
             predictions[0][one_to_five[2]] , predictions[0][one_to_five[3]] , 
             predictions[0][one_to_five[4]]]         # prediction 한 실제 값을 높은 순위부터 저장.
    
    if(is_it_mnist == True):    
        labels = [sorted_predictions[0][0] , sorted_predictions[0][1] , 
                  sorted_predictions[0][2] , '', '']             # mnist 같은경우엔 0 ~ 9 숫자가
    
    else:    
        labels = [quickdraw_classes[sorted_predictions[0][0]][1] , quickdraw_classes[sorted_predictions[0][1]][1] , 
                  quickdraw_classes[sorted_predictions[0][2]][1], '', '']    # quickdraw 라면 클래스 이름이 출력, 4위부턴 출력안한다.
    
    colors = ['gold', 'tomato', 'c', 'greenyellow', 'lightpink']
    explode = (0.1, 0.0, 0.0, 0.0, 0.0)
        
    figure = Figure(figsize=(6,5), dpi=70) 
    subplot = figure.add_subplot(111) 
    
    if(is_it_mnist == True):
        subplot.pie(ratio, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=180,
                        counterclock=False)
    else:    
        pie1 = subplot.pie(ratio, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=180,
                       labeldistance=1.35, rotatelabels =True, counterclock=False)   
        plt.setp(pie1[1], rotation_mode="anchor", ha="center", va="center")
    
    #subplot.set_title('What did you draw?')
    subplot.axis('equal')  
    pie = FigureCanvasTkAgg(figure, window) 
    pie.get_tk_widget().place(x=310, y=180, width=310, height=200)   
    
    recommendation_print() # 예측 한번 하고나면 새로운 추천단어 띄워주기
    
########################################## 원그래프 그리는 함수 ##########################################


# In[16]:


########################################## 숫자 (MNIST) 가중치 적용 ##########################################

def MNIST_apply():
    global closest_prediction
    is_it_mnist = True
    X_madeVal = []
    X_madeVal = predicting_start(X_madeVal)
        
    mnist_prediction = mnist_model.predict(X_madeVal)  # 직접 그린 그림으로 예측 해보려고 한다!!!

    sorted_mnist_predictions = mnist_prediction.argsort()  # 제일 틀린녀석부터 나열한다. 
    sorted_mnist_predictions = np.flip(sorted_mnist_predictions)  # 거꾸로 뒤집음으로써 제일 맞는녀석부터 나열한다.
    
    #for num, i in zip(range(1,6), sorted_mnist_predictions[0][range(0, 5)]):  # 1위부터 5위까지 출력한다.
    #    print(str(num) + '위 = ' + str(i))
        
    first_label.config(text="가장 닮은 숫자는 " + str(sorted_mnist_predictions[0][0]) + "이군요! (" + str(round(mnist_prediction[0][sorted_mnist_predictions[0][0]] * 100.0, 4)) + "%)")
    second_label.config(text="2위는 " + str(sorted_mnist_predictions[0][1]) + " (" + str(round(mnist_prediction[0][sorted_mnist_predictions[0][1]] * 100.0, 4)) + "%)")
    third_label.config(text="3위는 " + str(sorted_mnist_predictions[0][2])  + " (" + str(round(mnist_prediction[0][sorted_mnist_predictions[0][2]] * 100.0, 4)) + "%)")
    fourth_label.config(text="4위는 " + str(sorted_mnist_predictions[0][3])  + " (" + str(round(mnist_prediction[0][sorted_mnist_predictions[0][3]] * 100.0, 4)) + "%)")
    fifth_label.config(text="5위는 " + str(sorted_mnist_predictions[0][4])  + " (" + str(round(mnist_prediction[0][sorted_mnist_predictions[0][4]] * 100.0, 4)) + "%)")
        
    closest_prediction = str(sorted_mnist_predictions[0][0])
    save_it.config(text = '"' + closest_prediction + '"' + "\nClick to Save")      
    
    showing_pie_graph(sorted_mnist_predictions, mnist_prediction, is_it_mnist)
    save_it.config(state="normal")
    
########################################## 숫자 (MNIST) 가중치 적용 ##########################################


# In[17]:


########################################## 그림 (Quickdraw) 가중치 적용 ##########################################

def Quickdraw_apply():
    global closest_prediction
    is_it_mnist = False
    X_madeVal = []
    X_madeVal = predicting_start(X_madeVal)    
    
    quickdraw_prediction = quickdraw_model.predict(X_madeVal)  # 직접 그린 그림으로 예측 해보려고 한다!!!
    
    sorted_quickdraw_predictions = quickdraw_prediction.argsort()  # 제일 틀린녀석부터 나열한다. 
    sorted_quickdraw_predictions = np.flip(sorted_quickdraw_predictions)  # 거꾸로 뒤집음으로써 제일 맞는녀석부터 나열한다.
    #print(sorted_quickdraw_predictions)  
    
    #for num, i in zip(range(1,6), sorted_quickdraw_predictions[0][range(0, 5)]):  # 1위부터 5위까지 출력한다.
    #    print(str(num) + '위 = ' + quickdraw_classes[i][1] + ' ' + quickdraw_classes[i][2])
        
    first_label.config(text="가장 닮은 그림은 " + str(quickdraw_classes[sorted_quickdraw_predictions[0][0]][2]) + "이군요! (" + str(round(quickdraw_prediction[0][sorted_quickdraw_predictions[0][0]] * 100.0, 4)) + "%)")
    second_label.config(text="2위는 " + str(quickdraw_classes[sorted_quickdraw_predictions[0][1]][2]) + " (" + str(round(quickdraw_prediction[0][sorted_quickdraw_predictions[0][1]] * 100.0, 4)) + "%)")
    third_label.config(text="3위는 " + str(quickdraw_classes[sorted_quickdraw_predictions[0][2]][2])  + " (" + str(round(quickdraw_prediction[0][sorted_quickdraw_predictions[0][2]] * 100.0, 4)) + "%)")
    fourth_label.config(text="4위는 " + str(quickdraw_classes[sorted_quickdraw_predictions[0][3]][2])  + " (" + str(round(quickdraw_prediction[0][sorted_quickdraw_predictions[0][3]] * 100.0, 4)) + "%)")
    fifth_label.config(text="5위는 " + str(quickdraw_classes[sorted_quickdraw_predictions[0][4]][2])  + " (" + str(round(quickdraw_prediction[0][sorted_quickdraw_predictions[0][4]] * 100.0, 4)) + "%)")
        
    closest_prediction = str(quickdraw_classes[sorted_quickdraw_predictions[0][0]][1])   
    save_it.config(text = '"' + closest_prediction + '"' + "\nClick to Save")    

    showing_pie_graph(sorted_quickdraw_predictions, quickdraw_prediction, is_it_mnist)
    save_it.config(state="normal")

########################################## 그림 (Quickdraw) 가중치 적용 ##########################################


# In[52]:


def Save_Drawing():
        save_it.config(state="disabled", text = closest_prediction + ".jpg" + "\nSaved")
        number = 0

        first_path = './Drawing_Data'
        
        if not os.path.exists(first_path):
            os.makedirs(first_path)
            
        second_path = './Drawing_Data/' + closest_prediction
            
        if not os.path.exists(second_path):
            os.makedirs(second_path)        
        
        while(os.path.isfile(second_path + '/' + closest_prediction + '_' + str(number) + '.jpg')):
            # 해당하는 번호의 파일이 없을때까지 숫자를 증가시킨다.
            number = number + 1

        to_copy_file_dir = second_path + '/' + closest_prediction + '_' + str(number) + '.jpg'
        shutil.copyfile('./drawing_canvas.jpg', to_copy_file_dir)    # drawing_canvas 파일을 해당 폴더에 복사시킨다.
            
        


# In[53]:


########################################## 인터페이스 실행 ##########################################

thickness = 10

window=tkinter.Tk()

window.title("Image Prediction")
window.geometry("640x400+100+100")
window.resizable(False, False)


########################### 라벨들 ###########################
label=tkinter.Label(text="<< Tensorflow를 이용한 Image Prediction >>")
label.pack()

thickness_label=tkinter.Label(text="선 굵기 = " + str(thickness))
thickness_label.place(x=300, y=40, width=100, height=20)

save_it=tkinter.Label(text="예측이 정확하면 클릭")
save_it.place(x=295, y=100, width=115, height=20)
########################### 라벨들 ###########################



########################### 그린 이미지 나타내기 ###########################
path = "./drawing_canvas.jpg"
img = Image.open(path)
img = img.resize((112,112))
img = ImageTk.PhotoImage(img)
panel = tkinter.Label(window, image = img)
panel.place(x=415, y=44, width=112, height=112) # 처음 저장되어있는 사진 출력..

def change_image():    
    img = Image.open(path)
    img = img.resize((112,112))
    img = ImageTk.PhotoImage(img)    
    panel.configure(image=img)
    panel.image = img
########################### 그린 이미지 나타내기 ###########################



########################### 버튼들 ###########################
b1=tkinter.Button(window, text="그림 그리기\n\n's' 저장 'r' 다시그리기 'esc' 퇴장\n '[' 선 얇게 ']' 선 굵게", command=drawing_start)
b1.place(x=20, y=20, width=270, height=170) # (20,20 270,20, 270,170, 20,170)

b5=tkinter.Button(window, text="숫자 예측", command=MNIST_apply)
b5.place(x=20, y=195, width=130, height=35)

b6=tkinter.Button(window, text="그림 예측", command=Quickdraw_apply)
b6.place(x=160, y=195, width=130, height=35)

save_it=tkinter.Button(window, text="눌러서 저장", command=Save_Drawing, state="disabled")
save_it.place(x=296, y=125, width=112, height=40)
########################### 버튼들 ###########################



########################### 순위 출력 Labels ###########################
whole_label=tkinter.Label(relief='groove', bd=2)
whole_label.place(x=20, y=235, width=270, height=150)

first_label=tkinter.Label(text="", fg="blue")
first_label.place(x=25, y=250, width=260, height=20)

second_label=tkinter.Label(text="")
second_label.place(x=25, y=275, width=260, height=20)

third_label=tkinter.Label(text="")
third_label.place(x=25, y=300, width=260, height=20)

fourth_label=tkinter.Label(text="")
fourth_label.place(x=25, y=325, width=260, height=20)

fifth_label=tkinter.Label(text="")
fifth_label.place(x=25, y=350, width=260, height=20)
########################### 순위 출력 Labels ###########################



########################### 추천 출력 Labels ###########################
recommendation_label=tkinter.Label(text="이걸 그려봐요")
recommendation_label.place(x=530, y=35, width=100, height=20)

def recommendation_print():   
    i = randint(0, 99)
    recommendation1_label=tkinter.Label(text="< " + str(quickdraw_classes[i][2]) + " >")
    recommendation1_label.place(x=530, y=60, width=100, height=20)

    i = randint(0, 99)
    recommendation2_label=tkinter.Label(text="< " + str(quickdraw_classes[i][2]) + " >")
    recommendation2_label.place(x=530, y=85, width=100, height=20)

    i = randint(0, 99)
    recommendation3_label=tkinter.Label(text="< " + str(quickdraw_classes[i][2]) + " >")
    recommendation3_label.place(x=530, y=110, width=100, height=20)

    i = randint(0, 99)
    recommendation4_label=tkinter.Label(text="< " + str(quickdraw_classes[i][2]) + " >")
    recommendation4_label.place(x=530, y=135, width=100, height=20)
    
recommendation_print()
########################### 추천 출력 Labels ###########################



########################### SpinBox ###########################
def value_check(self):
    global thickness
    valid = False
    if self.isdigit():
        if (int(self) < 16 and int(self) > 0):
            thickness = int(self)
            thickness_label.config(text="선 굵기 = " + str(thickness))
            valid = True
    elif self == '':
        valid = True
    return valid

def value_error(self):
    thickness_label.config(text="1~15 사이만 입력")        
    
def spinBox_command(self):
    thickness = int(self)
    
validate_command=(window.register(value_check), '%P')
invalid_command=(window.register(value_error), '%P')
    
var = tkinter.DoubleVar(value=12) # 초기 선 굵기 12로 맞추기
spinbox=tkinter.Spinbox(window, from_ = 1, to = 15, textvariable = var, validate = 'all', command = spinBox_command, validatecommand = validate_command, invalidcommand=invalid_command)
spinbox.place(x=305, y=65, width=85, height=20)
########################### SpinBox ###########################


window.mainloop()
########################################## 인터페이스 실행 ##########################################


# In[ ]:




