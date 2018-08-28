from django.shortcuts import render
from index.forms import UploadFileForm
from django.http import HttpResponseRedirect
from index.models import UploadFileModel
from keras.models import Sequential,load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from django.conf import settings
from keras import backend as K
import h5py
from PIL import Image
import numpy as np
import os,glob
import cv2
import pandas as pd

def index(request):
    K.clear_session()
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            #return HttpResponseRedirect('/upload')
            return upload(request)
    else:
        form = UploadFileForm()
    return render(request, 'index/index.html', {'form': form})

def upload(request):
    length = len(UploadFileModel.objects.all())
    img_info = UploadFileModel.objects.all()[length - 1]
    img_i = img_info.file
    result = img(img_i)
    return render(request, 'index/upload.html', {'result': result})

def img(img_path):
    model = Sequential()
    model = load_model('model/0821.hdf5')
    pathi = 'model/image/newimg/input/' + str(img_path)
    #pathi = 'model/image/newimg/input/333.jpg'
    input_image_dir = pathi
    input_files = glob.glob(input_image_dir)
    input_X=[]

    image_w = 64
    image_h = 64
    for i, f in enumerate(input_files):
        img = Image.open(f) # --- (※6)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        input_X.append(data)

    input_X = np.array(input_X)
    #Y = np.array(Y)
    input_X = input_X.astype("float") / 256
    input_pre = model.predict(input_X)
    #print("인풋 카테고리 정확도",input_pre)

    input_cate=0
    for i,v in enumerate(input_pre):
        input_cate = v.argmax()
        #print(v.argmax())

    #output 카테고리 분류
    output_image_dir = 'model/image/newimg/'

    output_cate=[]
    output_cate_dir=[]
    pp = str(img_path)
    i = pp.split('.')[0]
    name = 'model/recomment' + str(i) + '.csv'
    dd = pd.read_csv(name, index_col = 0)
    k = list(dd['0'])
    for Path in k:
        output_files = glob.glob(output_image_dir+Path)
        output_X=[]

        image_w = 64
        image_h = 64
        for i, f in enumerate(output_files):
            img = Image.open(f) # --- (※6)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            output_X.append(data)

        output_X = np.array(output_X)
        #Y = np.array(Y)
        output_X = output_X.astype("float") / 256
        output_pre = model.predict(output_X)
        #print(output_pre)


        for i,v in enumerate(output_pre):
            temp=[]
            temp=v.argmax(),v.max()
            output_cate.append(temp)
            output_cate_dir.append(output_image_dir+Path)
            #print(v.argmax())

    #분류된 이미지와 인풋 이미지의 카테고리를 서로 비교

    rank=[]
    for i in range(len(output_cate)):
        if input_cate==output_cate[i][0]:
            rank.append(i)

    input_img_color = cv2.imread(pathi, cv2.IMREAD_COLOR )

    #print(input_img_color.shape)
    #print(input_img_color)
    input_height, input_width, input_channel = input_img_color.shape
    #print (height, width, channel)

    input_b_sum=0
    input_g_sum=0
    input_r_sum=0
    input_pixel_count=1
    for y in range(0, input_height):
        for x in range(0, input_width):
            b = input_img_color.item(y,x,0)
            g = input_img_color.item(y,x,1)
            r = input_img_color.item(y,x,2)
            input_pixel_count+=1
            input_b_sum+=b
            input_g_sum+=g
            input_r_sum+=r

    input_b_avg=int(input_b_sum/input_pixel_count)
    input_g_avg=int(input_g_sum/input_pixel_count)
    input_r_avg=int(input_r_sum/input_pixel_count)
    #print(input_b_avg,input_g_avg,input_r_avg)

    color_rank=[]
    #output_cate_dir
    for get in rank:
        #print(output_cate_dir[get])
        a = output_cate_dir[get]
        #print(Path)
        output_img_color=cv2.imread(a, cv2.IMREAD_COLOR )
        output_height, output_width, output_channel = output_img_color.shape
        #print (height, width, channel)
        output_b_sum=0
        output_g_sum=0
        output_r_sum=0
        output_pixel_count=1
        for y in range(0, output_height):
            for x in range(0, output_width):
                b = output_img_color.item(y,x,0)
                g = output_img_color.item(y,x,1)
                r = output_img_color.item(y,x,2)
                #print("bgr",b,g,r)
                output_pixel_count+=1
                output_b_sum+=b
                output_g_sum+=g
                output_r_sum+=r

        output_b_avg=int(output_b_sum/output_pixel_count)
        output_g_avg=int(output_g_sum/output_pixel_count)
        output_r_avg=int(output_r_sum/output_pixel_count)
        #print("평균",output_b_avg,output_g_avg,output_r_avg)
        vec_distance= ((input_b_avg-output_b_avg)**2 + (input_g_avg-output_g_avg)**2 + (input_r_avg-output_r_avg)**2)**0.5
        info= vec_distance, a
        color_rank.append(info)
        color_rank.sort()

    rank_num=1
    result = []
    for match, path in color_rank:
        p = path.split('/')
        pa = 'img/' + p[3] + '/' + p[4]
        r = {'index' : rank_num, 'path' : pa}
        result.append(r)
        rank_num+=1
    return result
