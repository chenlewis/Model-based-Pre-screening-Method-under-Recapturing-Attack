from scipy import signal
import random
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from scipy.stats import pearsonr
import pandas as pd
from scipy.stats import spearmanr
from PIL import Image
from sklearn import svm
import time

def write_csv(results, file_name):
    import csv
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['id', 'label'])
        writer.writerows(results)




def rgb_to_cmyk(r,g,b):
    cmyk_scale = 100
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / 255.
    m = 1 - g / 255.
    y = 1 - b / 255.

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale

def model_match(dir1, dir2):
    global k_halftone_combination_template, m_halftone_combination_template, c_halftone_combination_template, y_halftone_combination_template
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    files1_name = []
    files2_name = []
    similarity = []

    #   尺寸 分辨率 打印机型号

    # 224 1200dpi 176n
    # c_halftone = [(77, 89), (85, 118), (92, 148), (104, 81), (118, 141), (130, 74), (137, 104), (145, 133)]
    # m_halftone = [(92, 74), (85, 104), (77, 133), (119, 81), (103, 141), (145, 89), (137, 118), (130, 148)]
    # k_halftone = [(111, 67), (92, 89), (74, 111), (92, 133), (111, 155), (130, 133), (148, 111), (130, 89)]
    # y_halftone = [(111, 56), (83, 83), (55, 111), (83, 139), (111, 166), (139, 139), (167, 111), (139, 83)]
    # 224 600dpi 176n  
    c_halftone = [(44, 67), (59, 126), (74, 185), (96, 52), (126, 170), (148, 37), (163, 96), (178, 155)]
    m_halftone = [(74, 37), (59, 96), (43, 155), (126, 52), (96, 170), (179, 67), (163, 126), (148, 185)]
    k_halftone = [(111, 23), (74, 67), (36, 111), (74, 155), (111, 199), (148, 155), (186, 111), (148, 67)]
    y_halftone = [(111, 0), (55, 55), (0, 111), (54, 166), (111, 221), (167, 167), (221, 111), (168, 56)]

    correlate_c = []
    correlate_m = []
    correlate_k = []
    correlate_y = []
    image_name = []
    # print(len(c_halftone))

    for f1 in files1:
        a1, b1 = os.path.splitext(f1)
        img1 = cv2.imread(os.path.join(dir1 + f1))

        # for i
        # img1_c, img1_m, img1_y, img1_k = rgb_to_cmyk(img1[])
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)#将搜索图片转化为灰度图
        fft2 = np.fft.fft2(img1_gray)
        shift2center = np.fft.fftshift(fft2)
        log_shift2center = np.log(1 + np.abs(shift2center))

        if (a1[0] == 'c'):
            img1_gray_1 = log_shift2center[c_halftone[0][0]:c_halftone[0][0] + 3, c_halftone[0][1]:c_halftone[0][1] + 3]

            c_halftone_combination_template = img1_gray_1
            for i in range(1, len(c_halftone)):
                img1_gray_1 = log_shift2center[c_halftone[i][0]:c_halftone[i][0] + 3, c_halftone[i][1]:c_halftone[i][1] + 3]
                c_halftone_combination_template = np.c_[c_halftone_combination_template, img1_gray_1]
                # plt.subplot(221),plt.imshow(img1[:,:,[2,1,0]])
                # plt.subplot(222), plt.imshow(img1_gray_1)
                # plt.subplot(223), plt.imshow(img1[:,:,[2,1,0]])
                # plt.show()

            c_halftone_combination_template = np.asarray(c_halftone_combination_template).flatten()
            c_halftone_combination_template_mean = np.mean(c_halftone_combination_template)
            c_halftone_combination_template1 = c_halftone_combination_template - c_halftone_combination_template_mean
            # print(c_halftone_combination_template)

        elif (a1[0] == 'm'):
            img1_gray_1 = log_shift2center[m_halftone[0][0]:m_halftone[0][0] + 3, m_halftone[0][1]:m_halftone[0][1] + 3]
            m_halftone_combination_template = img1_gray_1
            for i in range(1, len(m_halftone)):
                img1_gray_1 = log_shift2center[m_halftone[i][0]:m_halftone[i][0] + 3, m_halftone[i][1]:m_halftone[i][1] + 3]
                m_halftone_combination_template = np.c_[m_halftone_combination_template, img1_gray_1]
            m_halftone_combination_template = np.asarray(m_halftone_combination_template).flatten()
            # m_halftone_combination_template_mean = np.mean(m_halftone_combination_template)
            # m_halftone_combination_template1 = m_halftone_combination_template - m_halftone_combination_template_mean
            # plt.plot(m_halftone_combination_template1)
            # plt.xlabel('number')
            # plt.ylabel('x-x_mean')
            # plt.show()
            # print(m_halftone_combination_template)
        elif (a1[0] == 'k'):
            img1_gray_1 = log_shift2center[k_halftone[0][0]:k_halftone[0][0] + 3, k_halftone[0][1]:k_halftone[0][1] + 3]
            k_halftone_combination_template = img1_gray_1
            for i in range(1, len(k_halftone)):
                img1_gray_1 = log_shift2center[k_halftone[i][0]:k_halftone[i][0] + 3, k_halftone[i][1]:k_halftone[i][1] + 3]
                k_halftone_combination_template = np.c_[k_halftone_combination_template, img1_gray_1]
                # plt.subplot(221), plt.imshow(img1[:, :, [2, 1, 0]])
                # plt.subplot(222), plt.imshow(img1_gray_1)
                # plt.subplot(223), plt.imshow(img1[:, :, [2, 1, 0]])
                # plt.show()
            k_halftone_combination_template = np.asarray(k_halftone_combination_template).flatten()
            # print(k_halftone_combination_template)
        elif (a1[0] == 'y'):
            img1_gray_1 = log_shift2center[y_halftone[0][0]:y_halftone[0][0] + 3, y_halftone[0][1]:y_halftone[0][1] + 3]
            y_halftone_combination_template = img1_gray_1
            for i in range(1, len(y_halftone)):
                img1_gray_1 = log_shift2center[y_halftone[i][0]:y_halftone[i][0] + 3, y_halftone[i][1]:y_halftone[i][1] + 3]
                y_halftone_combination_template = np.c_[y_halftone_combination_template, img1_gray_1]
            y_halftone_combination_template = np.asarray(y_halftone_combination_template).flatten()
            # print(y_halftone_combination_template)
        files1_name.append(a1)

    for f2 in files2:
          # 分离文件名和扩展名
        print(f2)
        a2, b2 = os.path.splitext(f2)
        image_name.append(a2)

        img2 = cv2.imread(os.path.join(dir2 + f2))
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        fft2 = np.fft.fft2(img2_gray)
        shift2center = np.fft.fftshift(fft2)
        log_shift2center = np.log(1 + np.abs(shift2center))
        img2_gray_1 = log_shift2center[c_halftone[0][0]:c_halftone[0][0] + 3, c_halftone[0][1]:c_halftone[0][1] + 3]
        img2_gray_2 = log_shift2center[m_halftone[0][0]:m_halftone[0][0] + 3, m_halftone[0][1]:m_halftone[0][1] + 3]
        img2_gray_3 = log_shift2center[k_halftone[0][0]:k_halftone[0][0] + 3, k_halftone[0][1]:k_halftone[0][1] + 3]
        img2_gray_4 = log_shift2center[y_halftone[0][0]:y_halftone[0][0] + 3, y_halftone[0][1]:y_halftone[0][1] + 3]
        c_halftone_combination = img2_gray_1
        m_halftone_combination = img2_gray_2
        k_halftone_combination = img2_gray_3
        y_halftone_combination = img2_gray_4
        files2_name.append(a2)
        for i in range(1,len(c_halftone)):
            img2_gray_1 = log_shift2center[c_halftone[i][0]:c_halftone[i][0] + 3, c_halftone[i][1]:c_halftone[i][1] + 3]
            img2_gray_2 = log_shift2center[m_halftone[i][0]:m_halftone[i][0] + 3, m_halftone[i][1]:m_halftone[i][1] + 3]
            img2_gray_3 = log_shift2center[k_halftone[i][0]:k_halftone[i][0] + 3, k_halftone[i][1]:k_halftone[i][1] + 3]
            img2_gray_4 = log_shift2center[y_halftone[i][0]:y_halftone[i][0] + 3, y_halftone[i][1]:y_halftone[i][1] + 3]
            c_halftone_combination = np.c_[c_halftone_combination, img2_gray_1]
            m_halftone_combination = np.c_[m_halftone_combination, img2_gray_2]
            k_halftone_combination = np.c_[k_halftone_combination, img2_gray_3]
            y_halftone_combination = np.c_[y_halftone_combination, img2_gray_4]


        # print(c_halftone_combination)

        c_halftone_combination = np.asarray(c_halftone_combination).flatten()
        m_halftone_combination = np.asarray(m_halftone_combination).flatten()
        k_halftone_combination = np.asarray(k_halftone_combination).flatten()
        y_halftone_combination = np.asarray(y_halftone_combination).flatten()
        # m_halftone_combination_mean = np.mean(m_halftone_combination)
        # m_halftone_combination1 = m_halftone_combination - m_halftone_combination_mean
        # plt.plot(m_halftone_combination1)
        # plt.xlabel('number')
        # plt.ylabel('x-x_mean')
        # plt.show()
            # image2 = np.asarray(img2).flatten()

      
        res_c = pearsonr(c_halftone_combination, c_halftone_combination_template)
        res_m = pearsonr(m_halftone_combination, m_halftone_combination_template)
        res_k = pearsonr(k_halftone_combination, k_halftone_combination_template)
        res_y = pearsonr(y_halftone_combination, y_halftone_combination_template)
        # dst = matchtemplate_done(img2_gray, img1_gray)

        # print(res_c[0])
        # print(res_m)
        # print(res_k)
        correlate_c.append(res_c[0])
        correlate_m.append(res_m[0])
        correlate_k.append(res_k[0])
        correlate_y.append(res_y[0])
    correlate_c = list(np.array(correlate_c).flatten())
    correlate_m = list(np.array(correlate_m).flatten())
    correlate_k = list(np.array(correlate_k).flatten())
    correlate_y = list(np.array(correlate_y).flatten())
    print(correlate_k)

  
    results = zip(files2_name, correlate_c, correlate_m, correlate_k, correlate_y)

    n_bin,bin,patch = plt.hist(correlate_k,bins=20,range=(-1,1))
    plt.xlabel('Match result')
    plt.show()
    # print(files2_name)

    write_csv(results, 'E:/1.csv')#存储各个图像块的cmky特征


def image_level():#以图像为单位得到各个图像的cmky特征
    csv_reader = csv.reader(open('E:/1.csv'))#读取各个图像块的cmky特征
    results = []
    labels = []
    scores = []
    scores_0 = []
    score_0 = []
    j = []
    imagename = []
    for row in csv_reader:
        results.append(row[0])#读取各个图像块的名称



    csv_reader = csv.reader(open('E:/1.csv'))
    for row in csv_reader:
        scores.append(row[3])#读取各个图像块的某一通道特征


    for i in range(len(scores)):
        scores_0.append(float(scores[i]))
    # print(scores_0)
    k_image = []
    amount = []

    # for i in range(len(results)):#划分标签
    #     if len(results[i].split('_')) == 4:
    #         labels.append(1)
    # 
    #     else:
    #         labels.append(0)

    for i in range(len(results)):

        if (results[i].split("_", -1)[-5:-1] != results[i - 1].split("_", -1)[-5:-1]):

            print(results[i].split("_", -1)[-5:-1])
            # imagename.append(results[i].split("_", -1)[-5:-1])
            imagename.append('_'.join(results[i].split("_")[:-1]))
            score_0 = [float(x) for x in score_0]
            if (score_0 != []):
                amount.append(score_0)
                j.append(np.mean(score_0))
                # print('j:', j)
            if len(results[i].split('_')) == 3:
                labels.append(1)
            else:
                labels.append(0)

            # print('labels:', len(labels))
            score_0 = []
            score_0.append(scores_0[i])
        elif (i != 0 and results[i].split("_", -1)[-5:-1] == results[i - 1].split("_", -1)[-5:-1]):
            score_0.append(scores_0[i])
            # print('score_0:', score_0)
            # k = score_0
    score_0 = [float(x) for x in score_0]
    amount.append(score_0)
    j.append(np.mean(score_0))
    print(amount[1])
    n_bin, bin, patch = plt.hist(amount[1], bins=20, range=(-1, 1))
    for i in range(len(amount)):
        n_bin, bin, patch = plt.hist(amount[i], bins=20, range=(-1, 1))

        k_image.append(n_bin.tolist())
    print(k_image)
    test = pd.DataFrame(index=imagename, data=k_image)
    test.to_csv('E:/k_image.csv')#得到图像某一通道下的20维特征
    results_all = zip(labels)#统计图像标签
    write_csv(results_all, 'E:/labels-patch.csv')#存储图像标签




# def correlate_svm():
#     csv_reader = csv.reader(open('E:/Forgery detection/d1-224-tamperdetection/character/d1-hpm176-character-overlap-rename/d1-hpm176-character-overlap-rename-600dpi1.csv'))
#     results = []
#     labels = []
#     scores = []
#     for row in csv_reader:
#         scores.append(row[1:5])
#     li_int = [map(float, e) for e in scores]
#     x_train = [list(i) for i in li_int]
#     csv_reader = csv.reader(open('E:/Forgery detection/d1-224-tamperdetection/character/d1-hpm176-character-overlap-rename/d1-hpm176-character-overlap-rename-600dpi1.csv'))
#     for row in csv_reader:
#         results.append(row[0])
#     for i in range(len(results)):
#         if len(results[i].split('_')) == 4:
#             labels.append(1)
#         else:
#             labels.append(0)
#     clf = svm.SVC(kernel='rbf', probability=True, decision_function_shape='ovo')
#     clf.fit(x_train, labels)
#     start=time.clock()
#     print(clf.predict_proba(x_train))
#     print(time.clock()-start)

if __name__ == "__main__":
    dir1 = 'E:/'#参考图像块位置
    dir2 = 'E:/'#待测图像块位置
    model_match(dir1, dir2)
    # image_level()
    # correlate_svm()

