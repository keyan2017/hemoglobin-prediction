import numpy as np
import os
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error


root = r'E:\experiment\maskrcnn\testdata\conjunctiva'
imgs = list(sorted(os.listdir(root)))

hb_pd = pd.read_excel('testdata/hb.xlsx')
hb_pd['name'] = hb_pd['name'].astype(str)

# Calculating picture entropy
def image_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    probabilities = histogram / float(np.sum(histogram))
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-7))
    return entropy

# Calculating BBHR
def calc_bhhr(image):
    hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsi_image[:, :, 0]
    total_pixels = h_channel.size
    above_100_pixels = np.count_nonzero(h_channel > 100)
    hhr = above_100_pixels / total_pixels
    if hhr == 0:
        bhhr = 0
    else:
        bhhr = 1
    return bhhr

 # Calculating PVM
def calc_pvm(image):
    channels = cv2.split(image)
    pvm_bgr = []
    for channel in channels:
        channel_flat = channel.flatten()
        sorted_values = np.sort(channel_flat)
        q40_index = int(0.91 * len(sorted_values))
        q60_index = int(0.94 * len(sorted_values))
        pvm_values_channel = sorted_values[q40_index:q60_index]
        pvm = np.mean(pvm_values_channel)
        pvm_bgr.append(pvm)
    return pvm_bgr

# Calculating bright
def calc_bright(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonzero_pixels = np.nonzero(gray_image)
    nonzero_values = gray_image[nonzero_pixels]
    bright = np.mean(nonzero_values)
    return bright

def color_drbt(image):
    channels = cv2.split(image)
    color_bgr = []
    for channel in channels:
        channel_flat = channel.flatten()
        sorted_values = np.sort(channel_flat)
        q00_index = int(0.85 * len(sorted_values))
        q100_index = int(100 * len(sorted_values))
        q04_index = int(0.853 * len(sorted_values))
        q24_index = int(0.868 * len(sorted_values))
        q50_index = int(0.925 * len(sorted_values))
        q52_index = int(0.928 * len(sorted_values))
        q88_index = int(0.982 * len(sorted_values))
        q98_index = int(0.997 * len(sorted_values))
        values_00_100 = sorted_values[q00_index:q100_index]
        values_04_14 = sorted_values[q04_index:q24_index]
        values_50_52 = sorted_values[q50_index:q52_index]
        values_88_98 = sorted_values[q88_index:q98_index]
        color_0 = np.mean(values_00_100)
        color_1 = np.mean(values_04_14)
        color_2 = np.mean(values_50_52)
        color_3 = np.mean(values_88_98)
        color = [color_0, color_1, color_2, color_3]
        color_bgr.append(color)
    return color_bgr

df = pd.DataFrame(columns=['entropy', 'bhhr', 'pvm_b', 'pvm_g', 'pvm_r', 'bright',
                               'b_0', 'b_1', 'b_2', 'b_3', 'g_0', 'g_1',
                               'g_2', 'g_3', 'r_0', 'r_1', 'r_2', 'r_3', 'hb'])
# Build data set
for idx in range(0, 1065):
    img_path = os.path.join(root, imgs[idx])
    image = cv2.imread(img_path)
    list = []
    entropy = [image_entropy(image)]
    bhhr = [calc_bhhr(image)]
    pvm = calc_pvm(image)
    bright = [calc_bright(image)]
    drbt = np.array(color_drbt(image)).flatten()
    list = [entropy, bhhr, pvm, bright, drbt]
    new_list = []
    for arr in list:
        for item in arr:
            new_list.append(item)

    name = imgs[idx][:-4]
    for i in range(0, 1065):
        if name == hb_pd.iloc[i, 0]:
            hb = hb_pd.iloc[i, 1]
    new_list.append(hb)
    df = df.append(pd.Series(new_list, index=df.columns), ignore_index=True)

# data standardization
zs = StandardScaler()
df.iloc[:, :-1] = zs.fit_transform(df.iloc[:, :-1])

# Divide the training set and test set
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construct regression model
model_LR = LinearRegression().fit(X_train, y_train)
model_DT = DecisionTreeRegressor().fit(X_train, y_train)
model_SVR = SVR(kernel='rbf').fit(X_train, y_train)
model_GBR = GradientBoostingRegressor().fit(X_train, y_train)

# Make predictions on the test set
pre_LR = model_LR.predict(X_test)
pre_DT = model_DT.predict(X_test)
pre_SVR = model_SVR.predict(X_test)
pre_GBR = model_GBR.predict(X_test)

# Build a DataFrame that predicts the result
pred = pd.DataFrame()
pred['y_test'] = y_test
pred['pre_LR'] = pre_LR
pred['pre_DT'] = pre_DT
pred['pre_SVR'] = pre_SVR
pred['pre_GBR'] = pre_GBR
# pred.to_excel('testdata/prediction.xlsx', index=False)

# evaluation model
evalu = pd.DataFrame()
evalu['model_name'] = ['LR', 'DT', 'SVR', 'GBR']

r2_LR = r2_score(y_test, pre_LR)
r2_DT = r2_score(y_test, pre_DT)
r2_SVR = r2_score(y_test, pre_SVR)
r2_GBR = r2_score(y_test, pre_GBR)
evalu['R2'] = [r2_LR, r2_DT, r2_SVR, r2_GBR]

evs_LR = explained_variance_score(y_test, pre_LR)
evs_DT = explained_variance_score(y_test, pre_DT)
evs_SVR = explained_variance_score(y_test, pre_SVR)
evs_GBR = explained_variance_score(y_test, pre_GBR)
evalu['EVS'] = [evs_LR, evs_DT, evs_SVR, evs_GBR]

mae_LR = mean_absolute_error(y_test, pre_LR)
mae_DT = mean_absolute_error(y_test, pre_DT)
mae_SVR = mean_absolute_error(y_test, pre_SVR)
mae_GBR = mean_absolute_error(y_test, pre_GBR)
evalu['MAE'] = [mae_LR, mae_DT, mae_SVR, mae_GBR]

print(df)
print(evalu)