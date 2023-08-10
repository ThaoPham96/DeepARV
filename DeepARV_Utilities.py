# Utils from DeepARV 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import keras
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.callbacks import Callback
import tensorflow.experimental.numpy as tnp
import math
import timeit
import datetime, os
import pandas as pd
from numpy import argmax
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import tensorflow_addons as tfa

def load_independent_test_set():
  print('\nLoading independent test set')
  # Load preprocessed X 
  independent_test = pd.DataFrame(np.load('/content/drive/MyDrive/liverpool_hiv/Cleaning_data/numpy_array/X_test_concat.npy'))
  # Load preprocessed Y 
  independent_test['Label'] = np.load('/content/drive/MyDrive/liverpool_hiv/Cleaning_data/numpy_array/Y_test_full.npy')
  print('\nDDI distribution in independent test set')
  print('-------------------------------------------')
  print(count_uniques_ddi(independent_test))
  return independent_test



def count_uniques_ddi(df):
  unique_counts = np.unique(df['Label'], return_counts=True )
  return pd.DataFrame([['Green', 'Yellow', 'Amber', 'Red'], unique_counts[0], unique_counts[1]],  index=['Clinical_label', 'Numeric_label', 'Amount']).T.dropna()

def plot_norm_confusion_matrix(y_pred, y_test):
  cf_matrix = tf.math.confusion_matrix(y_test,y_pred, 4)
  #cf_matrix = np.array(cf_matrix)
  cf_matrix_cal = np.array(cf_matrix)/np.sum(cf_matrix,axis=1)[:,np.newaxis]
  cf_matrix_normalised = np.around(cf_matrix_cal,2)
  group_counts = [value for value in cf_matrix_normalised.flatten()]
  labels = [f"{v1}"
            for v1 in
            group_counts]
  labels= np.asarray(labels).reshape(len(cf_matrix_normalised),
                                      len(cf_matrix_normalised))
  sns.set(font_scale=1)

  xy_label = ['Green','Yellow','Amber','Red']
  # Aspect ratio (e.g., 4:3 or 16:9)
  aspect_ratio =(4, 3/2)

  # Page size in inches (Letter size)
  page_width_inch = 8.5
  page_height_inch = 11

  # Calculate the figure size
  fig_width = page_width_inch  # Use full page width
  fig_height = fig_width * aspect_ratio[1] / aspect_ratio[0]


  plt.figure(figsize=(fig_width,fig_height), dpi=150)

  res = sns.heatmap(data=cf_matrix_normalised, cmap='Purples', annot=True, fmt=' ', linewidths=0.5, linecolor='white', annot_kws={'size': 11.5}, xticklabels = xy_label,
                        yticklabels = xy_label)
  res.tick_params(axis='x', labelsize=10)
  res.tick_params(axis='y', labelsize=10)
  plt.xlabel('DDI Prediction', size=11,fontweight='bold')
  plt.ylabel('True DDI', size=11,fontweight='bold')
  plt.title('Normalised Confusion Matrix Heatmap', fontsize=12, fontweight='bold')

def plot_confusion_matrix(y_pred, y_test):
    # Function to create a light grey colormap
    def create_light_grey_colormap():
        light_grey = '#f7f7f7'
        # Create the colormap using LinearSegmentedColormap.from_list()
        cmap = LinearSegmentedColormap.from_list('single_light_grey', [light_grey, light_grey], N=256)

        return cmap
    grey_cmap = create_light_grey_colormap()

    confusion_matrix_com = tf.math.confusion_matrix(y_test,y_pred, 4)

    xy_label = ['Green','Yellow','Amber','Red', 'Total']


    cf_matrix = np.array(confusion_matrix_com)
    cf_matrix = np.hstack((cf_matrix,cf_matrix.sum(axis=1).reshape(-1,1)))
    cf_matrix = np.vstack((cf_matrix,cf_matrix.sum(axis=0).reshape(1,-1)))

    group_counts = [value for value in cf_matrix.flatten()]
    labels = [f"{v1}"
              for v1 in
              group_counts]
    labels= np.asarray(labels).reshape(len(cf_matrix),len(cf_matrix))
    sns.set(font_scale=1)

    cf_values = pd.DataFrame(cf_matrix.copy())
    cf_values.iloc[:, -1] = float('nan')
    cf_values.iloc[-1, :] = float('nan')

    cf_total = pd.DataFrame(cf_matrix.copy())
    cf_total.iloc[:-1, :-1] = float('nan')

    # Aspect ratio (e.g., 4:3 or 16:9)
    aspect_ratio = (4, 3)

    # Page size in inches (Letter size)
    page_width_inch = 8.5
    page_height_inch = 11

    # Calculate the figure size
    fig_width = page_width_inch  # Use full page width
    fig_height = fig_width * aspect_ratio[1] / aspect_ratio[0]
    fig, ax = plt.subplots(figsize=(fig_width,fig_height), dpi=300)
    fig.suptitle('Confusion Matrix Heatmap', fontsize=12, fontweight='bold')

    res = sns.heatmap(ax=ax, data=cf_values, cmap='Blues', annot=True, fmt='.0f', linewidths=0.5, linecolor='white', annot_kws={'size': 11.5}, )
    res = sns.heatmap(ax=ax, data=cf_total, cmap=grey_cmap, annot=True, fmt='.0f', linewidths=0.5, linecolor='white', annot_kws={'size': 11.5}, xticklabels = xy_label,
                          yticklabels = xy_label)
    res.tick_params(axis='x', labelsize=10)
    res.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('DDI Prediction', size=11,fontweight='bold')
    ax.set_ylabel('True DDI', size=11,fontweight='bold')


def print_final_result(macro_results):
  df_hold = pd.DataFrame(columns =['Accuracy','Precision','Sensitivity',
                                'Specificity','F1_score', 'Balanced Accuracy'])
  df_result = {}
  for j in range(5):
    df_result[j] = pd.DataFrame(macro_results[j],index=['Green','Yellow','Amber','Red','Macro'],
                      columns =['Accuracy','Precision','Sensitivity',
                                'Specificity','F1_score', 'Balanced Accuracy'])
    df_hold.append(df_result[j])

  pd.set_option('float_format', '{:.4f}'.format)
  pd.set_option('display.max_column',None)
  pd.set_option("expand_frame_repr", False)
  pd.set_option('display.width',None)
  pd.set_option("colheader_justify", 'left')

  #print(df_result)
  return df_result


#This function plots heatmap
#for a given confusion matrix
        #require: confusion matrix,
        #...classes for x and y axis,
        #...and title label for x and y axis

def plot_normalised_cf(cf_matrix,
                          x_class_labels, y_class_labels,
                          axis_x_title,axis_y_title):

    #cf_matrix = np.array(cf_matrix)
    cf_matrix_cal = np.array(cf_matrix)/np.sum(cf_matrix,axis=1)[:,np.newaxis]
    cf_matrix_normalised = np.around(cf_matrix_cal,2)
    group_counts = [value for value in cf_matrix_normalised.flatten()]
    labels = [f"{v1}"
              for v1 in
              group_counts]
    labels= np.asarray(labels).reshape(len(cf_matrix_normalised),
                                       len(cf_matrix_normalised))
    sns.set(font_scale=1)
    from matplotlib import pyplot
    pyplot.figure(
                  figsize=(5, 2),
                  dpi=150
                  ) # width and height in inches
    res = sns.heatmap(
                      cf_matrix_normalised, annot=labels,
                      fmt='', annot_kws={'size': 11.5},
                      cmap='Purples',
                      xticklabels = x_class_labels,
                      yticklabels = y_class_labels
                      )
    res.set_xticklabels(
                        res.get_xmajorticklabels(),
                        fontsize = 10.5
                        )
    res.set_yticklabels(
                        res.get_ymajorticklabels(),
                        fontsize = 10.5
                        )
    plt.xlabel(axis_x_title, size=11,fontweight='bold')
    plt.ylabel(axis_y_title, size=11,fontweight='bold')


#This function calculates true negative (tn),
    #false positive (fp), false negative (fn),
    #and true positive (tp) for each DDI class
    #returns: accuracy, precision, sensitivity,
    #specificity, f1 score and balanced accuracy

def calculate_macro_metrics(df_vote, test):
  y_p_ohe = tf.keras.utils.to_categorical(df_vote.final_vote)
  y_t_ohe = tf.keras.utils.to_categorical(test.iloc[:,-1])
  metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=4)
  metric.update_state(y_p_ohe, y_t_ohe)
  result = metric.result()
  result = result.numpy()
  class_0 = metrics_func(result[0])
  class_1 = metrics_func(result[1])
  class_2 = metrics_func(result[2])
  class_3 = metrics_func(result[3])

  # Macro metrics
  #accuracy, precision,  sensitivity, specificity, f1_score, BAcc
  macro = np.mean((class_0, class_1, class_2, class_3), axis=0)
  #print('\n MACOR', macro)
  macros = macro.reshape(1,-1)
  each_class_results = np.vstack((class_0,class_1,class_2,class_3,macros))
  return macros, each_class_results
  
def metrics_func(matrix):
  tn = matrix[0][0]
  fp = matrix[1][0]
  fn = matrix[0][1]
  tp = matrix[1][1]

  accuracy = (tp+tn)/(tp+tn+fp+fn)
  precision = tp/(tp+fp)
  specificity = tn/(tn+fp)
  sensitivity = tp/(tp+fn)
  f1_score = (2*((precision*sensitivity) /
                   (precision+sensitivity)))
  BAcc = (sensitivity + specificity) / 2
  return accuracy, precision, sensitivity, specificity, f1_score, BAcc
  
#This function perform stratified 5-fold split for each DDI class
def stratified_fold_split(pd_df,     class_num = 4, folds=5):
    class_dict = {}
    for i in range(class_num):
        class_dict[i] = pd_df[pd_df.Label == i].reset_index(drop=True)
    class0_split = [class_dict[0][j::5] for j in range(5)]
    class1_split = [class_dict[1][j::5] for j in range(5)]
    class2_split = [class_dict[2][j::5] for j in range(5)]
    class3_split = [class_dict[3][j::5] for j in range(5)]

    fold = {}
    frame ={}
    for k in range(folds):
        frame[k] = [class0_split[k],class1_split[k],
                    class2_split[k], class3_split[k]]
        fold[k] = pd.concat(frame[k],ignore_index=True)
        print('--------------------------')
        print(count_uniques_ddi(fold[k]))
    return fold

def load_training_data():
  print('Loading data...')
  x_train = np.load('/content/drive/MyDrive/liverpool_hiv/Cleaning_data/numpy_array/X_train_concat.npy')
  y_train = np.load('/content/drive/MyDrive/liverpool_hiv/Cleaning_data/numpy_array/Y_train_full.npy')
  train_df = pd.DataFrame(data=x_train)
  train_df['Label'] = y_train
  unique, counts = np.unique(y_train, return_counts=True)
  print('Distribution of DDI class in training set')
  print(pd.DataFrame((['Green', 'Yellow', 'Amber', 'Red'], unique, counts), index=['Clinical_label', 'Numeric_label', 'Amount']).T)
  print('\n########################################\n')
  print('Spliting data into 5 stratified folds...')
  train_5fold = stratified_fold_split(train_df,   class_num = 4, folds=5)
  return train_5fold 


def split_majority(df, n_models,n=1604):
  '''
  Splits majority samples into n_models splits
  df - df containing majoirt
  n_models - number of models/data splits
  n - number of samples per data split
  '''
  # Dictionary to store majority splits
  majority_splits = {}
  for i in range(n_models):
    if i < (n_models-1):

      start = i*n
      end = (i+1)*n
      majority_splits[i] = df.iloc[start:end,:]
    else:
      majority_splits[i] = pd.concat([df.iloc[end:,:],df.sample(n=n)])
  return majority_splits




