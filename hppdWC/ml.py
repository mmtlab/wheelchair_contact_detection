# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:05:55 2022

@author: eferlius
"""

#%% IMPORTS
import matplotlib.pyplot as plt
import pandas as pd
import winsound
import time
import numpy as np
import scipy
import os
import csv
import vlc
import sklearn
import random

from . import plots


class Database:
    def __init__(self, pd_table, features_names, label_column, labels_names = None, dataBase_name = None):
        '''
        Creates a database and contains few functions to speed up common processes like 
        - showing the data 
        - extracting part of samples 
        - extracting given features 
        - splitting into test and train
        
        Parameters
        ----------
        pd_table : pandas dataframe
            Should have a header. The columns present in features_name are the features, the column whose name is label_column are the labels
        features_names : list of strings
            DESCRIPTION.
        label_column : string
            DESCRIPTION.
        labels_names : list of strings, optional
            name of the labels. The default is None.
        dataBase_name : string, optional
            name of the database. The default is None.

        Returns
        -------
        None.

        '''
        self.table = pd_table # original dataframe
        self.features_names = features_names # list of strings
        # extraction of the pandas df
        self.X_table = self.table[features_names] # cropped dataframe on columns, contains the features
        self.y_table = self.table[label_column] # only the column containing the labels
        # only the values
        self.X = self.X_table.values #2d array
        self.y = self.y_table.values #1d array

        self.nsampl = len(self.y)
        if not labels_names:
            self.labels_names = np.unique(self.y).tolist()
        else:
            self.labels_names = labels_names
        if not dataBase_name:
            self.name = ''
        else:
            self.name = dataBase_name

    def __str__(self):
        return ('dataBase of: \n- {nsampl} samples\n- {nfeat} features {featnames}\n- {nlabel} labels {labnames}'.\
        format(nsampl = self.nsampl, nfeat = len(self.X[0]), featnames = str(self.features_names), nlabel = len(np.unique(self.y)), labnames = str(self.labels_names)))


    def reduceSamples(self, n_samples = -1):
        '''
        Returns n_samples randomly taken from the database

        Parameters
        ----------
        n_samples : int, optional
            number of values to be randomly extracted. The default is -1.

        Returns
        -------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        X_table : TYPE
            DESCRIPTION.
        y_table : TYPE
            DESCRIPTION.

        '''
        # eventually reducing the dimension
        if n_samples > 0:
            indexes = np.sort(random.sample(range(0, self.nsampl), n_samples))
            X = self.X[indexes]
            y = self.y[indexes]
            X_table = self.X_table.iloc[indexes,:]
            y_table = self.y_table.iloc[indexes]
        else: # all the samples
            X = self.X
            y = self.y
            X_table = self.X_table
            y_table = self.y_table
        return X, y, X_table, y_table

    def train_test_split(self, test_size = 0.2, n_samples = -1):
        '''
        Splits the database into train and test.

        Parameters
        ----------
        test_size : float (0,1), optional
            How many samples for the test. The default is 0.2.
        n_samples : int, optional
            To reduce the number of samples.
            Please refer to reduceSamples(). 
            The default is -1.

        Returns
        -------
        X_train : 2d array
            features of the train data.
        X_test : 2d array
            features of the test data.
        y_train : 1d array
            labels of the train data.
        y_test : 1d array
            labels of the test data.

        '''
        # eventually reducing the samples. Keeping the arrays in order to have only them and not dataframes
        X, y, _, _ = self.reduceSamples(n_samples)

        # calling th function of sklearn
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = test_size)

        return X_train, X_test, y_train, y_test

    def scatterFeatures(self, n_samples = -1, featNames = None, labelColors = [''], mainTitle = '', alpha_scatter = 0.5, marker_scatter = '.', s_scatter = 1, alpha_hist = 0.5, bins = 20, superimpose = True, labelNames = None):
        '''
        Plots in a matrix one feature with respect to the other.
        On the diagonal, draws an histogram with the distribution of the given 
        feature for the different labels

        Parameters
        ----------
        n_samples : int, optional
            To reduce the number of samples.
            Please refer to reduceSamples(). 
            The default is -1.
        featNames : list of strings, optional
            features to be showed. The default is None, which means all the features
        Please refer to plots.scatterFeatures for the other parameters

        Returns
        -------
        Please refer to plots.scatterFeatures 
        '''
        # eventually reducing the samples. Keeping the tables to act on feat names
        _, _, X_table, y_table = self.reduceSamples(n_samples)

        # eventually considering only the given features
        if featNames == None:
            featNames = self.features_names
        if labelNames == None:
            labelNames = self.labels_names
        mainTitle = self.name + mainTitle


        # calling the function to plot
        fig, ax = plots.scatterFeatures(X_table[featNames].values, y_table, featNames, labelColors, mainTitle, alpha_scatter, marker_scatter, s_scatter, alpha_hist, bins, superimpose, labelNames)

        return fig, ax
