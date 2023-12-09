# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:23:06 2020

@author: 13390
"""
import sklearn
import os
import time
import shutil
import pandas as pd
import sklearn.preprocessing as prep
import numpy as np
np.set_printoptions(threshold=np.inf) 
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
os.environ['CUDA_VISIBLE_DEVICES']='1'
fold = 5


def He_init(shape, name=None):
    # He distribution
    # n = np.cumprod(shape)[-1]
    if np.size(shape) == 2:
        n = shape[0] + shape[1]
    else:
        n = shape[0] * shape[1] * shape[2] 
    stddev = np.sqrt(2.0 / n)
    return tf.Variable(tf.random.truncated_normal(shape, mean=0, stddev=stddev), trainable=True, name=name)  


def Xavier_init(shape, name=None):
    # Xavier distribution
    n = sum(shape)
    stddev = np.sqrt(2.0 / n)
    return tf.Variable(tf.truncated_normal(shape, mean=0, stddev=stddev), trainable=True, name=name)


class brainnetCNN():
    def __init__(self, batch_size, L2lamda): # 5e-3
        self.config = tf.compat.v1.ConfigProto() 
        self.config.gpu_options.allow_growth = True 
        self.initFunc = He_init  #  He_init
        self.n_e2e_filter = []
        self.n_e2n_filter = [64]
        self.n_n2g_filter = [128]
        self.l_hashcode = 24
        self.fc_size = [96]
        self.n_class = 2
        self.n_label = 8
        # Model type
        self.E2Etype = None  # self.CNNE2E OR self.XWE2E
        self.E2Ntype = [self.XWE2N]  # self.CNNE2N OR self.XWE2N
        self.N2Gtype = [self.CNNN2G]  # self.CNNN2G OR self.XWN2G
        
        self.batch_size = batch_size
        self.n_roi = 90
        self.conv_acti = tf.nn.relu  # tf.nn.leaky_relu
        self.leaky_relu_alpha = 0.33
        self.keep_prob_e2e = 1.0  
        self.keep_prob_e2n = 1.0
        self.keep_prob_n2g = 1.0
        self.keep_prob_fc = 0.8
        self.fc_acti = tf.nn.relu
        self.L1lamda = 5e-3
        self.L2lamda = L2lamda
        self.w_lamda = 1  
        self.n_evaluation_epochs = 10
        self.n_patience = 50  # 50
        self.threshold = 1e-3  # 5e-2
        self.net_density = 0.6
        self.beta = 1.0
        self.if_kernelsymmetry=True
        
    def CNNE2E(self, inputs, in_channels, out_channels, index, activation=tf.nn.relu):
        w_r = self.initFunc([1, self.n_roi, in_channels, out_channels], 'E2E' + str(index) + '_w_r')  
        r_outputs = tf.nn.conv2d(inputs, w_r, strides=[1, 1, 1, 1], padding='VALID', name='E2E' + str(index) + '_rs')  
        bias = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='E2E' + str(index) + '_bias') 
    
        if self.if_kernelsymmetry:  
            w_c = tf.transpose(w_r, [1,0,2,3])  
            c_outputs = tf.transpose(r_outputs,[0,2,1,3])
        else:
            w_c = self.initFunc([self.n_roi, 1, in_channels, out_channels], 'E2E2' + str(index) + '_w_c')
            c_outputs = tf.nn.conv2d(inputs, w_c, strides=[1, 1, 1, 1], padding='VALID', name='E2E2' + str(index) + 'cs')
        outputs = tf.nn.bias_add(tf.add(r_outputs, c_outputs), bias)
        #outputs = tf.layers.batch_normalization(outputs, training = self.training_flag)
        if activation is tf.nn.leaky_relu:  
            return activation(outputs, self.leaky_relu_alpha), [tf.concat([tf.expand_dims(w_r,0),tf.expand_dims(tf.transpose(w_c, [1,0,2,3]),0)],0)]
        else:
            return activation(outputs), [tf.concat([tf.expand_dims(w_r,0),tf.expand_dims(tf.transpose(w_c, [1,0,2,3]),0)],0)]

    def XWE2N(self, inputs, in_channels, out_channels, index, activation=tf.nn.relu):
        # inputs : [batch, in_height, in_width, in_channels]
        # filter : [filter_height, filter_width, in_channels, out_channels]
        # weights = xavier_init([1, self.n_roi, in_channels, out_channels], 'E2ENEW' + str(index) + '_filters')
        bias = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='XWE2N' + str(index) + '_bias')
        XWlayer = keras.layers.local.LocallyConnected2D(out_channels, kernel_size=(1, 90), kernel_initializer='he_normal',if_kernelsymmetry=False)

        # weights = XWlayer.weights
        # XWlayer.weights[0] = tf.add(weights[0], tf.transpose(weights[0], [1, 0, 2]))
        outputs = XWlayer(inputs)
        outputs = tf.nn.bias_add(outputs,bias)
        kernel = XWlayer.getKernel()
        if activation is tf.nn.leaky_relu:
            return activation(outputs, self.leaky_relu_alpha), [kernel]  # tf.convert_to_tensor()
        else:
            return activation(outputs), [kernel]  # tf.convert_to_tensor()
            
    def CNNE2N(self, inputs, in_channels, out_channels, index, activation=tf.nn.relu):
         w_r = self.initFunc([1, self.n_roi, in_channels, out_channels], 'E2N' + str(index) + '_r_filters')
         bias = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='E2N' + str(index) + '_bias')
         r_outputs = tf.nn.conv2d(inputs, w_r, strides=[1, 1, 1, 1], padding='VALID', name='E2N' + str(index) + '_r_layer')
 
         if self.if_kernelsymmetry:
             w_c = tf.transpose(w_r, [1,0,2,3])
             outputs = r_outputs
         else:
             w_c = self.initFunc([self.n_roi, 1, in_channels, out_channels], 'CNNE2N' + str(index) + '_c_filters')
             c_outputs = tf.nn.conv2d(inputs, w_c, strides=[1, 1, 1, 1], padding='VALID', name='CNNE2N' + str(index) + '_c_layer')
             outputs = tf.nn.bias_add(tf.add(r_outputs, tf.transpose(c_outputs, [0, 2, 1, 3])), bias)
         #outputs = tf.layers.batch_normalization(outputs, training = self.training_flag)
         if activation is tf.nn.leaky_relu:
             return activation(outputs, self.leaky_relu_alpha), [tf.concat([tf.expand_dims(w_r,0),tf.expand_dims(tf.transpose(w_c, [1,0,2,3]),0)],0)]
         else:
             return activation(outputs), [tf.concat([tf.expand_dims(w_r,0),tf.expand_dims(tf.transpose(w_c, [1,0,2,3]),0)],0)]
  
    def CNNN2G(self, inputs, in_channels, out_channels, index, activation=tf.nn.relu):
        filters = self.initFunc([self.n_roi, 1, in_channels, out_channels], 'N2G' + str(index) + '_filters')
        bias = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='N2G' + str(index) + '_bias')
        outputs = tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding='VALID', name='N2G' + str(index) + '_layer')
        outputs = tf.nn.bias_add(outputs, bias)
        #outputs = tf.layers.batch_normalization(outputs, training = self.training_flag)
        if activation is tf.nn.leaky_relu:
            return activation(outputs, self.leaky_relu_alpha), filters
        else:
            return activation(outputs), filters

    def FC(self, inputs, hidden_size, index, activation=tf.nn.sigmoid):
        inputs_size = 1
        inputs_shape = inputs.get_shape().as_list()  
        for i in range(len(inputs_shape) - 1):
            inputs_size = inputs_size * inputs_shape[i + 1]
        inputs = tf.reshape(inputs, [-1, inputs_size])  
        weights = self.initFunc([inputs_size, hidden_size], 'FC' + str(index) + '_weights')
        bias = tf.Variable(tf.constant(0.0, shape=[hidden_size, ]), name='FC' + str(index) + '_bias')
        outputs = tf.add(tf.matmul(inputs, weights), bias)  
        return activation(outputs, name='FC' + str(index)), weights
    
    def cal_pinv(self, A):
        s, u, v = tf.linalg.svd(A)
        s_no0 = tf.count_nonzero(s, dtype=tf.int32)
        s_all = tf.shape(s)[0]
        s_pre = s[0:s_no0,]
        s_pre_inv = tf.matrix_inverse(tf.matrix_diag(s_pre))
        s_inv = tf.pad(s_pre_inv,[[0,s_all-s_no0],[0,s_all-s_no0]],"CONSTANT")
        A_inv = tf.matmul(tf.matmul(v, s_inv), tf.transpose(u))
        return A_inv
    
    def comput_similarity(self, label_train):
        n = self.batch_size
        sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if label_train[i,0] == label_train[j,0]:
                    sim[i,j] = 1
                else:
                    sim[i,j] = 0
        return sim

    def makeGraph(self):
        self.x = tf.compat.v1.placeholder(tf.float32, (None, self.n_roi, self.n_roi, 1), name='x')
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.n_label], name='y') #
        self.y2 = tf.compat.v1.placeholder(tf.float32, [None, self.n_class], name='y')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.placeholder(tf.float32)
        self.training_flag = tf.placeholder(tf.bool)

        layer = self.x
        in_channels = 1
        if self.training_flag is True:
            keep_prob_e2e = self.keep_prob_e2e
            keep_prob_e2n = self.keep_prob_e2n
            keep_prob_n2g = self.keep_prob_n2g
            keep_prob_fc = self.keep_prob_fc
        else:
            keep_prob_e2e = 1.0
            keep_prob_e2n = 1.0
            keep_prob_n2g = 1.0
            keep_prob_fc = 1.0

        # E2E
        self.w_e2e = []
        for index in range(len(self.n_e2e_filter)):  # len(self.n_e2e_filter)=1
            layer, weightList = self.E2Etype[index](layer, in_channels, self.n_e2e_filter[index], index, self.conv_acti)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_e2e)
            #layer = tf.Print(layer,['E2Elayer:',layer])
            for weights in weightList:  
                L2loss = tf.contrib.layers.l2_regularizer(self.L2lamda)(weights)
                tf.add_to_collection('L2Losses', L2loss)  
                self.w_e2e.append(weights)  
            in_channels = self.n_e2e_filter[index]
            hidden_size = self.n_e2e_filter[index] * self.n_roi * self.n_roi

        # E2N
        self.w_e2n = []
        self.w_e2nA= []
        for index in range(len(self.n_e2n_filter)):
            layer, weightList= self.E2Ntype[index](layer, in_channels, self.n_e2n_filter[index], index, self.conv_acti)
            hidden_size = self.n_e2n_filter[index] * self.n_roi
            L2loss = tf.contrib.layers.l2_regularizer(self.L2lamda)(weightList[0])
            tf.add_to_collection('L2Losses', L2loss)
            self.w_e2n.append(weightList[0])
            if len(weightList) > 1:
                L2loss = tf.contrib.layers.l2_regularizer(self.L2lamda)(weightList[1])
                tf.add_to_collection('L2Losses', L2loss)
                self.w_e2nA.append(weightList[1])
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_e2n)
            in_channels = self.n_e2n_filter[index]
            #layer = tf.Print(layer,['E2Nlayer:',layer])

        # N2G
        self.w_n2g = []
        for index in range(len(self.n_n2g_filter)):
            layer, weights = self.N2Gtype[index](layer, in_channels, self.n_n2g_filter[index], index, self.conv_acti)
            hidden_size = self.n_n2g_filter[index]
            L2loss = tf.contrib.layers.l2_regularizer(self.L2lamda)(weights)
            tf.add_to_collection('L2Losses', L2loss)
            in_channels = self.n_n2g_filter[index]
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_n2g)
            self.w_n2g = weights
            #layer = tf.Print(layer,['N2Glayer:',layer])

        # FC
        self.w_fc = []
        for index in range(len(self.fc_size)):
            layer, weights = self.FC(layer, self.fc_size[index], index, activation=self.fc_acti)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob_fc)
            hidden_size = self.fc_size[index]
            L2loss = tf.contrib.layers.l2_regularizer(self.L2lamda)(weights)
            tf.add_to_collection('L2Losses', L2loss)
            self.w_fc.append(weights)
            #layer = tf.Print(layer,['FClayer:',layer])

    # Softmax Layer
        weight_softmax = self.initFunc([hidden_size, self.n_class], name='weight_softmax')
        self.w_softmax = weight_softmax
        L2loss = tf.contrib.layers.l2_regularizer(self.L2lamda)(weight_softmax)
        tf.add_to_collection('L2Losses', L2loss)

        bias_softmax = tf.Variable(tf.constant(0.0, shape=[self.n_class, ]), name='bias_softmax')
        layer = tf.reshape(layer, [-1, hidden_size])
        self.y_pre_sm = tf.nn.bias_add(tf.matmul(layer, weight_softmax), bias_softmax, name='softmax_layer')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pre_sm, labels=self.y2)

    # Hash Layer
        weight_hash = self.initFunc([hidden_size, self.l_hashcode], name='weight_hash')
        self.w_hash = weight_hash
        L2loss = tf.contrib.layers.l2_regularizer(self.L2lamda)(weight_hash)
        tf.add_to_collection('L2Losses', L2loss)

        bias_hash = tf.Variable(tf.constant(0.0, shape=[self.l_hashcode, ]), name='bias_hash')
        layer = tf.reshape(layer, [-1, hidden_size])
        self.y_pre = tf.nn.bias_add(tf.matmul(layer, weight_hash), bias_hash, name='hash_layer')

        self.w_reg = self.initFunc([self.n_label, self.l_hashcode], name='weight_reg')
        L2loss = tf.contrib.layers.l2_regularizer(self.L2lamda)(self.w_reg)
        tf.add_to_collection('L2losses', L2loss)
        reg_loss = tf.square(tf.subtract(self.y_pre, tf.matmul(self.y, self.w_reg))) #+ tf.square(tf.norm(self.w_reg, ord=2))
        
        sita_ij = 0.5*tf.matmul(self.y_pre,tf.transpose(self.y_pre))  #self.y_hash
        self.S_train = tf.cast(self.comput_similarity(self.y), dtype=tf.float32)
        sim_loss = tf.log(1.0+tf.exp(sita_ij)) - tf.matmul(self.S_train ,sita_ij)

        self.loss = tf.reduce_sum(reg_loss) / self.batch_size + tf.reduce_sum(sim_loss) / self.batch_size + 100*tf.reduce_mean(cross_entropy)
        #tf.reduce_sum(reg_loss) + 0.8*tf.reduce_sum(dif_loss) + 0.5*tf.reduce_sum(sim_loss)


    # loss and optimizer
        if len(tf.get_collection('L1Loss')) != 0:
            self.lossL1 = tf.add_n(tf.get_collection('L1Loss'))
        else:
            self.lossL1 = tf.Variable(0.0, trainable=False)

        if len(tf.get_collection('L2Losses')) != 0:
            self.lossL2 = tf.add_n(tf.get_collection('L2Losses'))
        else:
            self.lossL2 = tf.Variable(0.0, trainable=False)

        self.cost = self.loss + self.lossL2
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.cost, global_step=self.global_step)

        self.y_pre_hash = tf.nn.sigmoid(self.y_pre, name='Hash')
        self.y_hash = (tf.sign(self.y_pre_hash - 0.5) + 1) // 2  # down
   

    def initial(self):
        self.sess.run(tf.global_variables_initializer()) 

    def init(self):
        with tf.Graph().as_default():  
            self.makeGraph()
            self.sess = tf.Session(config=self.config)  
            self.initial()
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1)  
            self.train_writer = tf.compat.v1.summary.FileWriter('/home/ai/data/zhangyaqin/CNNzyq/log/train', self.sess.graph)  
            self.sess.graph.finalize()  

    def train(self, ab_roi, abtype, X_train, X_valid, X_test, Y_train, Y_valid, Y_test, Y_train2, Y_valid2, Y_test2, epochs, learn_rate, fold):
        acc_max = 0.0
        cost_min = 1e8
        best_valid_acc = 0.0
        min_valid_cost = sys.maxsize
        ax = []
        ay = []
        ax2 = []
        ay2 = []
        ax3 = []
        ay3 = []
        plt.ion()
        for e in np.arange(epochs): 
            train_hashcode = list()
            train_hashcode_y = list()
            valid_hashcode = list()
            valid_hashcode_y = list()
            valid_cost_list = list() 
            train_cost_list = list()
            save_valid_cost = list() 
                
            random_index = np.random.permutation(len(Y_train))  
            X_train = X_train[random_index, :, :]
            Y_train = Y_train[random_index]
            Y_train2 = Y_train2[random_index]

            training_steps = np.shape(X_train)[0] // self.batch_size  
            for step in range(training_steps):
                train_batch = X_train[step * self.batch_size: (step + 1) * self.batch_size, :, :] \
                             .reshape([-1, np.shape(X_train)[1], np.shape(X_train)[2], 1])
                y_train_batch = Y_train[step * self.batch_size: (step + 1) * self.batch_size]
                y_train_batch2 = Y_train2[step * self.batch_size: (step + 1) * self.batch_size]
               
                train_cost, _, train_loss, train_lossL1, train_lossL2, global_step, \
                w_e2e, w_e2n, w_e2nA, w_n2g, w_fc, w_softmax, w_hash, w_reg, train_hash_code= \
                    self.sess.run([self.cost, self.optimizer, self.loss, self.lossL1, self.lossL2, self.global_step,
                                   self.w_e2e, self.w_e2n, self.w_e2nA, self.w_n2g, self.w_fc, self.w_softmax, self.w_hash, self.w_reg, self.y_hash],
                                   feed_dict={self.learning_rate: learn_rate,
                                              self.x: train_batch,
                                              self.y: y_train_batch,
                                              self.y2: y_train_batch2,
                                              self.training_flag: True})
                summaries = tf.Summary(value=[tf.Summary.Value(tag="Cost_" + str(fold), simple_value=train_cost),
                                              tf.Summary.Value(tag="Loss_" + str(fold), simple_value=train_loss),
                                              tf.Summary.Value(tag="LossL1_" + str(fold), simple_value=train_lossL1),
                                              tf.Summary.Value(tag="LossL2_" + str(fold), simple_value=train_lossL2)])
                self.train_writer.add_summary(summaries, global_step)

                Str = "Global_step:{} Epoch: {}/{} step:{} Train cost:{}".format(global_step, e + 1, epochs, step, train_cost)
                # print(Str)
                
                train_hashcode.append(train_hash_code)
                train_hashcode_y.append(y_train_batch[:,0]) #
                train_cost_list.append(train_cost)

            hc = np.array(train_hashcode).reshape(-1, self.l_hashcode)  # ku
            y_hc = np.array(train_hashcode_y).reshape(-1)


            valid_batch = X_valid.reshape([-1, np.shape(X_valid)[1], np.shape(X_valid)[2], 1])
            y_valid_batch = Y_valid
            y_valid_batch2 = Y_valid2
            valid_cost, valid_hash_code = self.sess.run([self.cost, self.y_hash],feed_dict={
                                          self.x: valid_batch,
                                          self.y: y_valid_batch,
                                          self.y2: y_valid_batch2,
                                          self.training_flag: False})

            valid_hashcode = np.array(valid_hash_code).reshape(-1, self.l_hashcode)
            valid_hashcode_y = np.array(y_valid_batch[:,0]).reshape(-1)
            # comput_acc
            valid_acc, _, _ = getACC(hc, y_hc, valid_hashcode, valid_hashcode_y)

            Str = "Epoch: {}/{} Valid ACC:{}".format(e + 1, epochs, valid_acc)
            # print(Str)

            test_batch = X_test.reshape([-1, np.shape(X_test)[1], np.shape(X_test)[2], 1])
            y_test_batch = Y_test
            y_test_batch2 = Y_test2
            test_cost, test_hash_code = self.sess.run([self.cost, self.y_hash],
                                        feed_dict={self.x: test_batch,
                                                   self.y: y_test_batch,
                                                   self.y2: y_test_batch2,
                                                   self.training_flag: False})
            test_hashcode = np.array(test_hash_code).reshape(-1, self.l_hashcode)
            test_hashcode_y = np.array(y_test_batch[:,0]).reshape(-1)
            test_acc ,test_sen, test_spe = getACC(hc, y_hc, test_hashcode, test_hashcode_y)
            Str = "Epoch: {}/{} Test ACC:{}\n".format(e + 1, epochs, test_acc)
            # print(Str)
            
            # stop condition
            if e % self.n_evaluation_epochs == 0:
                 
                 plt.figure(1)
                 ax.append(e + 1)
                 ay.append(np.mean(valid_cost))
                 plt.clf()
                 plt.plot(ax, ay)
                 plt.pause(0.01)
                 plt.ioff()
 
                 
                 plt.figure(2)
                 ax2.append(e+1)
                 ay2.append(valid_acc*100)
                 plt.clf()
                 plt.plot(ax2, ay2)
                 plt.pause(0.01)
                 '''
                 if e % 2500 == 0:
                     plt.savefig('/home/ai/data/zhangyaqin/CNNzyq/fig/fold{}_acc.jpg'.format(fold))
                 '''
                 plt.ioff()
                 
                 
                 if valid_cost <= min_valid_cost:
                    p = 0
                   
                    min_valid_cost = valid_cost
                    best_valid_acc = valid_acc
                    best_test_acc = test_acc
                    best_test_sen = test_sen
                    best_test_spe = test_spe
                    Str = "Epoch: {}/{} Best Valid ACC:{} Best Test ACC:{}\n".format(e + 1, epochs, best_valid_acc, best_test_acc)
                    print(Str)

                    sio.savemat('/home/ai/data/zhangyaqin/coef/coef.mat', {'w_e2e': w_e2e,
                                            'w_e2n': w_e2n,
                                            'w_e2nA': w_e2nA,
                                            'w_n2g': w_n2g,
                                            'w_fc': w_fc,
                                            'weights_softmax': w_softmax,
                                            'weights_hash': w_hash,
                                            'weights_reg': w_reg})

                 else :
                    p += 1
                    
                    if best_test_acc <= test_acc:
                        best_test_acc = test_acc
                        best_test_sen = test_sen
                        best_test_spe = test_spe
                    
                    if p>self.n_patience:
                        Str = "Final Epoch: {}/{} Best Valid ACC:{} Best Test ACC:{}\n".format(e + 1, epochs, best_valid_acc, best_test_acc)
                        print(Str)
                        break




        self.sess.close()
        self.train_writer.close()

        return min_valid_cost ,best_valid_acc , best_test_acc, best_test_sen ,best_test_spe

def getACC(hc, y_hc, hc_pre, y):
    y_ = np.zeros(hc_pre.shape[0])
    for i in range(hc_pre.shape[0]):  # 219
        y1 = hc_pre[i, :].astype(int)
        r2 = []
        for j in range(hc.shape[0]):  # 576
            y2 = hc[j, :].astype(int)
            r1 = y1 ^ y2
            r2.append(r1.sum())
        r2_min = min(r2)
        r3 = []
        for k, x in enumerate(r2):
            if x == r2_min:
                r3.append(k)
        r4 = []
        for t in range(len(r3)):
            r4.append(y_hc[r3[t]])
        if r4.count(0) > r4.count(1):  # >=
            y_[i] = 0
        else:
            y_[i] = 1
    correct_prediction = np.equal(y_.astype(int), y.astype(int))
    acc = np.mean(correct_prediction)

    CM = sklearn.metrics.confusion_matrix(y, y_)
    tn, fp, fn, tp = CM.ravel()
    sen = tp / float((tp + fn))
    spe = tn / float((fp + tn))
    return acc, sen, spe

def load_data(fold):
    data = sio.loadmat('./Dataset/ALLASD{}_NETFC_SG_Pear.mat'.format(fold+1))
    X = data['net']
    X_train = data['net_train']
    X_valid = data['net_valid']  # vaild
    X_test = data['net_test']

    Idx = [2,3,4,5,6,7,8,9] # 3:Age 4:Sex 5:Handedness 6:FIQ 7:VIQ 8:PIQ 9:EYE Status
    Y = data['phenotype'][:, Idx]
    Y_train = data['phenotype_train'][:,Idx]
    Y_valid = data['phenotype_valid'][:,Idx]
    Y_test = data['phenotype_test'][:, Idx]
    col_idx = [1, 4, 5, 6] # 3:Age 6:FIQ 7:VIQ 8:PIQ 全面智商、语言智商和操作智商
    Y[:, col_idx], Y_train[:, col_idx], Y_valid[:, col_idx], Y_test[:, col_idx] = mapStd2(Y[:, col_idx],
                                                                                          Y_train[:, col_idx],
                                                                                          Y_valid[:, col_idx],
                                                                                          Y_test[:, col_idx])
    col_idx = [2, 3, 7]
    Y[:, col_idx], Y_train[:, col_idx], Y_valid[:, col_idx], Y_test[:, col_idx] = mapMinmax2(Y[:, col_idx],
                                                                                             Y_train[:, col_idx],
                                                                                             Y_valid[:, col_idx],
                                                                                             Y_test[:, col_idx])
    Y_train2 = data['phenotype_train'][:, 2]
    Y_valid2 = data['phenotype_valid'][:, 2]
    Y_test2 = data['phenotype_test'][:, 2]

    return X, X_train, X_valid, X_test, Y_train, Y_valid, Y_test, Y_train2, Y_valid2, Y_test2

def load_data_valid(fold):
    data = sio.loadmat('/home/ai/data/zhangyaqin/Dataset/ALLASD{}_NETFC_SG_Pear.mat'.format(fold+1))
    X = data['net']
    X_valid = data['net_valid'] #vaild
    Y_valid = data['Y_valid']
    return X, X_valid, Y_valid

def mapStd(X,X_train,X_valid,X_test):
    [subjNum, n_roi0, n_roi1]=np.shape(X)
    X=np.reshape(X,[subjNum,n_roi0*n_roi1])
    preprocessor=prep.StandardScaler().fit(X)
    X_train = np.reshape(X_train, [-1, n_roi0 * n_roi1])
    X_valid = np.reshape(X_valid, [-1, n_roi0 * n_roi1])
    X_test = np.reshape(X_test, [-1, n_roi0 * n_roi1])
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    X_train = np.reshape(X_train, [-1, n_roi0, n_roi1])
    X_valid = np.reshape(X_valid, [-1, n_roi0, n_roi1])
    X_test = np.reshape(X_test, [-1, n_roi0, n_roi1])
    return X_train,X_valid,X_test

def mapStd2(X,X_train,X_valid,X_test):
    # Z-标准化，均值为0，标准差为1
    preprocessor=prep.StandardScaler().fit(X)
    X = preprocessor.transform(X)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    return X,X_train,X_valid,X_test

def mapMinmax(X,X_train,X_valid,X_test):
    preprocessor=prep.MinMaxScaler().fit(X)
    X = preprocessor.transform(X)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    return X,X_train,X_valid,X_test

def mapMinmax2(X,X_train,X_valid,X_test):
    # 最小-最大缩放，将数据缩放到[-1, 1]
    preprocessor=prep.MinMaxScaler().fit(X)
    X = 2*preprocessor.transform(X)-1
    X_train = 2*preprocessor.transform(X_train)-1
    X_valid = 2*preprocessor.transform(X_valid)-1
    X_test = 2*preprocessor.transform(X_test)-1
    return X,X_train,X_valid,X_test


def convert2onehot(X):
    X_tmp = np.zeros([np.size(X, 0), 2])
    X_tmp[np.where(X == 0)[0], 0] = 1
    X_tmp[np.where(X == 1)[0], 1] = 1
    return X_tmp

def AEmain(L2lamda, L1lamda):
    validCostlist = []
    validAcclist = []
    testAcclist = []
    testSenlist = []
    testSpelist = []

    for this_fold in range(fold):
        print("Fold {},".format(this_fold+1))
        X, X_train, X_valid, X_test, Y_train, Y_valid, Y_test, Y_train2, Y_valid2, Y_test2 = load_data(this_fold)
        X_train, X_valid, X_test = mapStd(X, X_train, X_valid, X_test)

        Y_train2 = convert2onehot(Y_train2)
        Y_valid2 = convert2onehot(Y_valid2)
        Y_test2 = convert2onehot(Y_test2)

        time_start = time.time()
        fcresnet = brainnetCNN(72, L2lamda)
        fcresnet.init()

        valid_cost ,valid_acc , test_acc, \
        test_sen ,test_spe = fcresnet.train(None, 'ROIs_real',
                                       X_train, X_valid, X_test,
                                       Y_train, Y_valid, Y_test,
                                       Y_train2, Y_valid2, Y_test2,
                                       epochs=10000,  # 1e4
                                       learn_rate=1e-4, #1e-4
                                       fold=this_fold+1)

        time_end = time.time()
        validCostlist.append(valid_cost)
        validAcclist.append(valid_acc)
        testAcclist.append(test_acc)
        testSenlist.append(test_sen)
        testSpelist.append(test_spe)

    print('valid cost: {}, valid acc: {}, test acc: {}, test sen: {}, test spe: {}'
          .format(np.mean(validCostlist),
                  np.mean(validAcclist),
                  np.mean(testAcclist),
                  np.mean(testSenlist),
                  np.mean(testSpelist)))



if __name__ == '__main__':
    for L2lamda in [1.5]:#[5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6]:  [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
        L1lamda = L2lamda / 10
        print('L2lamda: {}'.format(L2lamda))
        if os.path.exists('/home/ai/data/zhangyaqin/CNNzyq/log/'):
            shutil.rmtree('/home/ai/data/zhangyaqin/CNNzyq/log/')
        AEmain(L2lamda=L2lamda, L1lamda=L1lamda)
            
        







