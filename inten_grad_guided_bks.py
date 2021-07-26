'''
Created on 24th June,2020

@author: yifan_zuo
'''
import tensorflow as tf
import numpy as np
#define sub-functions for creating variables
#ksize,w_shape,b_shape,strides are all list type, this is very important
def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

def max_pool_3x3(x,ratio):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, ratio, ratio, 1], padding='SAME')

def weight_variable(w_shape,out_ch=32,name=None):
    std_factor=w_shape[0]*w_shape[1]*out_ch
    std_dev=np.sqrt(1.8824/std_factor)
    initial_W = tf.truncated_normal(w_shape, stddev=std_dev)
    if name is None:
        return tf.Variable(initial_W,name="conv_weight",dtype=tf.float32)
    else:
        return tf.get_variable(name,dtype=tf.float32,initializer=initial_W)
    
def bias_variable(b_shape,name=None):
    initial_B = tf.constant(0.0, shape=b_shape)
    if name is None:
        return tf.Variable(initial_B,name="conv_bias",dtype=tf.float32)
    else:
        return tf.get_variable(name,dtype=tf.float32,initializer=initial_B)
    
def Prelu(input_tensor,name=None):
    initial_a=tf.constant(0.25, shape=[input_tensor.get_shape().as_list()[3]])
    if name is None:
        alphas=tf.Variable(initial_a,name="prelu_alpha",dtype=tf.float32)
    else:
        alphas=tf.get_variable(name,dtype=tf.float32,initializer=initial_a)
    pos = tf.nn.relu(input_tensor)
    neg = alphas * (input_tensor - abs(input_tensor)) * 0.5
    return pos+neg

#define batch normalization function
def batch_norm(x, n_out, phase_train,name=None):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    if name is None:
        beta = tf.Variable(tf.constant(0.0, shape=n_out),name="bn_beta",dtype=tf.float32)
        gamma = tf.Variable(tf.truncated_normal(n_out, 1.0, 0.02),name="bn_gamma",dtype=tf.float32)
    else:
        beta=tf.get_variable(name[0],dtype=tf.float32,initializer=tf.constant(0.0, shape=n_out))
        gamma =tf.get_variable(name[1],dtype=tf.float32,initializer=tf.truncated_normal(n_out, 1.0, 0.02))
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

#define sub-function for conv layer with prelu
def conv_Prelu_block(input_ten,w_shape,stride,skip_Prelu):
    W=weight_variable(w_shape,w_shape[3])
    conv_ten = tf.nn.conv2d(input_ten, W,stride, padding='SAME')
    B=bias_variable([w_shape[3]])
    if skip_Prelu:
        return conv_ten + B
    else:
        return Prelu(conv_ten + B)
        
#define sub-function for deconv layer with prelu
def deconv_Prelu_block(input_ten,w_shape,stride,skip_Prelu):
    input_shape=input_ten.get_shape().as_list()
    out_shape=tf.convert_to_tensor([input_shape[0],stride[1]*input_shape[1],stride[2]*input_shape[2],w_shape[2]])
    W=weight_variable(w_shape,w_shape[2])
    deconv_ten= tf.nn.conv2d_transpose(input_ten,W,out_shape,stride,padding="SAME")
    B=bias_variable([w_shape[2]])
    if skip_Prelu:
        return deconv_ten + B
    else:
        return Prelu(deconv_ten + B)
        
#from this on is the construction blocks of network
#define sub-functions for the intensity branch
            
def inten_FE(inten_feat, shape_list=[[7,7,1,64],[5,5,64,32]],stride=[1,1,1,1],skip_Prelu_list=[False,False]):
    for index in range(len(skip_Prelu_list)):
        inten_feat=conv_Prelu_block(inten_feat,shape_list[index],stride,skip_Prelu_list[index])
    return inten_feat

def inten_FDown(inten_feat, ratio, list_num, shape=[5,5,32,32],stride=[1,1,1,1],skip_Prelu=False):
    inten_feat_down=[inten_feat]
    for _ in range(list_num):
        inten_feat=conv_Prelu_block(inten_feat, shape, stride, skip_Prelu)
        inten_feat=max_pool_3x3(inten_feat,ratio)
        inten_feat_down.append(inten_feat)
    return inten_feat_down

def dep_FE(dep_feat,shape_list=[[5,5,1,64],[5,5,64,32]],stride=[1,1,1,1],skip_Prelu_list=[False,False]):
    for index in range(len(skip_Prelu_list)):
        dep_feat=conv_Prelu_block(dep_feat,shape_list[index],stride,skip_Prelu_list[index])
    return dep_feat

def dep_FUp(dep_feat,shape=[5,5,32,32],stride=[1,2,2,1],skip_Prelu=False):
    dep_feat_up=deconv_Prelu_block(dep_feat,shape,stride,skip_Prelu)
    return dep_feat_up

def inten_guided_fusion(inten_feat,dep_feat,shape_list=[[5,5,64,32],[5,5,32,32]],stride=[1,1,1,1],skip_Prelu_list=[False,True]):
    fusion_ten=tf.concat([inten_feat,dep_feat],3)
    for index in range(len(skip_Prelu_list)):
        fusion_ten=conv_Prelu_block(fusion_ten,shape_list[index],stride,skip_Prelu_list[index])
    return fusion_ten+dep_feat

def cross_dep_grad_fusion(dep_feat,grad_feat,shape_list=[[5,5,64,32],[5,5,32,32],[5,5,32,32]],stride=[1,1,1,1],skip_Prelu_list=[False,False,True]):
    grad_fusion=tf.concat([dep_feat,grad_feat],3)
    for index in range(len(skip_Prelu_list)):
        grad_fusion=conv_Prelu_block(grad_fusion,shape_list[index],stride,skip_Prelu_list[index])
    grad_fusion=grad_fusion+grad_feat
    dep_fusion=tf.concat([dep_feat,grad_fusion],3)
    for index in range(len(skip_Prelu_list)):
        dep_fusion=conv_Prelu_block(dep_fusion,shape_list[index],stride,skip_Prelu_list[index])
    dep_fusion=dep_fusion+dep_feat
    return dep_fusion,grad_fusion
    
def dep_recon(coar_dep,prev_feat,final_feat,shape_list=[[5,5,32,32],[5,5,32,32],[5,5,32,1]], stride=[1,1,1,1], skip_Prelu_list=[False,True,True]):
    for ind in range(len(skip_Prelu_list)-1):
        final_feat=conv_Prelu_block(final_feat, shape_list[ind], stride, skip_Prelu_list[ind])
    final_feat=final_feat+prev_feat
    final_feat=conv_Prelu_block(final_feat, shape_list[-1], stride, skip_Prelu_list[-1])
    return final_feat+coar_dep

#define function for reading data from hdf5
def reading_data(train_file,pat_ind_range,HR_batch_dims,LR_batch_dims):
    inten_bat=train_file['inten_patch'][pat_ind_range,:,:]
    inten_bat=inten_bat.reshape(HR_batch_dims)
    gth_dep_bat=train_file['depth_patch'][pat_ind_range,:,:]
    gth_dep_bat=gth_dep_bat.reshape(HR_batch_dims)
    LR_dep_bat=train_file['LR_noisy_depth_std5'][pat_ind_range,:,:]
    #LR_dep_bat=train_file['LR_depth_patch'][pat_ind_range,:,:]
    LR_dep_bat=LR_dep_bat.reshape(LR_batch_dims)
    return inten_bat,gth_dep_bat,LR_dep_bat
