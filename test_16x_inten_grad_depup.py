'''
Created on 17June,2020

@author: yifan_zuo
'''
from PIL import Image
import numpy as np
import tensorflow as tf
import inten_grad_guided_bks as igg

up_factor=16
data_name="dolls"
#read test HR inten img
#color_art = Image.open("/media/yifan/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/color_"+data_name+"_croped.png")#####for noise-free midd2006
color_art = Image.open("/media/yifan/Data/test_data/noisy/table2/_gth/"+data_name+"_color.png")####mid 2006
#color_art = Image.open("/media/yifan/Data/test_data/noisy/iccv15_test_imgs/gth/"+data_name+"/color_crop.png")####mid 2001
inten_art =color_art.convert("L")
np_inten=np.asarray(inten_art)
val_inten=np_inten.astype(np.float32)/255.0

#read test HR dep img
#gth_dep_art=Image.open("/media/yifan/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/"+data_name+"_croped.png")#####for noise-free midd2006
gth_dep_art=Image.open("/media/yifan/Data/test_data/noisy/table2/_gth/"+data_name+"_big.png")###mid 2006
#gth_dep_art=Image.open("/media/yifan/Data/test_data/noisy/iccv15_test_imgs/gth/"+data_name+"/depth_crop.png")###mid 2001
np_gth_dep=np.asarray(gth_dep_art)
val_gth_dep=np_gth_dep.astype(np.float32)/255.0

#read test LR dep img
#LR_dep_art=Image.open("/media/yifan/Data/test_data/noise_free/middlebury_bicubic_LR_test_pairs/"+data_name+"/"+data_name+"16x_bicubic.png")#####for noise-free midd2006
LR_dep_art=Image.open("/media/yifan/Data/test_data/noisy/table2/_input/"+data_name+"_big/depth_4_n.png")###mid 2006
#LR_dep_art=Image.open("/media/yifan/Data/test_data/noisy/iccv15_test_imgs/input/"+data_name+"/16_x_dep.png")###mid 2001
np_LR_dep=np.asarray(LR_dep_art)
val_LR_dep=np_LR_dep.astype(np.float32)/255.0

height=val_inten.shape[0]
width=val_inten.shape[1]
LR_height=height/up_factor
LR_width=width/up_factor
val_inten=val_inten.reshape((1,height,width,1))
val_gth_dep=val_gth_dep.reshape(1,height,width,1)
val_LR_dep=val_LR_dep.reshape(1,LR_height,LR_width,1)

#setting input size and training data addr
HR_patch_size=[height,width]
HR_batch_dims=(1,height,width,1)
LR_batch_dims=(1,LR_height,LR_width,1)

#setting input placeholders
HR_depth_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
HR_inten_batch_input=tf.placeholder(tf.float32,HR_batch_dims)
LR_depth_batch_input=tf.placeholder(tf.float32,LR_batch_dims)
coar_inter_dep_batch=tf.image.resize_images(LR_depth_batch_input,tf.constant(HR_patch_size,dtype=tf.int32),tf.image.ResizeMethod.BICUBIC)
LR_depgrad_batch_input=tf.norm(tf.image.sobel_edges(LR_depth_batch_input),2,4)
#HR_depgrad_batch_input=tf.norm(tf.image.sobel_edges(HR_depth_batch_input),2,4)

#gen_network construction
inten_feat=igg.inten_FE(HR_inten_batch_input)
dep_feat=igg.dep_FE(LR_depth_batch_input)
grad_feat=igg.dep_FE(LR_depgrad_batch_input)
inten_feat_down=igg.inten_FDown(inten_feat,2,3)
for ind in [3,2,1,0]:
    dep_feat=igg.dep_FUp(dep_feat)
    grad_feat=igg.dep_FUp(grad_feat)
    if ind:
        dep_feat=igg.inten_guided_fusion(inten_feat_down[ind],dep_feat)
        grad_feat=igg.inten_guided_fusion(inten_feat_down[ind],grad_feat)
        dep_feat,grad_feat=igg.cross_dep_grad_fusion(dep_feat,grad_feat)
    else:
        final_depfeat=igg.inten_guided_fusion(inten_feat_down[0],dep_feat)
        final_gradfeat=igg.inten_guided_fusion(inten_feat_down[0],grad_feat)
        final_depfeat,final_gradfeat=igg.cross_dep_grad_fusion(final_depfeat,final_gradfeat)      
HR_gen_dep=igg.dep_recon(coar_inter_dep_batch,dep_feat,final_depfeat)
#HR_grad_dep=igg.conv_Prelu_block(final_gradfeat,[1,1,32,1],[1,1,1,1],True)

#define loss for gen
saver_full=tf.train.Saver()

#begin comp_gen testing
with tf.Session() as sess:
    model_path="/media/yifan/Data/trained_models/inten_grad_guided_depup/noisy/16x/second_train/full_models3/16x_ny_igg_model.ckpt-0"
    saver_full.restore(sess, model_path)
    ten_fets=sess.run(HR_gen_dep,feed_dict={HR_inten_batch_input:val_inten,HR_depth_batch_input:val_gth_dep,LR_depth_batch_input:val_LR_dep})
    final_array=ten_fets*255.0+0.5
    final_array[final_array>255]=255.0
    final_array[final_array<0]=0.0
    final_array=final_array.astype(np.uint8).reshape((height,width))
    result_img=Image.fromarray(final_array)
    result_img.save("/media/yifan/Data/trained_models/inten_grad_guided_depup/noisy/16x/second_train/results/"+data_name+"16x_result.png")
    ######################computing rmse
    final_array=final_array.astype(np.double)
    np_gth_dep=np_gth_dep.astype(np.double)
    #print(np.sqrt(((final_array-np_gth_dep)**2).mean()))
    #print((np.absolute(final_array-np_gth_dep)).mean())
    print("below are evaluated on 1080*1320")
    print(np.sqrt(((final_array[0:1080,0:1320]-np_gth_dep[0:1080,0:1320])**2).mean()))
    print((np.absolute(final_array[0:1080,0:1320]-np_gth_dep[0:1080,0:1320])).mean())
    #######################
