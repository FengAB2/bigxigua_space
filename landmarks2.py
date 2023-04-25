import numpy as np
import cv2
from FaceBoxes.FaceBoxes import main
from synergy3DMM import SynergyNet
import matplotlib.pyplot as plt
import torch
def get_landmarks(model,image):

  # get landmark [[y, x, z], 68 (points)], mesh [[y, x, z], 53215 (points)], and face pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])
  lmk3d, mesh, pose = model.get_all_outputs(image)
  lmk3d= np.array(lmk3d)
  mesh= np.array(mesh)
  pose= np.array(pose)
  # print('lmk3d.shape:',lmk3d.shape)
  # print('mesh.shape:',mesh.shape)
  # print('pose.shape:',pose.shape)

  landmarks = lmk3d[0] # 提取每个面部特征点的三维坐标，并转置以适应下面的计算
  landmarks= landmarks.reshape(3,-1)
  # print('landmarks.shape',landmarks.shape)

  R,_ = cv2.Rodrigues(pose[0][0]) # 从旋转向量获取旋转矩阵
  t = pose[0][1]
  # print('t.shape',t.shape)

  # print('R.shape',R.shape)

  # print('mesh[0].shape',mesh[0].shape)



  rotated_lmk= np.dot(R,landmarks)
  rotated_lmk+=t[:,np.newaxis]
  # print('rotated_lmk.shape',rotated_lmk.shape)
  lmk_center= np.mean(rotated_lmk,axis=1)
  translated_lmk= rotated_lmk- lmk_center.reshape((-1,1))
  translated_3_lmk=translated_lmk[np.newaxis,:,:]
  translated_3_lmk= translated_3_lmk.reshape(1,68,3)

  


  rotated_mesh = np.dot(R, mesh[0]) # 旋转和平移每个顶点
  rotated_mesh+= t[:,np.newaxis]
  # print('rotated_mesh.shpae',rotated_mesh.shape)

  # 将模型中心点移动到原点
  center = np.mean(rotated_mesh, axis=1)
  # print('center.shape',center.shape)
  translated_mesh = rotated_mesh - center.reshape((-1, 1))
  
  
  # return translated_lmk,translated_mesh,translated_3_lmk

def main():    
  model = SynergyNet()
  image = cv2.imread('/content/SPACE/SynergyNet/img/sample_3.jpg')


  x= get_landmarks(model,image)
  xy_translated_mesh = x[1][[1,0], :]
  # print('yx_translated_mesh.shape',xy_translated_mesh.shape)
  # 绘制图像
  fig, ax = plt.subplots()
  ax.scatter(xy_translated_mesh[0, :], xy_translated_mesh[1, :], s=0.1)
  ax.set_aspect('equal')
  plt.show()
 
  xy_translated_lmk = x[0][[1,0], :]
  
  # print('yx_translated_mesh.shape',xy_translated_lmk.shape)
  # 绘制图像
  fig, ax = plt.subplots()
  ax.scatter(xy_translated_lmk[0, :], xy_translated_lmk[1, :], s=10)
  ax.set_aspect('equal')
  plt.show()
  # print('translated_mesh.shape',x[1].shape)
  # print('translated_lmk.shpae',x[0].shape)
  # print('translated_3_lmk.shape',x[2].shape)
if __name__=='__main__':
  main()


