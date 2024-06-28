from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys

import cv2
import os
from sys import platform
import argparse

from PyQt5 import QtGui

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure


class MyApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('widget.ui', self)
        self.setWindowTitle('姿态估计演示')  # 设置窗口标题

        # 将按钮的点击信号连接到自定义的槽函数
        self.choose.clicked.connect(self.open_file_dialog)
        self.download_2d_image.clicked.connect(self.save_2d)

    def save_2d(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "Images (*.png *.jpg *.bmp)", options=options)
        cv2.imwrite(fileName, self.image_data_2d)
        return

     # 自定义的槽函数，处理按钮点击事件
    def open_file_dialog(self):
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Images (*.png *.jpg *.bmp)")
        print(file_path)

        # 输入模型
        try:
            # Import Openpose (Windows/Ubuntu/OSX)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            try:
                # Windows Import
                if platform == "win32":
                    # Change these variables to point to the correct folder (Release/x64 etc.)
                    # sys.path.append(dir_path + '/../../python/openpose/Release');
                    sys.path.append('D:/GraduationProject/openpose-prosperity/openpose-prosperity/build_CPU/python/openpose/Release')
                    # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                    os.environ['PATH']  = os.environ['PATH'] + ';' + 'D:/GraduationProject/openpose-prosperity/openpose-prosperity/build_CPU/x64/Release;' +  'D:/GraduationProject/penpose-prosperity/openpose-prosperity/build_CPU/bin;'
                    import pyopenpose as op
                else:
                    # Change these variables to point to the correct folder (Release/x64 etc.)
                    sys.path.append('../../python')
                    # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                    # sys.path.append('/usr/local/python')
                    from openpose import pyopenpose as op
            except ImportError as e:
                print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
                raise e

            # Flags
            parser = argparse.ArgumentParser()
            parser.add_argument("--image_path", default=file_path, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
            args = parser.parse_known_args()

            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            params = dict()
            params["model_folder"] = "../../../models/"

            # Add others in path?
            for i in range(0, len(args[1])):
                curr_item = args[1][i]
                if i != len(args[1])-1: next_item = args[1][i+1]
                else: next_item = "1"
                if "--" in curr_item and "--" in next_item:
                    key = curr_item.replace('-','')
                    if key not in params:  params[key] = "1"
                elif "--" in curr_item and "--" not in next_item:
                    key = curr_item.replace('-','')
                    if key not in params: params[key] = next_item

            # Construct it from system arguments
            # op.init_argv(args[1])
            # oppython = op.OpenposePython()

            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()

            # Process Image
            datum = op.Datum()
            imageToProcess = cv2.imread(args[0].image_path)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Display Image
            print("Body keypoints: \n" + str(datum.poseKeypoints))

            # 显示原图像
            image_data = cv2.imread(file_path)
            size = (int(self.image.width()),int(self.image.height()))  
            shrink = cv2.resize(image_data, size, interpolation=cv2.INTER_AREA)  
            shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)                                          
            self.QtImg = QtGui.QImage(shrink.data, 
                                    shrink.shape[1], 
                                    shrink.shape[0],
                                    shrink.shape[1]*3,
                                    QtGui.QImage.Format_RGB888)

                                                                
            self.image.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

            # 显示2D估计图形
            image_data_2d = datum.cvOutputData
            self.image_data_2d=image_data_2d
            size = (int(self.image_2d.width()),int(self.image_2d.height()))  
            shrink2 = cv2.resize(image_data_2d, size, interpolation=cv2.INTER_AREA)  
            #cv2.imshow('img', shrink)
            shrink2 = cv2.cvtColor(shrink2, cv2.COLOR_BGR2RGB)                                          
            self.QtImg = QtGui.QImage(shrink2.data, 
                                    shrink2.shape[1], 
                                    shrink2.shape[0],
                                    shrink2.shape[1]*3,
                                    QtGui.QImage.Format_RGB888)

                                                                
            self.image_2d.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            # 保存图像到文件中
            # cv2.imwrite("2.jpg", image_data)
            # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            # cv2.waitKey(0)

            # 显示3d估计图形
            pose_3d=datum.poseKeypoints[0]
            pose_3d_17=pose_3d[0:17]
            pose_3d_17[15,:]=(pose_3d[15,:]+pose_3d[16,:])/2
            pose_3d_17[16,:]=(pose_3d[1,:]+pose_3d[8,:])/2


            keypoint_3d=pose_3d_17
            keypoint_3d[:,0]=keypoint_3d[:,0]/520
            keypoint_3d[:,1]=keypoint_3d[:,1]/640

            # 绘制 3D 散点图
            # 提取 x、y、z 轴数据
            x = keypoint_3d[:, 0]
            y = keypoint_3d[:, 1]
            z = keypoint_3d[:, 2]

            # 创建 3D 图形对象
            # Create a QVBoxLayout as layout
            layout = QVBoxLayout(self.image_3d)

            # Create a FigureCanvas as plot area
            fig = Figure()
            canvas = FigureCanvas(fig)

            # Add the FigureCanvas to the layout
            layout.addWidget(canvas)

            ax = fig.add_subplot(111, projection='3d')

            c='#5588A3'
            ax.scatter(x, y, z, c=c, marker='o')

            edges = np.array([[0,1],[2,1],[1,5],
                            [2,3],[3,4],[5,6],[6,7],
                            [1,16],[16,8],
                            [8,9],[9,10],[10,11],
                            [8,12],[12,13],[13,14]])

            for edge in edges:
                point1 = keypoint_3d[edge[0]]
                point2 = keypoint_3d[edge[1]]
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color=c)

            # 计算数据的范围
            x_range = np.ptp(x)
            y_range = np.ptp(y)
            z_range = np.ptp(z)
            max_range = max(x_range, y_range, z_range)

            # 设置每个坐标轴的范围
            mid_x = np.mean(x)
            mid_y = np.mean(y)
            mid_z = np.mean(z)
            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
            ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)



            # 显示网格
            ax.grid(True)
            # 隐藏坐标轴的刻度值
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax.tick_params(axis='z', which='both', bottom=False, top=False, labelbottom=False, labelleft=False)

            # Show the plot
            canvas.draw()


        except Exception as e:
            print(e)
            sys.exit(-1)
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = MyApp()
    myapp.show()
    sys.exit(app.exec_())

