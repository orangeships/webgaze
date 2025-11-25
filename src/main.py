import os
import cv2
import time
import numpy as np
import os
from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel

def main(dir):

    try:                        
        output_directory = os.path.join(dir, "results")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        start_model_load_time = time.time()
        model = EyeModel(dir)
        total_model_load_time = time.time() - start_model_load_time
        print(f"Total time to load model: {1000*total_model_load_time:.1f}ms")

        homtrans = HomTransform(dir)
        """ for higher resolution (max available: 1920x1080) """
        cap=cv2.VideoCapture(1, cv2.CAP_DSHOW)
        # cap.set(cv2.CAP_PROP_SETTINGS, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        """ 用户选择：加载历史校准数据或进行新校准 """
        print("\n" + "="*50)
        print("请选择操作：")
        print("1. 加载历史校准数据（跳过校准）")
        print("2. 进行新校准")
        print("="*50)
        
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            # 尝试加载历史校准数据
            calibration_file = os.path.join(dir, "results", "calibration_results.json")
            if os.path.exists(calibration_file):
                print(f"正在加载历史校准数据: {calibration_file}")
                if homtrans.load_calibration_results(calibration_file):
                    print("历史校准数据加载成功！")
                    STransG = homtrans.STransG
                else:
                    print("历史校准数据加载失败，将进行新校准")
                    STransG = homtrans.calibrate(model, cap, sfm=True)
            else:
                print(f"历史校准文件不存在: {calibration_file}")
                print("将进行新校准")
                STransG = homtrans.calibrate(model, cap, sfm=True)
        else:
            # 进行新校准
            print("开始新校准...")
            STransG = homtrans.calibrate(model, cap, sfm=True)

        print("============================")
        print(f"STransG\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.2f}'})}")

        homtrans.RunGazeOnScreen(model, cap, sfm=True)

        # gocv.PlotPupils(gray_image, prediction, morphedMask, falseColor, centroid)

    except Exception as e:
        print(f"Something wrong when running EyeModel: {e}")

if __name__ == '__main__':
    # 使用项目根目录作为基础路径（模型文件在根目录的intel文件夹中）
    # 获取项目根目录（src的父目录）
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        # 如果在src目录下运行，则使用父目录作为项目根目录
        dir = os.path.dirname(current_dir)
    else:
        # 如果在项目根目录下运行，则使用当前目录
        dir = current_dir
    main(dir)


