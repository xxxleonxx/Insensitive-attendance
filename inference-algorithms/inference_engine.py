import binascii
# from camera-hik.camera_engine import CameraEngine
import importlib
import time
import uuid

import cv2
import numpy as np
import schedule
from loguru import logger

from face_detect import FaceDetection
from face_recognition import FaceRecognition

camera = importlib.import_module(f'amera-hik.camera_engine')
mdb = importlib.import_module(f"clients.db_mongo").Client(host='mongodb://localhost', port=27017, database='vms')


def image_hex2array(hex_image):
    """16进制字符串转BGR图片"""
    bytes_image = binascii.unhexlify(hex_image)
    img_arr = np.asarray(bytearray(bytes_image), dtype='uint8')
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return image


class InferenceEngine:
    FaceDetector = FaceDetection()
    FaceRecognition = FaceRecognition()

    recognition_result = list()  # 用于存放识别正确的人员结果

    # database = DB()

    @classmethod
    def extract_feature(cls, image):
        """
        提取人脸图片特征
        """
        '''人脸检测、人脸对齐'''
        detect_results = cls.FaceDetector.inference_with_image_array(image)

        if detect_results:
            # --- 歪脸矫正 ---
            face_image = detect_results[0].get('align_face')

            # --- 提取特征 ---
            feature = cls.FaceRecognition.get_face_features_normalization_by_image_array(face_image)
            return feature, detect_results[0].get('raw_face')
        return None, None

    @classmethod
    def recognition(cls, image, face_encodings):
        """
        人脸识别核心逻辑：检测、对齐、特征提取、特征比对
        """
        '''人脸检测、人脸对齐'''

        features, face_img = cls.extract_feature(image)

        if features is not None:
            face_distance = np.linalg.norm(face_encodings - features, axis=1)  # 计算人脸特征向量的距离
            index = np.argmin(face_distance)  # 获取最小距离对应的索引
            """计算相似度"""
            sim = cls.FaceRecognition.search_face_v2(features, face_encodings[index])  # face_encodings 人脸库特征向量集
            unique_dict = {'name': 'face_similarity'}
            if sim > (mdb.get_one('GlobalVariable', unique_dict) / 100.):
                face_name = mdb.get_one_by_id('Face', face_uuid).get('face_name')
                job_number = mdb.get_one_by_id('Face', face_uuid).get('job_number')
                department = cls.database.department_list[index]
                person_type = cls.database.person_type_list[index]
                # print(job_num, name, department, person_type)
                return True, {'工号': job_num, '姓名': name, '部门': department, '类型': person_type}

        return False, {}

    @classmethod
    def identify_logic(cls):
        while True:
            try:
                if len(list(camera.CameraEngine.frame_dict.values())) == 0:
                    time.sleep(0.05)
                    continue
                if len(cls.database.feature_list) == 0:
                    time.sleep(0.1)
                    continue
                face_encodings = np.asarray(list(cls.database.feature_list))
                for ipv4 in list(camera.CameraEngine.frame_dict.keys()):
                    for time_stack in list(camera.CameraEngine.frame_dict[ipv4].keys()):
                        hex_image = camera.CameraEngine.frame_dict[ipv4][time_stack]['hex_image']
                        image_array = image_hex2array(hex_image)
                        success, result_dict = cls.recognition(image_array, face_encodings)
                        file_path = f"./temp/{uuid.uuid4()}.jpg"
                        cv2.imwrite(file_path, image_array)
                        if success:
                            logger.info('识别成功 {ipv4} {result_dict}.', ipv4=ipv4, result_dict=result_dict)
                            camera_direction = camera.CameraEngine.run_camera_info[ipv4]['方向']
                            device_group = camera.CameraEngine.run_camera_info[ipv4]['设备组']
                            result_dict['方向'] = camera_direction
                            result_dict['设备组'] = device_group
                            cls.recognition_result.append(result_dict)  # 向识别结果列表中添加结果信息
                            cls.database.add_to_client(result_dict['工号'], result_dict['姓名'], device_group,
                                                       file_path)
                        else:
                            device_group = camera.CameraEngine.run_camera_info[ipv4]['设备组']
                            # queue.put({"state": 1, "group": device_group, "image": image_array})
                            cls.database.add_to_client('999999999999999999', "陌生人", device_group, file_path)
                            logger.info('陌生人员,请登记.')
                        '''清理摄像头ip列表中的已检测完毕图片'''
                        camera.CameraEngine.frame_dict[ipv4].pop(time_stack)

            except Exception as exception:
                logger.error(exception)
                break

    @classmethod
    def run(cls):
        """启动人脸检测、识别推断引擎"""
        while True:
            try:
                cls.identify_logic()

            except Exception as exception:
                logger.error(exception)
                logger.info(' wait 1 minutes try again!')
                time.sleep(60)
                continue

    @classmethod
    def reload_face_lib(cls):  # todo 用apscheduler替换schedule
        schedule.every(120).seconds.do(cls.database.load_info)
        while True:
            schedule.run_pending()
            time.sleep(3)
