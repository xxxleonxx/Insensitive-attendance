import time
import uuid
import sys
import binascii
import threading
import numpy as np
import cv2
import schedule
import datetime

from loguru import logger
# from apscheduler.schedulers.blocking import BlockingScheduler
sys.path.append('..')
from camera_hik.camera_engine import CameraEngine
import db_client.db_mongo
from inference_algorithms.face_detect import FaceDetection
from inference_algorithms.face_recognition import FaceRecognition


mdb = db_client.db_mongo.MongoMethod(database='vms', host='127.0.0.1', port=27017)



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
        logger.debug('人脸检测开始')
        if features is not None:
            # logger.debug(face_encodings)
            face_distance = np.linalg.norm(face_encodings - features, axis=1)  # 计算人脸特征向量的距离
            logger.debug(face_distance)
            index = np.argmin(face_distance)  # 获取最小距离对应的索引
            """计算相似度"""
            sim = cls.FaceRecognition.search_face_v2(features, face_encodings[index])  # face_encodings 人脸库特征向量集
            logger.debug(sim)
            unique_dict = {'name': 'face_similarity'}
            item = mdb.get_one('GlobalVariable', unique_dict)
            data = item.get('args', {})
            face_dist = data.get('known_face_filter_dist')
            if sim > face_dist:
                job_num = mdb.job_num_list[index]
                name = mdb.name_list[index]
                department = mdb.department_list[index]
                person_type = mdb.person_type_list[index]
                # print(job_num, name, department, person_type)
                return True, {'工号': job_num, '姓名': name, '部门': department, '类型': person_type}

        return False, {}

    @classmethod
    def identify_logic(cls):
        while True:
            try:

                if len(list(CameraEngine.frame_dict.values())) == 0:

                    continue

                # if len(mdb.feature_list) == 0:
                #     time.sleep(0.1)
                #     logger.info(2)
                #     continue

                face_encodings = np.asarray(list(mdb.feature_list))

                for ipv4 in list(CameraEngine.frame_dict.keys()):
                    for time_stack in list(CameraEngine.frame_dict[ipv4].keys()):
                        hex_image = CameraEngine.frame_dict[ipv4][time_stack]['hex_image']
                        image_array = image_hex2array(hex_image)
                        success, result_dict = cls.recognition(image_array, face_encodings)
                        save_at = datetime.datetime.now().strftime('%Y-%m%d-%H%M%S-%f')
                        file_path = f"/home/taiwu/Project/Data_Storage_directory/imge_data/{save_at}.jpg"
                        cv2.imwrite(file_path, image_array)
                        target = dict()
                        if success:
                            logger.info('识别成功 {ipv4} {result_dict}.', ipv4=ipv4, result_dict=result_dict)
                            camera_direction = CameraEngine.run_camera_info[ipv4]['方向']
                            device_group = CameraEngine.run_camera_info[ipv4]['设备组']

                            target = {'姓名':result_dict.get('姓名'), '工号':result_dict.get('工号'),'方向':camera_direction,'设备组':device_group}
                            cls.recognition_result.append(result_dict)  # 向识别结果列表中添加结果信息
                            mdb.add('record_table', target)
                        else:
                            device_group = CameraEngine.run_camera_info[ipv4]['设备组']
                            result_dict['设备组'] = device_group
                            target = {'姓名':'陌生人', '工号':'11111111','设备组':device_group}
                            mdb.add('record_table', target)
                            logger.info('陌生人员,请登记.')
                        '''清理摄像头ip列表中的已检测完毕图片'''
                        CameraEngine.frame_dict[ipv4].pop(time_stack)

            except Exception as exception:
                logger.exception(exception)
                break

    @classmethod
    def run(cls):
        """启动人脸检测、识别推断引擎"""
        cls.FaceDetector.inference_with_image_array(np.zeros((400, 400, 3), dtype=np.uint8))
        logger.info("初始化人脸检测器成功")
        cls.FaceRecognition.get_face_features_normalization_by_image_array(np.zeros((112, 112, 3), dtype=np.uint8))
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
        logger.info('同步启动')
        schedule.every(120).seconds.do(mdb.load_info)
        while True:
            schedule.run_pending()
            time.sleep(3)


if __name__ == '__main__':
    p1 = threading.Thread(target=CameraEngine.run)
    p2 = threading.Thread(target=InferenceEngine.run)
    p1.start()
    p2.start()
