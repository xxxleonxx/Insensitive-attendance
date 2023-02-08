# note: https://github.com/SthPhoenix/InsightFace-REST
from numpy.linalg import norm

import onnxruntime
import numpy as np
import cv2
from loguru import logger
import binascii


class FaceRecognition:
    def __init__(self):
        model_path = '/home/taiwu/Project/resources/models/arcface_r100_v1.onnx'

        # --- debug mode ---
        # self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])  # for test
        self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.outputs = [e.name for e in self.onnx_session.get_outputs()]
        self.prepare()

    def prepare(self, **kwargs):
        """模型初始化"""
        logger.info("Warming up ArcFace ONNX Runtime engine...")
        self.onnx_session.run(output_names=self.outputs,
                              input_feed={
                                  self.onnx_session.get_inputs()[0].name: [np.zeros((3, 112, 112), np.float32)]})

    @staticmethod
    def normalize(embedding):
        """特征数据格式化"""
        embedding_norm = norm(embedding)
        normed_embedding = embedding / embedding_norm
        return normed_embedding

    def get_face_features_normalization_by_image_array(self, image_array):
        """获取特征"""

        # --- check ---
        if image_array is None:
            return None

        # --- check ---
        if image_array.shape != (112, 112, 3):
            image_array = cv2.resize(image_array, (112, 112))

        if not isinstance(image_array, list):
            image_array = [image_array]

        for i, img in enumerate(image_array):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            image_array[i] = img.astype(np.float32)

        image_array = np.stack(image_array)
        net_out = self.onnx_session.run(self.outputs, {self.onnx_session.get_inputs()[0].name: image_array})
        return self.normalize(net_out[0][0])

    @staticmethod
    def image_bytes_to_image_array(image_bytes, mode='RGB'):
        """
        数据格式转换
        """
        _image = np.asarray(bytearray(image_bytes), dtype='uint8')
        _image = cv2.imdecode(_image, cv2.IMREAD_COLOR)
        return _image

    def get_face_features_normalization_by_image_bytes(self, image_bytes):
        """获取特征"""
        # --- bytes to array ---
        image_array = self.image_bytes_to_image_array(image_bytes)

        # --- check size --- todo 如果是4通道，考虑通过cvtColor转3通道
        if image_array.shape != (112, 112, 3):
            image_array = cv2.resize(image_array, (112, 112))

        if not isinstance(image_array, list):
            image_array = [image_array]

        for i, img in enumerate(image_array):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            image_array[i] = img.astype(np.float32)

        image_array = np.stack(image_array)
        net_out = self.onnx_session.run(self.outputs, {self.onnx_session.get_inputs()[0].name: image_array})
        return self.normalize(net_out[0][0])

    @staticmethod
    def compare_faces_by_normalization(input_normalization, specimen_normalization):
        """
        计算相似度（使用格式化数据）
        """
        _sim = (1.0 + np.dot(input_normalization, specimen_normalization)) / 2.0
        return _sim

    def search_face(self, face_features_normalization, face_dict):
        """
        寻找近似人脸（face_dict为一对一）
        """
        # --- get face ---
        best_face_uuid, best_face_dist = None, None
        for face_uuid, face_features in face_dict.items():

            dist = self.compare_faces_by_normalization(face_features_normalization, face_features)

            # --- check --- 低相似度，直接过滤 （一般低于71%就不是一个人了）
            if dist > 0.71 and not best_face_dist:
                best_face_dist = dist
                best_face_uuid = face_uuid

            if best_face_dist and dist > best_face_dist:
                best_face_dist = dist
                best_face_uuid = face_uuid
        return best_face_uuid, best_face_dist

    def search_face_v2(self, predict_features, face_features, filter_sim=0.72):
        """
        寻找近似人脸（face_dict为一对多）
        filter_dist: 相似度判定值
        """
        sim = self.compare_faces_by_normalization(predict_features, face_features)
        return sim

    @staticmethod
    def clip_image(image_array, left, top, width, height):
        """剪裁图像"""

        # --- 根据deepstream的左上点坐标，进行剪裁图像 ---
        return image_array[int(top):int(top + height), int(left):int(left + width)]

    @staticmethod
    def image_hex_to_image_array(hex_str_image, mode='RGB'):
        """
        数据格式转换
        """
        try:
            bytes_image = binascii.unhexlify(hex_str_image)
            _image = np.asarray(bytearray(bytes_image), dtype='uint8')
            _image = cv2.imdecode(_image, cv2.IMREAD_COLOR)
            return _image
        except Exception as exception:
            logger.exception(exception)
            return None


if __name__ == '__main__':
    # --- init ---
    agent = FaceRecognition()
    agent.prepare()
    # --- test ---
    # p0 = cv2.imread('./face.jpg')
    # f0 = agent.get_face_features_by_image_array(p0)
    # p3 = cv2.imread('./worker.jpg')
    # p3 = cv2.resize(p3, (112, 112))
    # f3 = agent.get_face_features_by_image_array(p3)
    # sim = agent.compare_faces(f0, f3)
    # print(f"Similarity: {sim}")

    # --- test ---
    import requests

    url = 'https://lwres.yzw.cn/worker-avatar/Original/2020/1013/96c419ca-dbf2-4bf7-a072-92fde861a2bc.jpg'
    response = requests.get(url, headers={'content-type': 'application/json'})
    _image_bytes = response.content
    f0 = agent.get_face_features_normalization_by_image_bytes(_image_bytes)
    print(f"f0: {type(f0)}")

    p1 = cv2.imread('/home/taiwu/Project/resources/models/3.jpeg')
    f1 = agent.get_face_features_normalization_by_image_array(p1)
    sim = agent.compare_faces_by_normalization(f0, f1)
    print(f"Similarity: {sim}")
