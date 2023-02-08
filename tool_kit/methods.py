import base64
import numpy as np
import cv2


def image_decode(b64img):
    """
    解码base64图片
    Args:
        b64img: base64编码后的图片
    Returns:
    """
    img_str = base64.b64decode(b64img)  # base64解码为二进制
    img_buf = np.frombuffer(img_str, np.uint8)  # 二进制转为buffer
    img_mat = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)  # 将buffer转为opencv图像格式
    return img_mat


def img_bytes2array(image_bytes):
    _image = np.frombuffer(image_bytes, dtype=np.uint8)
    _image = cv2.imdecode(_image, cv2.IMREAD_COLOR)
    return _image
