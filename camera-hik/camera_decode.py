# update: 2022-6-24
"""
海康人脸摄像机接口
"""
from loguru import logger
import requests
import time


class CameraDecode(object):
    def __init__(self):
        self.camera_ipv4 = None     # 相机ip
        self.camera_user = None     # 相机用户名
        self.camera_pass = None     # 相机密码
        self.client = None          # 用于向相机发送请求获取人脸照片的客户端
        self.last_receive_at = None     # 上次收到返回图片的时间

    def connect_camera(self, camera_ipv4, camera_user, camera_pass):
        """
        连接摄像机
        """
        self.camera_ipv4 = camera_ipv4
        self.camera_user = camera_user
        self.camera_pass = camera_pass
        session = requests.session()
        request_url = f'http://{self.camera_ipv4}:80/ISAPI/Event/notification/alertStream'  # 设置认证信息
        auth = requests.auth.HTTPDigestAuth(self.camera_user, self.camera_pass)  # 发送请求，获取响应
        self.client = session.get(request_url, auth=auth, verify=False, stream=True)
        return session

    def throw(self, backdata, image_hex):
        """
        回传数据
        """
        if backdata is None:
            return
        if self.camera_ipv4 not in backdata:
            backdata[self.camera_ipv4] = dict()
        object_id = str(time.time())
        backdata[self.camera_ipv4][object_id] = dict()
        backdata[self.camera_ipv4][object_id]['tracking_is'] = False
        backdata[self.camera_ipv4][object_id]['hex_image'] = image_hex

    def read1(self, backdata=None):
        """读取海康相机抓拍到的图片数据"""
        # --- define ---
        image_hex = str()
        start_is = False
        print_is = False
        hex_start = ''
        hex_end = ''

        while True:

            # --- get ---
            line = self.client.raw.read(1)
            line = line.hex()

            # --- fill ---
            now_at = time.time()
            if not self.last_receive_at:
                self.last_receive_at = now_at
            if line:
                self.last_receive_at = now_at

            # --- fill ---
            hex_start += line

            # --- check ---
            if len(hex_start) > 8:
                hex_start = hex_start[-8:]

            # --- check ---
            if '0d0affd8' == hex_start:
                start_is = True
                image_hex = 'ff'
                logger.debug(' 图片头 0d0affd8')

            # --- fill ---
            if start_is:
                image_hex += line

            # --- check ---
            if start_is:

                hex_end += line

                if len(hex_end) > 8:
                    hex_end = hex_end[-8:]

                if 'ffd90d0a' == hex_end:
                    print_is = True
                    image_hex = image_hex[:-4]
                    logger.debug(' 图片尾 ffd90d0a.')

            # --- fill ---
            if print_is:
                self.throw(backdata, image_hex)
                image_hex = str()
                hex_start = str()
                hex_end = str()
                start_is = False
                print_is = False


if __name__ == '__main__':
    # --- init ---
    middledata = {}
    agent = CameraDecode()
    agent.connect_camera(camera_ipv4='10.8.21.204', camera_user='admin', camera_pass='a1234567')
    agent.read1({})
