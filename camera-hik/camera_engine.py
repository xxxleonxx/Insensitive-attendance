# from util.db_operate import DB
from loguru import logger
from camera_decode import CameraDecode
import time
from concurrent.futures import ThreadPoolExecutor

import threading


class CameraEngine:
    """
    """
    # database = DB()
    camera_info_list = []  # 相机信息列表，当手动添加设备后，会调用DB.camera_get_all方法给其刷新赋值

    '''用于存放正在运行的摄像头信息。例如{'192.168.0.1':'用户名': 'xxx', '密码': 'xxx', '方向':'xxx', '设备组':'xxx'},
    便于程序运行过程中查询摄像头信息，不需要再去查询数据库，减少耗时'''
    run_camera_info = {}

    '''用于存放从摄像头获取到的图片'''
    frame_dict = {}

    @classmethod
    def update_camera_info(cls):
        """
        更新摄像头信息
        """
        # cls.camera_info_list = cls.database.camera_get_all()
        cls.camera_info_list = [['192.168.0.181', 'admin', 'DEVdev123', 0, '通道三']]
        '''从列表中获取相机ip，用户名，密码，方向，设备组信息'''
        for camera_info in cls.camera_info_list:
            cls.run_camera_info[camera_info[0]] = {'用户名': camera_info[1],
                                                   '密码': camera_info[2],
                                                   '方向': camera_info[3],
                                                   '设备组': camera_info[4]}

    @classmethod
    def listen_hik(cls, camera_info_list):
        """
        监听海康相机，获取人脸抓拍图片
        camera_info_list：相机信息列表，[ip, 用户名, 密码, 方向, 设备组]
        """
        ThreadPoolExecutor()
        task_info = dict()  # {<ipv4>: (<task>, <hik>)}
        while True:
            # --- check --- 检查是否有突然掉线的相机，如果有则进行重置
            for ipv4 in task_info.keys():
                task, hik = task_info.get(ipv4)
                if not hik:
                    continue

                # --- reset ---
                now_at = time.time()
                if hik.last_receive_at and now_at - hik.last_receive_at > 60:
                    task_info[ipv4] = None, None

            # --- check --- 检查不存活的相机，进行尝试
            for item in camera_info_list:
                # --- check ---
                camera_ipv4, camera_user, camera_passwd = item[:3]
                if camera_ipv4 not in task_info:
                    task_info[camera_ipv4] = None, None

                # -- check ---
                if task_info.get(camera_ipv4)[1] is not None:
                    continue

                try:
                    hik = CameraDecode()
                    session = hik.connect_camera(camera_ipv4, camera_user, camera_passwd)
                    t = threading.Thread(target=hik.read1, args=(cls.frame_dict,))
                    t.daemon = True
                    t.start()
                    task_info[camera_ipv4] = t, hik

                    cls.run_camera_info[camera_ipv4]['task'] = t
                    cls.run_camera_info[camera_ipv4]['session'] = session
                    # task = pool.submit(hik.read1, cls.frame_dict)

                    # task_info[camera_ipv4] = task, hik
                    logger.debug('{camera}is connected.', camera=camera_ipv4)

                except Exception as exception:
                    logger.debug(exception)
                    logger.debug('wait 1 minutes try again!')
                    time.sleep(60)
                    continue

    @classmethod
    def run(cls):
        """
        建立相机人脸抓拍数据获取连接
        """
        # while True:
        #     try:
        '''从数据库获取相机信息列表'''
        cls.update_camera_info()
        logger.info(' 获取相机信息列表')

        cls.listen_hik(cls.camera_info_list)


if __name__ == '__main__':
    CameraEngine.run()
