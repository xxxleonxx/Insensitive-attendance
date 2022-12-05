"""
websocket发数据
"""
import asyncio
import base64
import importlib
import json
import os
import threading

import cv2
import numpy as np
from loguru import logger


class LineManage(object):
    """"""
    mdb = importlib.import_module(f"clients.db_mongo").Client(host='mongodb://localhost', port=27017, database='vms')
    line_dict = {}  # {<line_id>: <ws>} | line_id: websocket连接id | ws: websocket链接对象

    @classmethod
    def run_forever(cls):
        """
        调用协程方法
        """
        tasks = [cls.check_send()]
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.wait(tasks))

    @classmethod
    def run_background(cls, is_back_run=True):
        """
        后台运行
        """
        t1 = threading.Thread(target=cls.run_forever)
        t1.start()

    @classmethod
    async def check_send(cls):

        # --- define ---
        last_send_id = str()

        while True:

            try:
                # --- fill face_type_name_dict ---
                """
                face_type_name_dict = {<type_uuid>: <name>}
                """
                face_type_name_dict = dict()
                for item in cls.mdb.get_all('face_info'):
                    uuid = str(item.get('_id'))
                    face_type_name_dict[uuid] = item.get('姓名')
                # --- get send_data ---
                """
                send_data = {
                    send_id: 数据id
                    send_list: 数据列表
                }
                """
                send_data = dict()
                for item in cls.mdb.get_all('cache'):
                    if item.get('send_data'):
                        send_data = item.get('send_data')



                # --- check ---
                if not send_data:
                    continue

                # --- check ---
                send_id = send_data.get('send_id')
                if not send_id:
                    continue

                # --- check ---
                if send_id == last_send_id:
                    continue

                # --- check ---
                send_list = send_data.get('send_list')
                if send_list is None or len(send_list) == 0:
                    continue

                # --- debug ---
                # await asyncio.sleep(3)
                # await asyncio.sleep(0.5)
                logger.info('send count is {len_send}｜online count is {len_line}'
                            , len_send=len(send_list), len_line=len(cls.line_dict.values())
                            )

                # --- update ---
                last_send_id = send_id

                # --- send ---
                for line_id in list(cls.line_dict.keys()):

                    try:

                        # --- check ---
                        if not cls.check_line_is_live(line_id):
                            logger.info("websocket link broken.")
                            cls.line_dict.pop(line_id)
                            continue

                        # --- send ---
                        """
                        send_list = [
                            {
                                base_face_uuid: 底库人脸id
                                snap_face_image: 抓拍人脸
                                base_face_image_path: 底库人脸路径
                                face_similarity: 相似度
                            }
                        ]
                        """
                        for data in send_list:

                            # --- check ---
                            if data.get('snap_face_image') is None:
                                continue

                            # --- define ---
                            """
                            send_dict = {
                                input_face_b64: 抓拍人脸图像
                                face_uuid: 人脸id
                                face_name: 人脸名称
                                known_face_b64: 底库人脸图像
                                face_similarity: 相似度
                                face_type_name_list: 人员类型
                            }
                            """
                            send_dict = dict(
                                input_face_b64=cls.image_to_b64(data.get('snap_face_image')),
                                # input_face_b64=str(),
                                known_face_b64=str(),
                                face_uuid=str(),
                                face_name=str(),
                                face_similarity=data.get('face_similarity'),
                                face_type_name_list=list(),
                            )

                            base_face_image_path = data.get('base_face_image_path')
                            if base_face_image_path and os.path.isfile(base_face_image_path):
                                frame = cv2.imread(base_face_image_path)
                                if frame is not None:
                                    _, image = cv2.imencode('.jpg', frame)
                                    base64_data = base64.b64encode(image)  # byte to b64 byte
                                    s = base64_data.decode()  # byte to str
                                    send_dict['known_face_b64'] = f'data:image/jpeg;base64,{s}'

                            # --- fill face_uuid and face_name ---
                            """
                            Face: 陌生人脸表
                            Face.face_name: 人脸名称
                            """
                            face_uuid = data.get('base_face_uuid')
                            if face_uuid:
                                send_dict['face_uuid'] = face_uuid
                                face = cls.mdb.get_one_by_id('Face', face_uuid)
                                if face and face.get('face_name'):
                                    send_dict['face_name'] = face.get('face_name')

                                # --- fill face_type_name_list ---
                                face_type_uuid_list = face.get('face_type_uuid_list')
                                if face_type_uuid_list:
                                    send_dict['face_type_name_list'] = [face_type_name_dict.get(i)
                                                                        for i in face_type_uuid_list
                                                                        if face_type_name_dict.get(i)]
                            line = cls.line_dict.get(line_id)
                            send_json = json.dumps(send_dict)
                            await line.send_text(send_json)


                    except Exception as exception:

                        # --- check ---
                        if not cls.check_line_is_live(line_id):
                            cls.line_dict.pop(line_id)

                        if exception.__class__.__name__ == 'RuntimeError':
                            logger.exception(cls.get_line_state())
                        else:
                            logger.exception(exception)


            except Exception as exception:
                logger.exception("{exception}", exception=exception)
                logger.exception("wait 1 minutes try again!")
                await asyncio.sleep(60)

    @classmethod
    def get_line_total(cls):
        count = 0
        for k, v in cls.line_dict.items():
            count += 1
        return count

    @classmethod
    def check_line_is_live(cls, line_id):
        d1 = {
            0: 'CONNECTING',
            1: 'CONNECTED',
            2: 'DISCONNECTED',
        }
        line = cls.line_dict.get(line_id)
        if line and d1.get(line.client_state.value) != 'DISCONNECTED':
            return True
        else:
            return False

    @classmethod
    def get_line_state(cls):
        d1 = {
            0: 'CONNECTING',
            1: 'CONNECTED',
            2: 'DISCONNECTED',
        }
        d2 = dict()  # {<line_id>: <state>}
        for line_id, line in cls.line_dict.items():
            state = d1.get(line.client_state.value)
            _id = line_id[-6:]
            d2[_id] = state
        return d2

    @staticmethod
    def image_to_b64(image):
        frame = np.asarray(image)  # list to numpy array
        _, image = cv2.imencode('.jpg', frame)
        base64_data = base64.b64encode(image)
        s = base64_data.decode()
        return f'data:image/jpeg;base64,{s}'
