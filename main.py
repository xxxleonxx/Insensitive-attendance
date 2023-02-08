import multiprocessing
import threading
import time

import db_client.db_mongo
from loguru import logger
from task import TASK
from camera_hik.camera_engine import CameraEngine
from inference_algorithms.inference_engine import InferenceEngine

from websocket_server.app import app_run
# from business_layer.gate_control import GateControl
from interface.api import app


mdb = db_client.db_mongo.MongoMethod(database='vms', host='127.0.0.1', port=27017)


def start_the_thread():
    logger.info('线程启动')
    mdb.load_info()
    TASK['抓拍线程'] = threading.Thread(target=CameraEngine.run)  # target 后面线程程序要带（）
    TASK['抓拍线程'].start()
    TASK['定时重载人员库线程'] = threading.Thread(target=InferenceEngine.reload_face_lib)
    TASK['定时重载人员库线程'].start()
    TASK['人脸识别线程'] = threading.Thread(target=InferenceEngine.run)
    TASK['人脸识别线程'].start()


def run_analyze():
    p1 = multiprocessing.Process(target=start_the_thread)  # target 后面线程程序要带（）
    p1.daemon = False
    p1.start()


def run_time():
    while True:
        at = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        logger.info(at)
        time.sleep(60)

#
# def run_websocket():
#     p4 = multiprocessing.Process(target=run_time)
#     p4.daemon = False
#     p4.start()


def run_websocket():
    p4 = multiprocessing.Process(target=app_run)
    p4.daemon = True
    p4.start()


# def run_access_control():
#     p5 = multiprocessing.Process(target=GateControl.run)
#     p5.daemon = True
#     TASK["闸机控制进程"] = p5
def run_server():
    logger.info(" 启动服务端.")
    server = app.run(host="0.0.0.0", port=8003, access_log=False, dev=False)
    server.serve_forever()

def run():
    """启动所有任务"""
    logger.info('main主进程正在运行.')
    multiprocessing.set_start_method('spawn')
    run_websocket()  # 前端websocket连接进程
    # run_access_control()
    run_analyze()  # 出入信息记录、定时重载人员底库、启动子服务

    run_server()                # 启动基础服务


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()

