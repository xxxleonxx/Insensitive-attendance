import importlib
import sys
sys.path.append('')
import multiprocessing
import threading
import pymongo
from loguru import logger
from task import TASK
from websocket_server.app import app_run

sys.path.append('camera_hik')
sys.path.append('db_client')
sys.path.append('interface_algorithms')
mdb = importlib.import_module("db_mongo").MongoMethod(database='vms', host='127.0.0.1', port=27017)
camera = importlib.import_module('camera_engine')
interface_engine = importlib.import_module('interface_engine')



def start_the_thread():
    logger.info('线程启动')
    mdb.load_info()
    TASK['抓拍线程'] = threading.Thread(target=camera.run)
    TASK['人脸识别线程'] = threading.Thread(target=interface_engine.run)


    try:
        for thread in TASK.values():
            if thread:
                thread.setDaemon(True)
                thread.start()

    except:
        pass


def run_analyze():
    p1 = multiprocessing.Process(target=start_the_thread)
    p1.daemon = True
    p1.start()





def run_websocket():
    p4 = multiprocessing.Process(target=websocket_app_run)
    p4.daemon = True
    p4.start()


def run_access_control():
    p5 = multiprocessing.Process(target=AccessControl.run)
    p5.daemon = True
    TASK["闸机控制进程"] = p5

