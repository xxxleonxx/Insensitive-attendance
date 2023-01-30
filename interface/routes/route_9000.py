import importlib
import sys
from loguru import logger
import multiprocessing
sys.path.append('..')
task = importlib.import_module('task')
sys.path.append('../db_client')
mdb = importlib.import_module("db_mongo").MongoMethod(database='vms', host='127.0.0.1', port=27017)


async def action_9001(**sources):
    """
    启动闸机控制
    """
    # --- check ---
    data = sources.get('data')
    if not data:
        return dict(code=1, details=f"something is wrong.")

    # --- set GlobalVariable.gate_control_status ---
    unique_dict = {'name': 'gate_control_status'}
    update_dict = {'status': data.get('status')}
    mdb.update_one('GlobalVariable', unique_dict, update_dict)
    task.TASK['闸机控制进程'].start()
    logger.info('启动闸机控制进程成功')
    return dict(code=0, data=data)


async def action_9002(**sources):
    """
    关闭闸机控制
    """
    # --- get GlobalVariable.gate_control_status ---
    data = sources.get('data')
    if not data:
        return dict(code=1, details=f"something is wrong.")
    if task.TASK["闸机控制进程"].is_alive():
        task.TASK["闸机控制进程"].terminate()
        task.TASK["闸机控制进程"] = multiprocessing.Process(target=AccessControl.run)
        task.TASK["闸机控制进程"].daemon = True
        unique_dict = {'name': 'gate_control_status'}
        update_dict = {'status': data.get('status')}
        mdb.update_one('GlobalVariable', unique_dict, update_dict)
        logger.info('关闭闸机控制进程成功')
        return dict(code=0, data=data)
    else:
        logger.info('关闭闸机控制进程失败')
        return dict(code='404', msg="something is wrong.")


