import importlib
import sys


sys.path.append('../db-client')
mdb = importlib.import_module("db_mongo").MongoMethod(database='vms', host='127.0.0.1', port=27017)


async def action_1101(**sources):
    """
    保存主题设置
    """
    # --- check ---
    data = sources.get('data')
    if not data:
        return dict(code=1, details=f"something is wrong.")

    # --- set GlobalVariable.ThemeSettings ---
    unique_dict = {'name': 'ThemeSettings'}
    update_dict = {'args': data}
    mdb.update_one('GlobalVariable', unique_dict, update_dict)

    return dict(code=0, data=data)


async def action_1102(**sources):
    """
    获取主题设置
    """
    # --- get GlobalVariable.ThemeSettings ---
    unique_dict = {'name': 'ThemeSettings'}
    item = mdb.get_one('GlobalVariable', unique_dict)

    # --- check ---
    if not item:
        return dict(code=1, details=f"something is wrong.")

    return dict(code=0, data=item.get('args'))
