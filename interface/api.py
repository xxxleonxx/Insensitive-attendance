import importlib
import sys
import time

import jwt
from loguru import logger
from sanic import Sanic, Request, response
from sanic.response import json
from sanic_cors import CORS

from creat_token import login_required,creat_token

sys.path.append('../db-client')
sys.path.append('../interface/route')
app = Sanic(__name__)
app.config.SECRET = 'EL_PSY_KONGROO_LEON'
CORS(app, supports_credentials=True)
mdb = importlib.import_module("db_mongo").MongoMethod(database='vms', host='127.0.0.1', port=27017)
app.ext.dependency(login_required)

ROUTE = \
    {
        1000: 'route_1000.action_1000',  # 配置主机唯一标识
        1001: 'route_1000.action_1001',  # 查询、配置远程参数
        1002: 'route_1000.action_1002',  # 获取功能状态

        1101: 'route_1100.action_1101',  # 保存主题设置
        1102: 'route_1100.action_1102',  # 获取主题设置

    }
methods_list = {}


@app.get("/hello")
async def hello(request):
    return json({'msg': 'successful'})


def now_ts(unit='s'):
    """
    获取当前时间戳
    Example:
        now_ts()
        1420041600
    """
    unit_dict = {
        'ms': 1,
        's': 0,
    }
    t = time.time()
    if unit_dict.get(unit):
        t = t * (1000 ** unit_dict.get(unit))
    return round(t, 2)


def _get_method_by_code(code, tag='v3'):
    """
    根据code获取method（不调用不加载）
    """

    try:
        if tag == 'v3':
            if code in methods_list:
                return methods_list[code]
            else:
                file_name, method_name = ROUTE.get(code).split('.')
                script = importlib.import_module(f"routes.{file_name}")
                method = getattr(script, method_name)
                methods_list[code] = method

                return method

    except Exception as exception:
        logger.exception(exception)


@app.post('/actions')
async def actions(request: Request, user=login_required):
    users = await user(request)
    logger.debug(users)
    if not users.get('skip_is'):
        token = creat_token(users)
        logger.debug(token)
        response.json({'authorization': token})

    sources = request.json
    tag = sources.get('tag', 'v1')
    code = int(sources.get('code'))
    page = sources.get('page', 1)
    size = sources.get('size', 10)
    ban_keys = ['tag', 'code']
    sources = {key: val for key, val in sources.items() if key not in ban_keys}
    sources['g_user_id'] = users.get('uid')
    sources['g_user_name'] = users.get('username')
    sources['g_user_pass'] = users.get('password')
    sources['g_role_id'] = users.get('role_id')

    try:
        run_at = now_ts()
        method = _get_method_by_code(code=code, tag=tag)
        result = await method(**sources)
        logger.info("actions.action_{code} use time {use_time}s", code=code, use_time=round(now_ts() - run_at, 2))
        if tag == 'v3':
            if code in [
            ] and result.__class__.__name__ == 'dict':
                return json(dict(
                    code=result.get('code'),
                    data=result.get('data', [])[(page - 1) * size: page * size],
                    page=page,
                    size=size,
                    total=len(result.get('data', [])),
                ))
            elif result.__class__.__name__ in ['file', 'dict']:
                return json(result)
            else:
                return dict(data=result, type=result.__class__.__name__)

    except Exception as exception:
        logger.exception(" tag: {tag}, code: {code}", tag=tag, code=code)
        logger.exception(exception)
        return json(dict(code=-1, data=[], message=f"something is wrong. [{exception.__class__.__name__}]"))


# @app.put('/actions')
# async def upload(request: Request, user=login_required):
#     try:
#         sources = dict()


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8003, dev=True)
