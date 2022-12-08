import importlib
import sys
import time
import creat_token
from loguru import logger
from sanic import Sanic, Request
from sanic_cors import CORS

from route import ROUTE

sys.path.append('../db-client')
app = Sanic(__name__)
CORS(app, supports_credentials=True)
mdb = importlib.import_module("db_mongo").MongoMethod(database='vms', host='127.0.0.1', port=27017)


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
            if code in ROUTE:
                return ROUTE[code]
            else:
                file_name, method_name = ROUTE.get(code).split('.')
                script = importlib.import_module(f"actions.v3.{file_name}")
                method = getattr(script, method_name)
                ROUTE[code] = method
                return method

    except Exception as exception:
        logger.exception(exception)


@app.post('/actions')
async def actions(request: Request, response, user: dict = app.ext.dependency(creat_token.login_required)):
    # user = app.ext.dependency(creat_token.login_required())
    if not user.get('skip_is'):
        response.headers['authorization'] = creat_token.get_token_by_usr(user)
    sources = await request.json()
    tag = sources.get('tag', 'v1')
    code = int(sources.get('code'))
    page = sources.get('page', 1)
    size = sources.get('size', 10)
    ban_keys = ['tag', 'code']
    sources = {key: val for key, val in sources.items() if key not in ban_keys}
    sources['g_user_id'] = user.get('uid')
    sources['g_user_name'] = user.get('username')
    sources['g_user_pass'] = user.get('password')
    sources['g_role_id'] = user.get('role_id')

    try:
        run_at = now_ts()
        method = _get_method_by_code(code=code, tag=tag)
        # methods.debug_log(f"actions.action_{code}", f"1: {method}")
        result = await method(**sources)
        logger.debug("actions.action_{code} use time {use_time}s", code=code, use_time=round(now_ts() - run_at, 2))
        if tag == 'v3':
            if code in [
            ] and result.__class__.__name__ == 'dict':
                return dict(
                    code=result.get('code'),
                    data=result.get('data', [])[(page - 1) * size: page * size],
                    page=page,
                    size=size,
                    total=len(result.get('data', [])),
                )
            elif result.__class__.__name__ in ['FileResponse', 'dict']:
                return result
            else:
                return dict(data=result, type=result.__class__.__name__)

    except Exception as exception:
        logger.exception(" tag: {tag}, code: {code}", tag=tag, code=code)
        logger.exception(exception)
        return dict(code=-1, data=[], message=f"something is wrong. [{exception.__class__.__name__}]")


if __name__ == '__main__':
    pass
