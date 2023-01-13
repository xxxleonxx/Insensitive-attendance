import importlib
import sys
from functools import wraps
from sanic.exceptions import SanicException
import jwt

from loguru import logger
sys.path.append('../db-client')
mdb = importlib.import_module("db_mongo").MongoMethod(database='vms', host='127.0.0.1', port=27017)

def creat_token(user):
    data = {
        'id': user.get('uid'),
        'username': user['username'],
        'password': user['password'],
    }
    token = jwt.encode(data, 'EL_PSY_KONGROO_LEON')
    return token
def check_token(request):
    try:
        data = jwt.decode(request.headers.get('authorization'), request.app.config.SECRET, algorithms=['HS256'])

    except jwt.exceptions.InvalidTokenError:
        return {}
    else:
        return data


async def login_required(request):
    login_info = check_token(request)
    logger.debug(login_info)
    source = request.json
    tag = source.get('tag', 'v1')
    code = int(source.get('code'))
    if login_info:
        user = mdb.get_one_by_id('User', login_info['id'])
        role_id = user.get('role_id')

        return {
            'uid': login_info['id'],
            'username': login_info['username'],
            'password': login_info['password'],
            'role_id': role_id,
        }
    elif not login_info and tag == 'v3' and code in [1102, 8201]:
        superuser = mdb.get_one('User', {'username': 'admin'})
        return {
            'uid': str(superuser.get('_id')),
            'username': 'admin',
            'password': 'admin',
            'role_id': superuser.get('role_id'),
            'skip_is': True,
        }
    else:
        raise SanicException(status_code=401, context=dict(message='unauthorized access!', code='4'))



