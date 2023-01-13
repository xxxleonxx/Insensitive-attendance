import importlib
import sys
import time
from werkzeug.security import generate_password_hash

sys.path.append('../db-client')
mdb = importlib.import_module("db_mongo").MongoMethod(database='vms', host='127.0.0.1', port=27017)


async def action_1001(**sources):
    """
    用户管理，创建用户
    提示: User
    """
    if not sources.get('role_id'):
        return dict(data='', code=1)

    user = mdb.get_one('User', {'username': sources.get('username')})
    if user:
        return dict(data='', code=2)

    data = {
        'username': sources.get('username'),
        'password': generate_password_hash(sources.get('password')),
        'role_id': sources.get('role_id'),
        'create_at': time.time(),
    }
    uuid = mdb.add('User', data)
    return dict(data=uuid, code=0)


async def action_1002(**sources):
    """
    用户管理，删除用户
    提示: User
    """
    uuid = sources.get('uuid')
    if not uuid:
        return dict(data='', code=1)

    if uuid == sources.get('g_user_id'):
        return dict(data='', code=2)

    user = mdb.get_one_by_id('User', uuid)
    if not user:
        return dict(data='', code=3)

    if user.get('username') in ['admin']:
        return dict(data='', code=4)

    mdb.remove_one_by_id('User', uuid)
    return dict(data=uuid, code=0)


async def action_1003(**sources):
    """
    用户管理，修改用户
    提示: User
    """
    uuid = sources.get('uuid')
    role_id = sources.get('role_id')
    if not uuid:
        return dict(data='', code=1)

    update_dict = {
        'username': sources.get('username'),
        'role_id': role_id,
    }
    mdb.update_one_by_id('User', sources.get('uuid'), update_dict)
    return dict(data=sources.get('uuid'), code=0)


async def action_1004(**sources):
    """
    用户管理，列表
    提示: User
    """
    outputs = []
    for item in mdb.get_all('User'):
        role = mdb.get_one_by_id('UserRole', item.get('role_id'))
        data = {
            'uuid': str(item.get('_id')),
            'username': item.get('username'),
            'role_id': item.get('role_id'),
            'role_name': role.get('role_name'),
        }
        outputs.append(data)
    return dict(data=outputs, code=0)


async def action_1005(**sources):
    """
    用户管理，新增角色
    提示: UserRole
    """
    data = {
        'role_name': sources.get('role_name'),
        'role_acl': sources.get('role_acl'),
        'create_at': time.time(),
    }
    role_id = mdb.add('UserRole', data)
    return dict(data=role_id, code=0)


async def action_1006(**sources):
    """
    用户管理，删除角色
    提示: UserRole
    sign = true 系统设定 不可删除
    """
    uuid = sources.get('uuid')
    if not uuid:
        return dict(data='', code=1)

    users = mdb.filter('User', {'role': uuid})
    if users.count():
        return dict(data='', code=2)

    role = mdb.get_one_by_id('UserRole', uuid)
    if not role:
        return dict(data='', code=3)

    if role.get('sign'):
        return dict(data='', code=4)

    mdb.remove_one_by_id('UserRole', uuid)
    return dict(data=uuid, code=0)


async def action_1007(**sources):
    """
    用户管理，修改角色
    """
    uuid = sources.get('uuid')
    if not uuid:
        return dict(data='', code=1)

    update_dict = {
        'role_name': sources.get('role_name'),
        'role_acl': sources.get('role_acl'),
    }
    mdb.update_one_by_id('UserRole', uuid, update_dict)
    return dict(data=uuid, code=0)


async def action_1008(**sources):
    """
    用户管理，角色列表
    提示: UserRole
    """
    outputs = []
    for item in mdb.get_all('UserRole'):
        outputs.append({
            'role_id': str(item.get('_id')),
            'role_name': item.get('role_name'),
            'role_acl': item.get('role_acl'),
        })
    return dict(data=outputs, code=0)


async def action_1009(**sources):
    """
    用户管理，获取当前用户权限
    提示: UserRole
    """
    g_role_id = sources.get('g_role_id')
    role_acl = mdb.get_one_by_id('UserRole', g_role_id).get('role_acl')
    return dict(data=role_acl, code=0)
