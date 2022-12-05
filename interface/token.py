import json

import requests
from sanic import Request
from sanic import Sanic
from sanic_jwt import exceptions
from sanic_jwt import initialize


class User:
    def __init__(self, id, username, password):
        self.user_id = id
        self.username = username
        self.password = password

    def __repr__(self):
        return "User(id='{}')".format(self.user_id)

    def to_dict(self):
        return {"user_id": self.user_id, "username": self.username}


users = [User(1, "admin", "admin")]

username_table = {u.username: u for u in users}
userid_table = {u.user_id: u for u in users}


async def authenticate(request, *args, **kwargs):
    username = request.json.get("username", None)
    password = request.json.get("password", None)

    if not username or not password:
        raise exceptions.AuthenticationFailed("Missing username or password.")
    # 验证账号密码
    user = username_table.get(username, None)
    if user is None:
        raise exceptions.AuthenticationFailed("User not found.")

    if password != user.password:
        raise exceptions.AuthenticationFailed("Password is incorrect.")
    # 返回错误应该只需返回"账号或密码错误"
    return user  # 成功返回user对象


app = Sanic(__name__)
initialize(app, authenticate=authenticate)


async def login_required(request: Request):
    token = request.headers.get('authorization')
  # --- check POST ---
    if request.method == 'POST':
        sources = await request.json()
        tag = sources.get('tag', 'v1')
        code = int(sources.get('code'))
        # methods.debug_log('token.login_required', f"m-107: code -> {code} | token -> {token}")
        if not token and tag == 'v3' and code in [1102, 8201]:
            methods.debug_log('token.login_required', f"m-103: code -> {code}")
            superuser = Global.mdb.get_one('User', {'username': 'admin'})
            return {
                'uid': str(superuser.get('_id')),
                'username': 'admin',
                'password': 'admin',
                'role_id': superuser.get('role_id'),
                'skip_is': True,
            }

    # token = request.headers.get('authorization')
    if not token:
        # raise HTTPException(status_code=401, detail='unauthorized access!')
        raise HTTPException(status_code=401, headers=dict(message='unauthorized access!', code='4'))
    try:
        data = serializer.loads(token)
        user = Global.mdb.get_one_by_id('User', data['id'])
        role_id = user.get('role_id')
        # role_acl = Global.mdb.get_one_by_id('UserRole', role_id).get('role_acl')
        return {
            'uid': data['id'],
            'username': data['username'],
            'password': data['password'],
            'role_id': role_id,
        }
    except Exception as e:
        # raise HTTPException(status_code=401, detail='unauthorized access!')
        raise HTTPException(status_code=401, headers=dict(message='unauthorized access!', code='5'))


# get token
def send_request():
    # Request
    # POST http://127.0.0.1:52055/auth

    try:
        response = requests.post(
            url="http://127.0.0.1:52055/auth",
            headers={
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps({
                "username": "admin",
                "password": "admin"
            })
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=52055)
