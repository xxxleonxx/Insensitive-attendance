import requests
from loguru import logger


# --- get token ---
service_url = 'http://127.0.0.1:52055'
url = f'{service_url}/restful/tokens'
data = {
    'username': 'admin',
    'password': 'admin',
}
response = requests.post(url=url, json=data)
code = response.json().get('code')
token = response.headers.get('authorization')
print(code, token)

# --- test 1101 保存主题设置 ---
# url = f'{service_url}/actions'
# data = {
#     'tag': 'v3',  # 接口版本
#     'code': 1101,  # 接口号
#     'data': {1: [{2: [3, 5]}]},  # 时间范围 - 开始（非空项）
# }
# response = requests.post(url=url, json=data, headers={'authorization': token})
# print(response.json())

# --- test 1102 获取主题设置 ---
url = f'{service_url}/actions'
data = {
    'tag': 'v3',  # 接口版本
    'code': 1102,  # 接口号
}
print(url)
# response = requests.post(url=url, json=data, headers={'authorization': token})
response = requests.post(url=url, json=data)
print(response.json())



