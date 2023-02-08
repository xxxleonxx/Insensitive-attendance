import sys
import json
import time
import socket
import importlib
from loguru import logger

sys.path.append('db_client')
mdb = importlib.import_module("db_mongo").MongoMethod(database='vms', host='127.0.0.1', port=27017)


class GateControl:

    @classmethod
    def open(cls, control_num, addr):
        """addr的格式应为("ip", port)"""
        # TODO:           ↑     ↑
        send_data = {"cmd": "twkongzhi", "ip": addr[0], "DO": control_num, "status": 0}
        send_data = json.dumps(send_data)
        print(send_data)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind(("0.0.0.0", 52084))
        udp_socket.sendto(send_data.encode("utf-8"), addr)
        udp_socket.close()
        pass

    @classmethod
    def run(cls):
        pass
        # while True:
        #     try:
        #         index = cls.database.get_push_access_index()
        #         record = cls.database.get_access_record_by_index(index)
        #         print("############-------########", record)
        #         if record is None:
        #             time.sleep(0.6)
        #             continue
        #         direction = record[6]
        #         group = record[7]
        #         # print(record)
        #         info = cls.database.get_access_device(group, direction)
        #         logger.info(info)
        #         if info is None:
        #             cls.database.update_push_access_index(index, int(index) + 1)  # 更新push_access表推送索引记录
        #             continue
        #         # print(info)
        #         addr = info[0]
        #         tmp = addr.split(":")
        #         ip = tmp[0]
        #         port = tmp[1]
        #         control_num = info[1]
        #         cls.open(int(control_num), (ip, int(port)))
        #         logger.debug('access.push_access', f"m-57: access open.")
        #
        #         cls.database.update_push_access_index(index, int(index) + 1)  # 更新push_access表推送索引记录
        #     except Exception as e:
        #         logger.exception(e)
        #         logger.exception('access.push_access wait 1s try again!')
        #         time.sleep(1)
        #         continue
