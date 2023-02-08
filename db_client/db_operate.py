import sqlite3
import numpy as np
# from util.log_by_stdlib import *
import base64
import db_mongo
mdb = db_mongo.MongoMethod(database='vms', host='127.0.0.1', port=27017)
class DB:
    def __init__(self):
        self.conn = sqlite3.connect('/home/taiwu/Project/Insensitive-attendance/db_client/vms.db', check_same_thread=False)
        self.job_num_list = []
        self.name_list = []
        self.feature_list = []
        self.sex_list = []
        self.department_list = []
        self.post_list = []
        self.person_type_list = []

    def camera_get_all(self):
        """
        获取所有相机信息
        :return: 查询到的所有信息
        """
        sql = "select * from camera"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        camera_info_list = cursor.fetchall()
        cursor.close()

        return camera_info_list

    def load_info(self):
        """
        从数据库加载相关字段数据
        :return: all_job_num: 包含所有员工的工号的列表
                 all_name: 包含所有员工的姓名的列表
                 all_face_code: 包含所有员工的人脸特征码的列表
                 all_sex: 包含所有员工的性别的列表
                 all_department: 包含所有员工的部门的列表
                 all_post: 包含所有员工的岗位的列表
                 all_person_type: 包含所有人员类型的列表
        """
        self.job_num_list.clear()
        self.name_list.clear()
        self.feature_list.clear()
        self.sex_list.clear()
        self.department_list.clear()
        self.post_list.clear()
        self.person_type_list.clear()

        db_info = self.select_from_info()
        for info in db_info:
            self.job_num_list.append(info[0])
            self.name_list.append(info[1])
            self.feature_list.append(np.frombuffer(info[2], dtype=np.float32))
            self.sex_list.append(info[3])
            self.department_list.append(info[4])
            self.post_list.append(info[5])
            self.person_type_list.append(info[6])
        # debug_log('db_operate', f"m-51: 加载人脸数据，数量：{len(self.job_num_list)}.")
        return

    def select_from_info(self):
        """
        数据库查询操作
        :return: 查询到的所有信息
        """
        sql = "select * from `info`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        db_info = cursor.fetchall()
        cursor.close()
        return db_info

    def get_all_person_idcard(self):
        """
        获取所有人员身份证号
        :return: 所有人员身份证号
        """
        results = []
        sql = "select `工号` from `info`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        db_info = cursor.fetchall()
        cursor.close()
        for idcard in db_info:
            results.append(idcard[0])
        return results

    def add(self, job_num, name, face_code, sex='', department='', post='', person_type='', pic_from='', pic_id=''):
        """
        向info表中添加人脸数据
        :param job_num: 工号
        :param name: 姓名
        :param face_code: 人脸特征码
        :param sex: 性别
        :param department: 部门
        :param post: 岗位
        :param person_type: 人员类型
        :param pic_from: 图片来源：0同步，1手动添加
        :param pic_id: 图片索引
        :return: 无
        """

        sql = "insert into `info`(`工号`, `姓名`, `特征`, `性别`, `部门`, `职务`, `人员类型`, `数据来源`, `图片ID`) " \
              "values (?,?,?,?,?,?,?,?,?)"
        cursor = self.conn.cursor()
        cursor.execute(sql, (job_num, name, face_code, sex, department, post, person_type, pic_from, pic_id))
        self.conn.commit()
        debug_log("util.db_operate", f"添加人员--{job_num}--{name}.")
        cursor.close()

    def delete(self, job_num, pic_id, pic_from):
        """
        删除info表中指定人脸数据
        """
        sql = "delete from `info` where `工号`=? and `图片ID`=? and `数据来源`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (job_num, pic_id, pic_from))
        self.conn.commit()
        cursor.close()

    def get_access_record_by_index(self, index):
        """
        查询access_records表指定索引数据
        """
        sql = "SELECT * FROM `access_records` WHERE `索引`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (index,))
        record = cursor.fetchone()
        cursor.close()
        return record

    def add_access_records(self, date='', job_num='', name='', department='', person_type='', direction='', group=''):
        """
        数据库添加人员进出记录操作
        :param date: 日期时间
        :param job_num: 工号
        :param name: 姓名
        :param department: 部门
        :param person_type: 人员类型
        :param direction: 方向
        :param group: 设备组
        :return: 无
        """

        sql = "insert into `access_records`(`时间`, `工号`, `姓名`, `部门`, `人员类型`, `方向`, `设备组`) " \
              "values (?,?,?,?,?,?,?)"
        cursor = self.conn.cursor()
        cursor.execute(sql, (date, job_num, name, department, person_type, direction, group))
        self.conn.commit()
        cursor.close()

    def get_identify_threshold(self):
        """设置人脸识别阈值"""
        sql = "select `识别阈值` from `remote_config` where `id`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        val = cursor.fetchone()
        cursor.close()
        return int(val[0])

    def set_identify_threshold(self, val):
        """设置人脸识别阈值"""
        sql = "update `remote_config` set `识别阈值`=? where `id`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql, (val,))
        self.conn.commit()
        cursor.close()

    def get_remote_config_tag(self):
        """获取remote_config表主机标识字段"""
        sql = "SELECT `主机标识` FROM `remote_config` WHERE `id`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        db_info = cursor.fetchone()
        cursor.close()
        return db_info[0]

    def update_remote_config_tag(self, tag):
        """更新remote_config表主机标识字段"""
        sql = "UPDATE `remote_config` SET `主机标识`=? WHERE `id`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql, (tag,))
        self.conn.commit()
        cursor.close()

    def update_remote_config_interval(self, interval):
        """更新remote_config表同步间隔字段"""
        sql = "UPDATE `remote_config` SET `同步间隔`=? WHERE `id`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql, (interval,))
        self.conn.commit()
        cursor.close()

    def update_remote_config_sync_address(self, address):
        """更新remote_config表时间间隔字段"""
        sql = "UPDATE `remote_config` SET `同步路径`=? WHERE `id`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql, (address,))
        self.conn.commit()
        cursor.close()

    def update_remote_config_push_address(self, address):
        """更新remote_config表推送路径字段"""
        sql = "UPDATE `remote_config` SET `推送路径`=? WHERE `id`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql, (address,))
        self.conn.commit()
        cursor.close()

    def get_remote_interval(self):
        sql = "select `同步间隔` from `remote_config`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        db_info = cursor.fetchone()
        cursor.close()
        if len(db_info) > 0:
            return int(db_info[0])
        else:
            return 300

    def get_remote_sync_address(self):
        sql = "select `同步路径` from `remote_config`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        db_info = cursor.fetchone()
        cursor.close()
        if len(db_info) > 0:
            return db_info[0]
        else:
            return ''

    def get_remote_push_address(self):
        sql = "select `推送路径` from `remote_config`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        db_info = cursor.fetchone()
        cursor.close()
        if len(db_info) > 0:
            return db_info[0]
        else:
            return ''

    def get_access_device(self, group, direction):
        """获取闸机通讯地址、控制编号"""
        sql = "select `通讯地址`, `控制编号` from `access` where `设备组`=? and `方向`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (group, direction))
        info = cursor.fetchone()
        cursor.close()
        # print(info)
        return info

    def get_access_direction(self):
        sql = "select `方向` from `access`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        info = cursor.fetchall()
        cursor.close()
        print(info)
        return info

    def add_access_device(self, device_name='', device_group='', address='', control_num='', direction=2):
        """
        access表添加闸机设备信息
        :param device_name: 设备名称
        :param device_group: 设备组
        :param address: 通讯地址
        :param control_num: 控制编号
        :param direction:方向（0进场，1，出场，2双向）
        """
        sql = "insert into `access`(`设备名称`, `设备组`, `通讯地址`, `控制编号`, `方向`) " \
              "values (?,?,?,?,?)"
        cursor = self.conn.cursor()
        cursor.execute(sql, (device_name, device_group, address, control_num, direction))
        self.conn.commit()
        cursor.close()

    def delete_access_device(self, device_group='', direction=2):
        """
        access表添加闸机设备信息
        :param device_group: 设备组
        """
        sql = "delete from `access` where `设备组`=? and `方向`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (device_group, direction))
        self.conn.commit()
        cursor.close()

    def add_camera_device(self, camera_ip='', username='', password='', direction='', group=''):
        """
        camera表添加相机设备信息
        :param camera_ip: ip
        :param username: 用户名
        :param password: 密码
        :param direction: 方向
        :param group: 设备组别
        """
        sql = "insert into `camera`(`ip`, `用户名`, `密码`, `方向`, `设备组`) " \
              "values (?,?,?,?,?)"
        cursor = self.conn.cursor()
        cursor.execute(sql, (camera_ip, username, password, direction, group))
        self.conn.commit()
        cursor.close()

    def delete_camera_device(self, camera_ip=''):
        """
        camera表删除相机设备信息
        :param camera_ip: ip
        """
        sql = "delete from `camera` where `ip`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (camera_ip,))
        self.conn.commit()
        cursor.close()

    def update_camera_device(self, camera_ip='', username='', password='', direction='', group='', new_ip=''):
        """
        cameras表删除相机设备信息
        :param camera_ip: ip
        :param username: 用户名
        :param password: 密码
        :param direction: 方向
        :param group: 设备组别
        :param new_ip: 新ip
        """
        sql = "UPDATE `camera` SET `ip`=?, `用户名`=?, `密码`=?, `方向`=?, `设备组`=? WHERE `ip`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (new_ip, username, password, direction, group, camera_ip))
        self.conn.commit()
        cursor.close()

    def get_push_result_index(self):
        """
        获取push_result表中推送信息索引
        """
        sql = "SELECT `索引` FROM `push_result`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        index = cursor.fetchone()
        cursor.close()
        return index[0]

    def update_push_result_index(self, index, new_index):
        """
        更新push_result表中推送记录索引
        """
        sql = "UPDATE `push_result` SET `索引`=? WHERE `索引`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (new_index, index))
        self.conn.commit()
        cursor.close()

    def update_access_record_state(self, index, state):
        """更新access_records表指定索引对应记录的状态，0是未上传，1是已上传"""
        sql = "UPDATE `access_records` SET `状态`=? WHERE `索引`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (state, index))
        self.conn.commit()
        cursor.close()

    def get_push_access_index(self):
        """
        获取push_access表中推送信息索引
        """
        sql = "SELECT `索引` FROM `push_access`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        index = cursor.fetchone()
        cursor.close()
        return index[0]

    def update_push_access_index(self, index, new_index):
        """
        更新push_result表中推送记录索引
        """
        sql = "UPDATE `push_access` SET `索引`=? WHERE `索引`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (new_index, index))
        self.conn.commit()
        cursor.close()

    def get_person_list(self, page, num):
        sql = "select count(*) from `info` where `图片ID`=0"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchone()
        cursor.close()
        count = res[0]

        sql = "select `工号`, `姓名`, `人员类型` from `info` where `图片ID`=0 limit ?,?"
        i = int(page) - 1
        n = num
        cursor = self.conn.cursor()
        cursor.execute(sql, (i, n))
        db_info = cursor.fetchall()
        cursor.close()

        results = []
        if len(db_info) == 0:
            return results

        for one in db_info:
            job_num = one[0]
            name = one[1]
            person_type = one[2]
            sql = "select `图片ID` from `info` where `工号`=?"
            cursor = self.conn.cursor()
            cursor.execute(sql, (job_num,))
            img_ids = cursor.fetchall()
            cursor.close()

            f = open(f"./face_lib/{job_num}_0.jpg", mode="rb")
            b64_img = base64.b64encode(f.read()).decode()
            f.close()
            one_dict = {"job_num": job_num,
                        "name": name,
                        "person_type": person_type,
                        "face_img": b64_img,
                        "face_lib": []}

            if len(img_ids) == 0:
                continue
            for img_id in img_ids:
                f1 = open(f"./face_lib/{job_num}_{img_id[0]}.jpg", mode="rb")
                b64_img1 = base64.b64encode(f1.read()).decode()
                f1.close()
                one_dict["face_lib"].append({"pic_id": img_id[0], "url": b64_img1})
            results.append(one_dict)
        return results, count

    def get_device_list(self, group):
        """获取闸机通讯地址、控制编号"""
        sql = "select `ip`, `方向` from `camera` where `设备组`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (group,))
        info1 = cursor.fetchall()
        cursor.close()
        camera_list = []
        if len(info1) != 0:
            for one in info1:
                camera_list.append({"name": one[0], "ip": one[0], "direction": one[1]})

        sql = "select `设备名称`, `通讯地址`, `方向`, `控制编号` from `access` where `设备组`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (group,))
        info2 = cursor.fetchall()
        cursor.close()
        access_list = []
        if len(info2) != 0:
            for one in info2:
                access_list.append({"name": one[0], "address": one[1], "direction": one[2], "control_num": one[3]})

        return {"camera": camera_list, "access": access_list}

    def get_function_status(self):
        sql = "select * from `function`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        info = cursor.fetchone()
        cursor.close()
        return {'push_status': info[0], 'sync_status': info[1], 'control_status': info[2]}

    def set_function_push_status(self, status):
        sql = "UPDATE `function` SET `数据推送`=? WHERE `索引`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql, (status,))
        self.conn.commit()
        cursor.close()

    def set_function_sync_status(self, status):
        sql = "UPDATE `function` SET `人员同步`=? WHERE `索引`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql, (status,))
        self.conn.commit()
        cursor.close()

    def set_function_control_status(self, status):
        sql = "UPDATE `function` SET `闸机控制`=? WHERE `索引`=1"
        cursor = self.conn.cursor()
        cursor.execute(sql, (status,))
        self.conn.commit()
        cursor.close()

    def get_person_list_by_job_num(self, job_num):
        sql = "SELECT `姓名`, `人员类型` FROM `info` WHERE `工号`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (job_num,))
        info = cursor.fetchone()
        cursor.close()
        if len(info) == 0:
            return {}
        f = open(f"./face_lib/{job_num}_0.jpg", mode="rb")
        b64_img = base64.b64encode(f.read()).decode()
        result_dict = {"job_num": job_num,
                       "name": info[0],
                       "person_type": info[1],
                       "face_img": b64_img,
                       "face_lib": []}
        sql = "select `图片ID` from `info` where `工号`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (job_num,))
        img_ids = cursor.fetchall()
        cursor.close()
        if len(img_ids) == 0:
            return result_dict
        for img_id in img_ids:
            f1 = open(f"./face_lib/{job_num}_{img_id[0]}.jpg", mode="rb")
            b64_img1 = base64.b64encode(f1.read()).decode()
            f1.close()
            result_dict["face_lib"].append({"pic_id": img_id[0], "url": b64_img1})
        return result_dict

    def add_to_client(self, job_num, name, group, pic):
        """
        向client表中添加人脸数据
        """

        sql = "insert into `client`(`工号`, `姓名`, `组别`, `图片`) " \
              "values (?,?,?,?)"
        cursor = self.conn.cursor()
        cursor.execute(sql, (job_num, name, group, pic))
        self.conn.commit()
        debug_log("util.db_operate", f"添加识别结果到client表--{job_num}--{name}.")
        cursor.close()

    def get_client_info_by_index(self, index):
        """
        获取push_result表中推送信息索引
        """
        sql = "SELECT `姓名`, `组别`, `图片` FROM `client` WHERE `索引`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (index,))
        info = cursor.fetchone()
        cursor.close()
        return info

    def get_push_client_index(self):
        """
        获取push_client表中推送记录索引
        """
        sql = "SELECT `索引` FROM `push_client`"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        index = cursor.fetchone()
        cursor.close()
        return index[0]

    def update_push_client_index(self, index, new_index):
        """
        更新push_client表中推送记录索引
        """
        sql = "UPDATE `push_client` SET `索引`=? WHERE `索引`=?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (new_index, index))
        self.conn.commit()
        cursor.close()



if __name__ == '__main__':
    db = DB()
    db.load_info()
    ll = db.select_from_info()
    print(ll[0])
    unique = {
    "人员类型": "0",
    "图片ID": "0",
    "姓名": "李乐童",
    "工号": ll[0][0],
    "性别": "男",
    "数据来源": "0",
   "特征":ll[0][2],
   "职务":"",
    "部门": ""
}
    # mdb.add('face_info',unique)
    item = mdb.get_all('face_info')
    for i in item:
        print(i)


# DB.get_person_list(1, 5)
# print(DB.get_person_list_by_job_num("371122199807281229"))
