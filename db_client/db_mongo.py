import bson
import numpy as np

from loguru import logger
from pymongo import MongoClient
from sanic_jinja2 import SanicJinja2
from sanic_motor import BaseModel


class MongoMethod(MongoClient):
    data = dict()
    ls = list()

    def __init__(self, host='mongodb://localhost', port=27017, database='vms'):  # username='admin', password='admin'):
        # super().__init__(f'mongodb://{username}:{password}@{host}:{port}')
        super().__init__(f'mongodb://{host}:{port}')
        self.db = self[database]
        self.job_num_list = []
        self.face_name_list = []
        self.feature_list = []
        self.department_list = []
        self.person_type_list = []

    def load_info(self):
        """
        从数据库加载相关字段数据
        :return: all_job_num: 包含所有员工的工号的列表
                 all_name: 包含所有员工的姓名的列表
                 all_face_code: 包含所有员工的人脸特征码的列表
                 all_department: 包含所有员工的部门的列表
                 all_person_type: 包含所有人员类型的列表
        """
        self.job_num_list.clear()
        self.face_name_list.clear()
        self.feature_list.clear()
        self.department_list.clear()
        self.person_type_list.clear()

        face_info = self.db['face_info'].find()
        for info in face_info:
            self.job_num_list.append(info.get('工号'))
            self.face_name_list.append(info.get('姓名'))
            self.feature_list.append(np.frombuffer(info.get('特征'), dtype=np.float32))
            self.department_list.append(info.get('部门'))
            self.person_type_list.append(info.get('人员类型'))
        logger.info('加载人脸数据，数量: {quantity}', quantity=len(self.job_num_list))
        return

    @staticmethod
    def check_mongodb(db_name, collection):
        client = MongoClient('mongodb://admin:admin@localhost/')
        database_name = client.list_database_names()

        if db_name not in database_name:
            logger.error('error')
        else:
            db = client[db_name]
            collection_name = db.list_collection_names()
            logger.debug(collection_name)
            if collection not in collection_name:
                logger.error('collection_name error')
            else:
                logger.debug('true')
        return True

    def creat_capped_collection(self, collection_name=None, size=1000, doc_max=1000):
        self.db.create_collection(collection_name, capped=True, size=size, max=doc_max)
        return True

    def add(self, table_name, data):
        """
        保存数据 单条或多条 并返回ObjectId
        doc: https://www.cnblogs.com/aademeng/articles/9779271.html
        """
        table = self.db[table_name]
        if type(data) == dict:
            return str(table.insert_one(data).inserted_id)
        elif type(data) == list:
            return [str(i) for i in table.insert_many(data).inserted_ids]

    def get_all(self, table_name):
        target = self.db[table_name]
        return target.find()

    def get_index(self, table_name):
        target = self.db[table_name]
        return target.index_information()

    def get_all_vague(self, table_name, field_name, condition):
        # 模糊查询
        target = self.db[table_name]
        return target.find({field_name: {'$regex': condition}})

    def filter_by_aggregate(self, table_name, condition=None, sort_dict=None, page=None, size=None):
        # 聚合查询（不分组） allowDiskUse = True 允许使用磁盘处理 可以解决内存限制的问题
        target = self.db[table_name]
        target_dict = list()
        if isinstance(condition, dict):
            target_dict.append({'$match': condition})
        if isinstance(sort_dict, dict):
            target_dict.append({'$sort': sort_dict})
        if isinstance(page, int) and isinstance(size, int):
            skip = 0 if page == 1 else (page - 1) * size
            target_dict.append({'$skip': skip})
            target_dict.append({'$limit': size})
        result = target.aggregate(target_dict, allowDiskUse=True)
        return result

    def filter_by_project(self, table_name, condition):
        # 投射文档中某一字段
        target = self.db[table_name]
        target_dict = list()
        if isinstance(condition, dict):
            target_dict.append({'$project': condition})

        result = target.aggregate(target_dict, allowDiskUse=True)
        return result

    def get_one_by_id(self, table_name, object_id):
        """
        单条获取
        """
        table = self.db[table_name]
        return table.find_one({'_id': bson.ObjectId(str(object_id))})

    def update_one(self, table_name, unique_dict, update_dict, default_dict=None, need_back=False):
        """
        # if get:
        #     update update_one_by_id
        # else:
        #     insert default_dict
        # return: get
        """
        data = self.get_one(table_name, unique_dict)
        if data:
            object_id = str(data.get('_id'))
            update_dict.update(unique_dict)
            print(unique_dict)
            print(update_dict)
            self.update_one_by_id(table_name, object_id, update_dict)
        else:
            if not default_dict:
                default_dict = dict()
            if '_id' in default_dict:
                del default_dict['_id']
            default_dict.update(update_dict)
            default_dict.update(unique_dict)
            object_id = self.add(table_name, default_dict)
        if need_back:
            return self.get_one_by_id(table_name, object_id)
        else:
            return object_id

    def update_one_by_id(self, table_name, object_id, update_dict):
        """
        单条数据
        """
        table = self.db[table_name]
        return table.update_one({'_id': bson.ObjectId(str(object_id))}, {'$set': update_dict}).modified_count

    def get_one(self, table_name, unique_dict):
        """
        单条获取
        """
        table = self.db[table_name]
        data = table.find_one(unique_dict)
        if data:
            return data
        else:
            return dict()

    def get_col_list(self, table_name, condition, _dic, filed_name):
        """
        只返回某一列的数据（by wdj）
        例子：
        item = mycol.find({}, {'特征': 1, '_id': 0})
        ll = [eval(i.get('特征')) for i in item]
        """
        table = self.db[table_name]
        items = table.find(condition, _dic)
        item = [eval(i.get(filed_name)) for i in items]
        return item


if __name__ == '__main__':

    mdb = MongoMethod(database='vms', host='127.0.0.1')
    a = mdb.get_all('face_info')

    job_num_list = []
    face_name_list = []
    feature_list = []
    department_list = []
    person_type_list = []
    for info in a:
        job_num_list.append(info.get('工号'))
        face_name_list.append(info.get('姓名'))
        feature_list.append(np.frombuffer(info.get('特征'), dtype=np.float32))
        department_list.append(info.get('部门'))
        person_type_list.append(info.get('人员类型'))
    print(job_num_list)
    print(face_name_list)
    print(feature_list)
