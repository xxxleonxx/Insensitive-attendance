import bson
from pymongo import MongoClient


class MongoMethod(MongoClient):
    data = dict()
    ls = list()

    def __init__(self, host='fra-middleware-mongo', port=27017, database='vms', username='admin', password='admin'):
        super().__init__(f'mongodb://{username}:{password}@{host}:{port}')
        self.db = self[database]

    @staticmethod
    def check_mongodb(db_name, collection):
        client = MongoClient('mongodb://admin:admin@192.168.0.21:7030/')
        database_name = client.list_database_names()

        if db_name not in database_name:
            print('error')
        else:
            db = client[db_name]
            collection_name = db.list_collection_names()
            print(collection_name)
            if collection not in collection_name:
                print('collection_name error')
            else:
                print('true')
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

    def find_person(self, table_name, name='None'):
        # 根据名字查询人的信息
        target = self.db[table_name]
        self.data = target.find({'face_name': name})
        return self.data

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

    def get_arrive_time(self, table_name, name):
        target = self.findperson(table_name, name)
        target_dict = target.__getitem__(0)
        arrive_time = target_dict.get('record_at')
        return arrive_time

    def get_leave_time(self, table_name, name):
        target = self.findperson(table_name, name)
        target_dict = target.__getitem__(0)
        leave_time = target_dict.get('modify_at')
        return leave_time

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
