import importlib

mdb = importlib.import_module(f"clients.db_mongo").Client(host='mongodb://localhost', port=27017, database='vms')


class System_Init(object):
    def __init__(self):
        self.face_features_list = list()
        self.job_number_list = list()
        self.name_list = list()
        self.department_list = list()

    def data_load(self):
        features = mdb.get_col_list('face_info', {}, {'特征': 1, '_id': 0})
        job_number = mdb.get_col_list('face_info', {}, {'特征': 1, '_id': 0})
        name = mdb.get_col_list('face_info', {}, {'特征': 1, '_id': 0})
        department = mdb.get_col_list('face_info', {}, {'特征': 1, '_id': 0})

        self.face_features_list = [eval(i.get('特征')) for i in features]
        self.job_number_list = [eval(i.get('工号')) for i in job_number]
        self.name_list = [eval(i.get('姓名')) for i in name]
        self.department_list = [eval(i.get('部门')) for i in department]

    def creat_collection(self):
        mdb.creat_capped_collection(collection_name='cache', size=50 * 1024 * 1024, doc_max=100)  # 缓存 容量50mb
        mdb.creat_capped_collection(collection_name='log', size=500 * 1024 * 1024, doc_max=100)  # 日志 容量500mb
