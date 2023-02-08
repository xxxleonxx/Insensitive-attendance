import cv2
import db_client.db_mongo
import numpy as np
from tool_kit.methods import img_bytes2array
from inference_algorithms.inference_engine import InferenceEngine
mdb = db_client.db_mongo.MongoMethod(database='vms', host='127.0.0.1', port=27017)
def save_image_to_blob(image_bytes):
    # image = img_bytes2array(image_bytes)  # 图片二进制数据转array
    print(type(image_bytes))
    feature, face_img = InferenceEngine.extract_feature(image_bytes)  # 人脸检测、对齐、特征提取
    print(type(face_img))
    # save_img = cv2.resize(face_img, (80, 96))
    # cv2.imwrite(f"/home/taiwu/Project/Data_Storage_directory/face_comparison_library", save_img)
    feature_encoding = feature.astype(np.float32).tobytes()  # array转bytes
    mdb.add('face_info',{
    "_id": "63870042db499a959e4f774d",
    "人员类型": "0",
    "图片ID": "0",
    "姓名": "李乐童",
    "工号": "370181199411202714",
    "性别": "男",
    "数据来源": "0",
   "特征":feature_encoding,
   "职务":"",
    "部门": ""
})

if __name__ == '__main__':
    img = cv2.imread('/home/taiwu/Project/Data_Storage_directory/2.jpg')
    save_image_to_blob(img)
