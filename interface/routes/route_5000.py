import sys

import sanic
import numpy as np

sys.path.append('../..')
import multiprocessing
from loguru import logger
from sanic import Sanic
from sanic.response import json
from tool_kit.methods import image_decode
from db_client.db_mongo import MongoMethod
from inference_algorithms.inference_engine import InferenceEngine

mdb = MongoMethod(database='vms', host='127.0.0.1', port=27017)


async def action_5001(**sources):
    try:
        person = sources.get('data')  # 应用端请求内容
        if not person:
            return dict(code=1, details=f"something is wrong.")

        job_num = person['job_num']
        face_name = person['name']
        person_type = person['person_type']
        pic_id = person['pic_id']
        b64_img = person['b64img']
        if b64_img is None or b64_img == '':
            return json({'code': 205, 'msg': 'base64数据为空'})  # 回传结果

        image = image_decode(b64_img)
        feature, face_img = InferenceEngine.extract_feature(image)
        if feature is None:
            logger.debug("添加劳务人员信息失败，图片中未检测到人脸.")
            return json({'code': 204, 'msg': '添加劳务人员信息失败，图片中未检测到人脸'})  # 回传结果
        save_img = cv2.resize(face_img, (80, 96))
        cv2.imwrite(f"/home/taiwu/Project/Data_Storage_directory/face_comparison_library/{job_num}_{pic_id}.jpg",
                    save_img)
        feature_encoding = feature.astype(np.float32).tobytes()

        mdb.add(job_num=job_num, name=face_name, face_code=feature_encoding, person_type=person_type, pic_from=1,
                pic_id=pic_id)
        logger.info("添加劳务人员信息成功：{job_num}, {name}.", job_num=job_num, name=name)
        return json({'code': 200, 'msg': '成功'})  # 回传结果

    except Exception as e:
        logger.exception("添加劳务人员信息失败,{e}", e=e)
        return json({'code': 404, 'msg': e})  # 回传结果
