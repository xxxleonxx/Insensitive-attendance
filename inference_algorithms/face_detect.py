"""
人脸检测、人脸对齐
"""
import numpy as np
from skimage import transform as trans
import onnxruntime
import os.path as osp
import cv2


src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


# In[66]:


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    """
    歪脸矫正方法
    see: https://github.com/deepinsight/insightface/blob/master/recognition/arcface_mxnet/common/face_align.py
    """
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped




def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class FaceDetection:
    def __init__(self, model_file=None, session=None):
        if not model_file:
            model_file = './resources/models/stage02_recognize.onnx'

        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            # self.session = onnxruntime.InferenceSession(self.model_file, None)  # for cpu
            self.session = onnxruntime.InferenceSession(self.model_file, providers=['CUDAExecutionProvider'])  # for gpu
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        # print("input_shape:",input_shape)
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        # print('image_size:', self.image_size)
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        # print("input_name:",self.input_name)
        # print("output_name:",self.output_names)
        self.input_mean = 127.5
        self.input_std = 128.0
        # assert len(outputs)==10 or len(outputs)==15
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]  # 基本执行这个
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        # else:
        #     self.session.set_providers(['CUDAExecutionProvider'], [{'device_id': ctx_id}])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size
        img_tmp = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img_tmp = np.asfarray(img_tmp, dtype="float32")
        blob = cv2.dnn.blobFromImage(img_tmp, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean))
        self.session.run(self.output_names, {self.input_name: blob})

    def init_det_threshold(self, det_threshold):
        """
        单独设置人脸检测阈值
        :param det_threshold: 人脸检测阈值
        :return:
        """
        self.det_thresh = det_threshold

    def forward(self, img, threshold=0.6, swap_rb=True):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        # print('input_size:',input_size)
        blob = cv2.dnn.blobFromImages([img], 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=swap_rb)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        # print("net_outs:::",net_outs[0])
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc  # 3
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            # print("scores:",scores)
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                # print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            # print(anchor_centers.shape,bbox_preds.shape,scores.shape,kps_preds.shape)
            pos_inds = np.where(scores >= threshold)[0]
            # print("pos_inds:",pos_inds)
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        # print("....:",bboxes_list)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, det_thresh=None, metric='default', swap_rb=True):
        """

        :param img: 原始图像
        :param input_size:  输入尺寸,元组或者列表
        :param max_num: 返回人脸数量, 如果为0,表示所有,
        :param det_thresh: 人脸检测阈值,
        :param metric: 排序方式,默认为面积+中心偏移, "max"为面积最大排序
        :param swap_rb: 是否进行r b通道转换, 如果传入的是bgr格式图片,则需要为True
        :return:
        """
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]

        # resize方法选择,缩小选择cv2.INTER_AREA , 放大选择cv2.INTER_LINEAR
        resize_interpolation = cv2.INTER_AREA if img.shape[0] >= input_size[0] else cv2.INTER_LINEAR
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=resize_interpolation)

        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        if det_thresh is None:
            det_thresh = self.det_thresh
        scores_list, bboxes_list, kpss_list = self.forward(det_img, det_thresh, swap_rb)
        # print("====",len(scores_list),len(bboxes_list),len(kpss_list))
        # print("scores_list:",scores_list)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def inference_with_image_array(self, image_array):
        """
        算法推断
        """
        try:
            bboxes, kpss = self.detect(image_array, input_size=(640, 640))  # 得到box跟关键点
            # methods.debug_log('FaceDetectionEngine', f"m-416 | bboxes: {bboxes}, kpss: {kpss}")

            results = list()
            for bbox, pts in zip(bboxes, kpss):
                x1, y1, x2, y2, score = bbox.astype(int)
                # methods.debug_log('FaceDetectionEngine', f"m-420 | x1: {x1}, y1: {y1},  x2: {x2}, y2: {y2}")
                face_iamge = image_array[y1:y2, x1:x2]
                align_face = norm_crop(image_array, pts)  # 切忌使用 image_array 而非 face_iamge
                results.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'raw_face': face_iamge,  # 人脸图像
                    'align_face': align_face,  # 歪脸矫正图像 todo 存在图像模糊的问题
                })
            return results

        except Exception as exception:
            print('FaceDetectionEngine', f"m-434: exception | {exception}")
            return None

    @staticmethod
    def image_bytes_to_image_array(image_bytes, mode='RGB'):
        """
        数据格式转换
        """
        try:
            _image = np.asarray(bytearray(image_bytes), dtype='uint8')
            _image = cv2.imdecode(_image, cv2.IMREAD_COLOR)
            return _image
        except Exception as exception:
            print('FaceDetectionEngine', f"m-448: exception | {exception}")
            return None


if __name__ == '__main__':
    # --- init ---
    model_path = '../'
    model_path += 'stage02_recognize.onnx'
    # model_path = 'models/scrfd/stage02_recognize.onnx'
    # model_path = 'models/scrfd/scrfd_2.5g_kps.onnx'
    agent = FaceDetection(model_file=model_path)
    agent.prepare(0)

    # img_path = 'data/test2.jpg'
    # img_path = '/home/server/resources/TestData/2022/0315/IMG_20220315_101522.jpg'
    # img_path = '/home/server/resources/TestData/2022/0315/20220318173524.jpg'
    # img_path = '/home/server/resources/TestData/2022/0315/worker_003.jpg'
    img_path = '../test.jpg'
    image_array = cv2.imread(img_path)
    results = agent.inference_with_image_array(image_array)
    # cv2.imwrite('output-face.jpg', results[0]['align_face'])
    cv2.imshow('face', results[0]['raw_face'])
    cv2.imshow('align_face', results[0]['align_face'])
    cv2.waitKey()

