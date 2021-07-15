# -*- coding: utf-8 -*-
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from model_service.pytorch_model_service import PTServingBaseService

import time
# from metric.metrics_manager import MetricsManager
from torchvision.transforms import functional as F
import logging as log
import json
from PIL import Image

from pathlib import Path
import sys

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords

Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)

logger.info(torch.__version__)
logger.info(torchvision.__version__)


# class ImageClassificationService(PTServingBaseService):
@torch.no_grad()
class ImageClassificationService():
    def __init__(self, model_name, model_path):
        self.model = attempt_load(model_path, map_location='cpu')
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.imgsz = check_img_size(640, s=self.stride)  # check image size
        self.hao = {"person": 1, 'bus': 2, 'fire hydrant': 3, 'tie': 4}

        print("model already")

    def _preprocess(self, data):
        # https两种请求形式
        # 1. form-data文件格式的请求对应：data = {"请求key值":{"文件名":<文件io>}}
        # 2. json格式对应：data = json.loads("接口传入的json体")
        preprocessed_data = ''
        ## 多重字典
        for k, v in data.items():
            for file_name, file_content in v.items():
                # img = Image.open(file_content).convert("RGB")
                # preprocessed_data[k] = torch.unsqueeze(F.to_tensor(img), dim=0).to(self.device)
                preprocessed_data = file_content
        # return preprocessed_data
        return LoadImages(preprocessed_data, img_size=self.imgsz, stride=self.stride)

    def _inference(self, data):
        result = list()
        for path, img, im0s, vid_cap in data:
            img = torch.from_numpy(img).to('cpu')
            img = img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

            len(pred)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s.copy(), getattr(data, 'frame', 0)
                s += '%gx%g ' % img.shape[2:]  # print string
                boxes = []
                labels = []
                scorces = []
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)},"  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = self.names[c]

                        boxes.append([i.tolist() for i in xyxy])
                        labels.append(self.hao[label])
                        scorces.append(conf.tolist())
            print(result)
            result.append({'boxes': boxes, 'labels': labels, 'scorces': scorces})
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        #  if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
        #      MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        #   if self.model_name + '_LatencyInference' in MetricsManager.metrics:
        #       MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        #    if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
        #       MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        # data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        data.append({'latency_time': pre_time_in_ms + infer_in_ms + post_time_in_ms})
        return data


if __name__ == "__main__":
    ic = ImageClassificationService('yolov5s.pt', './yolov5s.pt')
    result = ic.inference({'input_img': {'input_img': '.'}})

    print(result)
