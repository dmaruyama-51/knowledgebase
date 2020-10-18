#-------------
# 環境合わせ
#-------------
# install dependencies: (use cu101 because colab has CUDA 10.1)
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
 
# install detectron2:
!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

#-------------
# データ取得→COCOフォーマットに変換
#-------------
!git clone https://github.com/Shenggan/BCCD_Dataset.git

!git clone https://github.com/yukkyo/voc2coco.git
 
%cd voc2coco/
 
!python voc2coco.py --ann_dir sample/Annotations \
--ann_ids sample/dataset_ids/train.txt \
--labels sample/labels.txt \
--output sample/bccd_train_cocoformat.json \
--ext xml
 
!python voc2coco.py --ann_dir sample/Annotations \
--ann_ids sample/dataset_ids/test.txt \
--labels sample/labels.txt \
--output sample/bccd_test_cocoformat.json \
--ext xml

#-------------
# データセットの準備
#-------------

def get_bccd_dicts(img_dir, json_path):
  with open(json_path) as f:
      imgs_anns = json.load(f)
  images = imgs_anns["images"]
  annotations = imgs_anns["annotations"]
  dataset_dicts = []
  ids = []
  for im in images:
    record = {}
 
    filename = os.path.join(img_dir, im["file_name"])
    record["file_name"] = filename
    record["height"] = im["height"]
    record["width"] = im["width"]
    record["image_id"] = im["id"]
 
    dataset_dicts.append(record)
    ids.append(im["id"])
  
  annos_dict = {}
  for id in ids:
    objs = []
    for anno in annotations:
      if anno["image_id"] == id:
        obj = {
            "bbox": anno["bbox"],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": anno["category_id"],
            "segmentation": anno["segmentation"]
        }
        objs.append(obj)
    annos_dict[id] = objs
 
  for dic in dataset_dicts:
    for id in annos_dict.keys():
      if dic["image_id"] == id:
        dic["annotations"] = annos_dict[id]
 
  return dataset_dicts
 
img_dir = "BCCD_Dataset/BCCD/JPEGImages/"
train_json_path = "voc2coco/sample/bccd_train_cocoformat.json"
test_json_path = "voc2coco/sample/bccd_test_cocoformat.json"
 
train_dataset_dicts = get_bccd_dicts(img_dir, train_json_path)
test_dataset_dicts = get_bccd_dicts(img_dir, test_json_path)

from detectron2.data.datasets import register_coco_instances
register_coco_instances("bccd_train", {}, train_json_path, img_dir)
register_coco_instances("bccd_test", {}, test_json_path, img_dir)

# 可視化
bccd_train = MetadataCatalog.get("bccd_train")
 
for d in random.sample(train_dataset_dicts, 3):
  img = cv2.imread(d["file_name"])
  visualizer = Visualizer(img[:, :, ::-1], metadata=bccd_train, scale=0.5)
  out = visualizer.draw_dataset_dict(d)
  cv2_imshow(out.get_image()[:, :, ::-1])


#-------------
# 学習
#-------------

from detectron2.config import get_cfg

cfg = get_cfg()

# Faster RCNN
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   

# Dataset（ "," がないとエラー吐かれる）
cfg.DATASETS.TRAIN = ("bccd_train",)
cfg.DATASETS.TEST = ("bccd_test", )

# ハイパーパラメータ
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001  
cfg.SOLVER.MAX_ITER = 500

from detectron2.engine import DefaultTrainer
 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


#-------------
# 評価
#-------------

# 結果の確認
%load_ext tensorboard
 
%tensorboard --logdir output

# 推論結果の可視化
# モデルに学習済み重みを読み込ませる
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # faster rcnn
cfg.DATASETS.TEST = ("bccd_test", )
predictor = DefaultPredictor(cfg)
 
for d in random.sample(test_dataset_dicts, 3): 
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=bccd_test, 
                   scale=0.8,  #何倍に画像を表示するか
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


# COCO API で評価
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("bccd_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "bccd_test")
inference_on_dataset(trainer.model, val_loader, evaluator)


 