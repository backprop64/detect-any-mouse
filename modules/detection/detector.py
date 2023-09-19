import torch
import os
import argparse

from detector_dataset import DetectorDataset
from detector_evals import eval

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader

pretrained_weights = {
    "coco_detector_50": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "coco_detector_101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "coco_mask_50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "coco_mask_101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "LVIS_mask_50": "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "LVIS_mask_101": "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
}


class Detector:
    def __init__(
        self,
        cfg_path: str = None,
        model_path: str = None,
        model_type: str = None,
        output="/mouse_detector_output",
    ):
        self.cfg = get_cfg()

        if model_type and not (cfg_path and model_path):
            self.create_new_detector(model_type)
        else:
            self.load_existing_model(cfg_path, model_path)
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 0 
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device:", self.cfg.MODEL.DEVICE)
        self.output_dir = output
        self.cfg.OUTPUT_DIR = output
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def save_config(self):
        output_file = os.path.join(self.cfg.OUTPUT_DIR, "config.yaml")
        with open(output_file, "w") as f:
            f.write(self.cfg.dump())  # save config to file

    def load_existing_model(self, cfg_path: str = None, model_path: str = None):
        self.cfg.merge_from_file(cfg_path)
        if model_path:
            self.cfg.MODEL.WEIGHTS = model_path
        print("starting model weights coming from:", model_path)
        return

    def create_new_detector(self, model_type: str = None):
        self.cfg.merge_from_file(
            model_zoo.get_config_file(pretrained_weights[model_type])
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            pretrained_weights[model_type]
        )
        print("starting model weights coming from:", self.cfg.MODEL.WEIGHTS)
        return

    def train_detector(
        self,
        dataset_name="train_dataset",
        learning_rate=1e-3,
        weight_decay=1e-3,
        iterations=300,
        checkpoint_every=100,
    ):
        self.cfg.OUTPUT_DIR = os.path.join(self.output_dir, "training")
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
    
        self.cfg.DATASETS.TRAIN = (dataset_name,)
        self.cfg.DATASETS.TEST =  (dataset_name,)
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.MAX_ITER = iterations
        self.cfg.SOLVER.STEPS = ()
        self.cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_every
        self.cfg.SOLVER.BASE_LR = learning_rate
        self.cfg.SOLVER.WEIGHT_DECAY = weight_decay
        self.cfg.SOLVER.WARMUP_ITERS = 100

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        self.trainer = DefaultTrainer(self.cfg)
        self.save_config()

        self.trainer.resume_or_load(resume=False)

        self.trainer.train()

        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.save_config()

    def evaluate_detector(
        self,
        datset_name="eval_dataset",
        detection_threshold = 0.7
    ):
        self.cfg.DATASETS.TEST = datset_name
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold  

        evaluator = COCOEvaluator(
            datset_name, self.cfg, False, output_dir=self.cfg.OUTPUT_DIR
        )

        predictor = DefaultPredictor(self.cfg)

        ## compute AP metrics
        dataset_dicts = DatasetCatalog.get(datset_name)
        evaluation_metrics = eval(predictor, dataset_dicts)
        
        ## compute AP metricsxsx
        data_loader = build_detection_test_loader(self.cfg, datset_name)
        coco_metrics = inference_on_dataset(predictor.model, data_loader, evaluator)

        evaluation_metrics.update(coco_metrics['bbox'])

        try:
            segm_dict = coco_metrics['segm']
            new_segm_dict = {'mask_' + key: value for key, value in segm_dict.items()}
            evaluation_metrics.update(new_segm_dict)
        except:
            print('only box predictions')
        
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~ Evaluation Summary ~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')

        for metric,score in evaluation_metrics.items():
            print(metric,score)

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create, train, and evaluate models single catagory object detection tasks")

    parser.add_argument('--output_path', type=str, help='Path to save the trained model')

    parser.add_argument('--dataset', type=str, default='none', help='Path to the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--weights', type=str, help='Path to the weights file')
    parser.add_argument('--model_type', type=str, help='Train a new model from scratch based on detection type: box, mask, or keypoint')

    parser.add_argument('--iterations', type=int, default=500, help='Number of iterations (default: 1000)')
    parser.add_argument('--checkpoint_every', type=int, default=100.0, help='Interval to save model checkpoints during training (default: 0.0, disabled)')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for the optimizer (default: 0.0)')

    parser.add_argument('--train_ratio', type=float, default=1.0, help='Percentage of datapoints to use for training (default: 0.8)')
    parser.add_argument('--test_ratio', type=float, default=0.0, help='Percentage of datapoints to use for testing (default: 0.2)')
    parser.add_argument('--train', action='store_true',help='Percentage of datapoints to use for training (default: 0.8)')
    parser.add_argument('--test',action='store_true', help='Percentage of datapoints to use for training (default: 0.8)')

    args = parser.parse_args()

    if args.dataset != "none":
        dataset = DetectorDataset(
            args.dataset,
            args.output_path,
        )

        dataset.make_train_test_splits(args.train_ratio, args.test_ratio)

    if args.config and args.weights:
        print("starting with initial model")
        mouse_detector = Detector(
            model_path=args.weights,
            cfg_path=args.config,
            output=args.output_path,
        )

    else:
        print("training model from scratch")
        mouse_detector = Detector(
            model_type=args.model_type,
            output=args.output_path,
        )

    if args.train:
        mouse_detector.train_detector(
            "train_split",
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            iterations=args.iterations,
            checkpoint_every=args.checkpoint_every,
        )
        
    mouse_detector.save_config()

    if args.test:
        mouse_detector.evaluate_detector("test_split")

