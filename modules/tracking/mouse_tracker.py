from sort import Sort
import numpy as np
import os
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from tqdm import tqdm
import argparse
import cv2
import csv
from video_utils import add_padding, squarify_crop, save_cropped_video
import glob

class MouseTracker:
    def __init__(
        self,
        cfg_path: str = None,
        model_path: str = None,
        num_mice=1,
        output=None,
    ):
        self.cfg = get_cfg()
        self.load_existing_model(cfg_path, model_path)

        self.cfg.OUTPUT_DIR = output
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.TEST.DETECTIONS_PER_IMAGE = num_mice

        self.predictor = DefaultPredictor(self.cfg)
        print("using device:", self.cfg.MODEL.DEVICE)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def load_existing_model(self, cfg_path: str = None, model_path: str = None):
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.WEIGHTS = model_path
        print("starting model weights coming from:", model_path)
        return

    def save_tracking_data(self, file_type="csv"):
        ###
        print("######### save_tracking_data  #########")
        max_id = 0
        merged_track_ids = [] 
        for i, det, tracks in tqdm(self.all_tracks):
            for track in tracks:
                merged_track_ids.append(track.tolist() + [i])
                id = track.tolist()[-1]
                if id > max_id:
                    max_id = id
        all_trajectories = []
        for id in range(1, max_id + 1):
            trajectory = [t for t in merged_track_ids if t[4] == id]
            if len(trajectory) > 0:
                trajectory = sorted(trajectory, key=lambda x: x[-1])
                all_trajectories.append(trajectory)
        
        for data in all_trajectories:
            headers = ["top-x", "top-y", "bottom-x", "bottom-y", "id", "frame"]
            mouse_id = data[0][-2]
            # Write the data to a CSV file
            filename = os.path.join(self.cfg.OUTPUT_DIR, f"mouse_{str(mouse_id)}_data.csv")
            print('mouse id', mouse_id, 'tracking data saved to',filename )
            with open(filename, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(headers)  # Write the headers as the first row
                csv_writer.writerows(data)

        print("######### done save_tracking_data  #########")
        return

    def get_detections_video(self, video_path, num_frames=None):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set threshold for detection

        all_detections = []

        cap = cv2.VideoCapture(video_path)
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if num_frames:
            num_frames = min(num_frames, max_frames)
        else:
            num_frames = max_frames

        for f in tqdm(range(num_frames)):
            ___, frame = cap.read()
            try:
                outputs = self.predictor(frame)
                detections = (
                    outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
                )  # np.array(detections)
                scores = outputs["instances"].to("cpu").scores.numpy()
                detection_scores = np.concatenate(
                    [detections, np.expand_dims(scores, axis=1)], axis=1
                )
                all_detections.append(detection_scores)
            except:
                all_detections.append(np.empty((0, 5)))

        return all_detections

    def track_video(
        self, video_path, visulize=False, action_cam=False, output_path=None
    ):
        mot_tracker = Sort(max_age=25, min_hits=10, iou_threshold=0.3)  # max_age=50, min_hits=15, iou_threshold=0.5)
        video_path = video_path
        print("### analyzing video ###")
        detections = self.get_detections_video(video_path=video_path)
        self.all_tracks = []
        for i, det in tqdm(enumerate(detections)):
            tracks = mot_tracker.update(det)
            self.all_tracks.append([i, det.astype(int), tracks.astype(int)])
        
        self.save_tracking_data()
        
        if visulize:
            self.visualize_tracking(video_path, output_path)
            
        if action_cam:
            self.visulize_action_cams(video_path, output_path)

    def visualize_tracking(self, video_path, output_path):
        video = cv2.VideoCapture(video_path)

        frame_width = int(video.get(3))
        frame_height = int(video.get(4))

        size = (frame_width, frame_height)
        visulization_path = os.path.join(
            output_path, video_path.split(os.sep)[-1][:-4] + "_tracking_visulized.avi"
        )
        result = cv2.VideoWriter(
            visulization_path, cv2.VideoWriter_fourcc(*"MJPG"), 15, size
        )

        print("### visulizing_video ###")
        for i, det, tracks in tqdm(self.all_tracks):
            ret, frame = video.read()

            for t in range(det.shape[0]):
                x1 = det[t][0]
                y1 = det[t][1]
                x2 = det[t][2]
                y2 = det[t][3]
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 10, 255), 1)

            for t in range(tracks.shape[0]):
                x1 = tracks[t][0]
                y1 = tracks[t][1]
                x2 = tracks[t][2]
                y2 = tracks[t][3]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)

                id = str(tracks[t][4])
                cv2.putText(
                    frame,
                    id,
                    (x1, y2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                    1,
                )

            result.write(frame)

        video.release()
        result.release()

        cv2.destroyAllWindows()

    def visulize_action_cams(self, video_path, output_path):
        max_id = 0
        all_tracks = []
        for i, det, tracks in tqdm(self.all_tracks):
            for track in tracks:
                all_tracks.append(track.tolist() + [i])
                id = track.tolist()[-1]
                if id > max_id:
                    max_id = id

        all_trajectories = []
        for id in range(1, max_id + 1):
            trajectory = [t for t in all_tracks if t[4] == id]
            trajectory = sorted(trajectory, key=lambda x: x[-1])
            all_trajectories.append(trajectory)

        for trajectory in all_trajectories:
            if len(trajectory) > 30:
                print("trajectory for id:", trajectory[0][-2])
                save_cropped_video(trajectory, video_path, output_path)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default="/nfs/turbo/justincj-turbo/kaulg/behavior-toolbox/VH4_ZT16_2022-07-20_20-00-22_subclip_1.mp4",
        help="Path to the input video",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="output_tracking_final",
        help="Path to save the output",
    )
    parser.add_argument(
        "-n",
        "--num_mice",
        type=int,
        default=1,
        help="Path to save the output",
    )
    parser.add_argument(
        "-m",
        "--model_weights",
        type=str,
        default="/nfs/turbo/justincj-turbo/kaulg/behavior-toolbox/exparaments/mgen_hp_search/train_ratio_0.65/lr_0.001_wd_0.001/training/model_final.pth",
        help="Path to the model weights",
    )
    parser.add_argument(
        "-c",
        "--model_config",
        type=str,
        default="/nfs/turbo/justincj-turbo/kaulg/behavior-toolbox/exparaments/mgen_hp_search/train_ratio_0.65/lr_0.001_wd_0.001/training/config.yaml",
        help="Path to the model configuration",
    )

    parser.add_argument('--visulize', action='store_true',help='Percentage of datapoints to use for training (default: 0.8)')
    parser.add_argument('--action_cam', action='store_true',help='Percentage of datapoints to use for training (default: 0.8)')

    args = parser.parse_args()

    tracker = MouseTracker(
        cfg_path=args.model_config,
        model_path=args.model_weights,
        num_mice=args.num_mice,
        output=args.output_path,
    )

    tracker.track_video(
        video_path=args.video_path,
        visulize=args.visulize,
        action_cam=args.action_cam,
        output_path=args.output_path,
    )
    