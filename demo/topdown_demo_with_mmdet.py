# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmpose/blob/main/demo/topdown_demo_with_mmdet.py
# https://mmpose.readthedocs.io/en/latest/demos.html
'''
[w/ mmdet]
mmdet model zoo - https://mmdetection.readthedocs.io/en/latest/model_zoo.html

python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} \
    [--output-root ${OUTPUT_DIR}] [--save-predictions] \
    [--show] [--draw-heatmap] [--device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR}] [--kpt-thr ${KPT_SCORE_THR}]
EXAMPLE
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth \
    --input /mnt/sdc/glee623/CODES/base_files/demo_imgs/s_01_act_02_subact_01_ca_01_000015.jpg --show --draw-heatmap \
    --output-root /mnt/sdc/glee623/CODES/output_ECE/

python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth \
    --input /mnt/sdc/glee623/projects/ContextAware-PoseFormer_old/ContextPose/data/h36m_validation.pkl --save-predictions --draw-heatmap \
    --output-root /mnt/sdc/glee623/dataset/Human36M_vitpose_output/
    --device 2

w/o mmdet
python demo/image_demo.py \
    ${IMG_FILE} ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --out-file ${OUTPUT_FILE} \
    [--device ${GPU_ID or CPU}] \
    [--draw_heatmap]
'''
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
import pickle

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log
from tqdm import tqdm

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:3', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'
    os.makedirs(os.path.join(args.output_root,args.pose_checkpoint.split("/")[-1][:-4]),exist_ok=True)

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if args.input == 'webcam':
        input_type = 'webcam'
        
    else:
        try:
            input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
        except:
            input_type = 'folder'
            if args.input.endswith('pkl'):
                input_type = 'pickle'

    if input_type == 'image':
        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type == 'folder':
        file_lists = os.listdir(args.input)
        print("saving img results ...")
        for file_name in tqdm(file_lists):
            img = os.path.join(args.input,file_name)
            pred_instances = process_one_image(args, img, detector,
                                            pose_estimator, visualizer)

            if args.save_predictions:
                pred_instances_list = split_instances(pred_instances)

            if output_file:
                img_vis = visualizer.get_image()
                output_file = os.path.join(args.output_root, args.pose_checkpoint.split("/")[-1][:-4], file_name)
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

            if args.save_predictions:
                args.pred_save_path = f'{args.output_root}{args.pose_checkpoint.split("/")[-1][:-4]}/results_{file_name}.json'
                with open(args.pred_save_path, 'w') as f:
                    json.dump(
                        dict(
                            meta_info=pose_estimator.dataset_meta,
                            instance_info=pred_instances_list),
                        f,
                        indent='\t')
    elif input_type == 'pickle':
        target_path = "/mnt/sdc/glee623/dataset/Human36M/images/"
        target_file = pickle.load(file=open(args.input,'rb'))
        print("saving img results ...")
        for data in tqdm(target_file):
            file_name = data['image']
            img = os.path.join(target_path,file_name)
            pred_instances = process_one_image(args, img, detector,
                                            pose_estimator, visualizer)

            if args.save_predictions:
                pred_instances_list = split_instances(pred_instances)

            if output_file:
                img_vis = visualizer.get_image()
                
                # os.makedirs(os.path.join(args.output_root, args.pose_checkpoint.split("/")[-1][:-4], "img"),exist_ok=True)
                output_file = os.path.join(args.output_root, args.pose_checkpoint.split("/")[-1][:-4], "img", file_name)
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

            if args.save_predictions:
                os.makedirs(f'{args.output_root}{args.pose_checkpoint.split("/")[-1][:-4]}/json/{file_name.split("/")[0]}',exist_ok=True)
                args.pred_save_path = f'{args.output_root}{args.pose_checkpoint.split("/")[-1][:-4]}/json/{file_name[:-4]}.json'
                with open(args.pred_save_path, 'w') as f:
                    json.dump(
                        dict(
                            meta_info=pose_estimator.dataset_meta,                  
                            instance_info=pred_instances_list),
                        f,
                        indent='\t')
    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions and input_type != "folder":
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
