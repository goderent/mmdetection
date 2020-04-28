import argparse

import cv2
import torch

from mmdet.apis import inference_detector, init_detector, show_result
from mmcv.video.io import VideoWriter_fourcc


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'videosrc', help='video file src')
    parser.add_argument('output', help='output file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    videoCapture = cv2.VideoCapture(args.videosrc)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (720, 576), True)

    ret_val, img = videoCapture.read()
    #print('Press "Esc", "q" or "Q" to exit.')
    while ret_val:

        result = inference_detector(model, img)

        img2 = show_result(
            img, result, model.CLASSES, score_thr=args.score_thr, show=False)
        # show_result(
        #          img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1)
        videoWriter.write(img2)
        ret_val, img = videoCapture.read()

    videoWriter.release()
    videoCapture.release()


if __name__ == '__main__':
    main()
