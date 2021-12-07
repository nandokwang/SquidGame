import torch
import cv2
import time
import argparse

import pytesseract
from pytesseract import Output

import posenet
from sg_tracker import *

pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    # cap = cv2.VideoCapture('example.mp4')
    tracker = Sort()
    cap = cv2.VideoCapture('./video/demo_dance_3p_number_tag_450p.mp4')
    ocr_thresh = 0.1
    ocr_options = "outputbase digits"
    
    ocr_unique_number_hard = [[147, 200, 193, 220, '001'], # Delete later
                                [375, 171, 416, 242, '002'], 
                                [672, 188, 705, 209, '003']]

    start = time.time()
    frame_count = 0
    num_frames = 0
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)
        
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale


        bbox_for_tracker = []
        for pi in range(len(pose_scores)):

            idxs = [5, 6, 11, 12] # sholders, hips 나중에 생각해보기(SORT bbox input)
            
            x = keypoint_coords[pi, idxs, 1]
            y = keypoint_coords[pi, idxs, 0] # (10, 17, 2)

            x1 = min(x)
            x2 = max(x)
            y1 = min(y)
            y2 = max(y)
            
            bbox_for_tracker.append([x1, y1, x2, y2, 0.5])

        track_bbs_ids = tracker.update(np.array(bbox_for_tracker[:3]))

        if frame_count == 0:
            ocr_unique_number = []
            # crop_img = img[y:y+h, x:x+w]
            for bbox in bbox_for_tracker[:3]:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cropped_imgs = display_image[y1:y2, x1:x2, :]
                ocr_results = pytesseract.image_to_data(cropped_imgs, output_type=Output.DICT, config=ocr_options)
                texts = ocr_results['text']
                confs = ocr_results['conf']
                xs = ocr_results['left']
                ys = ocr_results['top']
                ws = ocr_results['width']
                hs = ocr_results['height']

                for conf, text, x, y, w, h in zip(confs, texts, xs, ys, ws, hs):
                    if float(conf) > ocr_thresh:
                        # cropped_imgs = cv2.rectangle(cropped_imgs, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # cropped_imgs = cv2.putText(cropped_imgs, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        ocr_unique_number.append([x1+x, y1+y, x1+x+w, y1+y+h, text])
                        # print(conf, text, [x, y, w ,h])
                    

        elif frame_count == 1:
            print('Frame 1')
            print(track_bbs_ids)
            print(np.array(ocr_unique_number_hard))
            
            # number_tag = iou(track_bbs_ids, ocr_unique_number_hard)
            # [1: '001', 2: '002', 3:'003']
            


        # print(len(track_bbs_ids))
        for track_bbs_id in track_bbs_ids:
            x1 = int(track_bbs_id[0])
            y1 = int(track_bbs_id[1])
            x2 = int(track_bbs_id[2])
            y2 = int(track_bbs_id[3])
            unique_id = str(int(track_bbs_id[4]))

            t_size = cv2.getTextSize(unique_id, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            cv2.putText(display_image, unique_id, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 5, [0,0,0], 5)

        # Draw result
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        num_frames += 1

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()