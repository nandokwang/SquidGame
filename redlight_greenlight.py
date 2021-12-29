import torch
import cv2
import time
import argparse

import posenet
from sg_tracker import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    tracker = Sort()
    cap = cv2.VideoCapture('./video/demo_dance_3p_number_tag_1080p.mp4') # demo_dance_3p_number_tag_1080p
    ocr_thresh = 0.1
    
    start = time.time()
    frame_count = 0
    num_frames = 0
    status = dict()
    match_kp_sortid = dict()
    match_tag = dict()
    movement_threshold = 30
    is_passed_threshold = 950

    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (640,480))

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

        # Extract bbox coords from shoulder and hips(for OCR+SORT)
        bbox_for_tracker = posenet.make_tracking_bbox(keypoint_coords)

        # SORT Tracking
        track_bbs_ids = tracker.update(np.array(bbox_for_tracker[:3]))

        # Get matching idx btw bbox from keypoint and bbox from SORT
        kp_match_iou = get_iou_idx(track_bbs_ids[:, :4], np.array(bbox_for_tracker[:3])[:, :4])

        for i, kmi in enumerate(kp_match_iou):
            match_kp_sortid[int(track_bbs_ids[i,4])] = np.array(keypoint_coords[kmi])
        
        # When frame count 0, Crop each participants -> run OCR and return OCR information
        if frame_count == 0:
            ocr_unique_number = posenet.start_game(bbox_for_tracker[:3], display_image, ocr_thresh)

        # When frame count 1, Matching IOU btw OCR bbox and SORT bbox coords
        elif frame_count == 1:
            ocr_match_iou = get_iou_idx(track_bbs_ids[:, :4], np.array(ocr_unique_number)[:, :4])

            for i, mi in enumerate(ocr_match_iou):
                match_tag[int(track_bbs_ids[i,4])] = str(mi+1).zfill(3)
                status[int(track_bbs_ids[i,4])] = [np.expand_dims(keypoint_coords[i,:,:], 0), 0, 0, 0]

        # Visualize Participants number
        display_image = posenet.viz_participants_number(track_bbs_ids, display_image, match_tag, frame_count)
        
        # Visualize passing line
        display_image = posenet.draw_passing_line(display_image, is_passed_threshold)

        # Update status
        if frame_count > 1:
            status = tracker.update_coords_history(status, match_kp_sortid)
            status = tracker.track_movement(status, threshold=movement_threshold)
            status = tracker.passed_participants(status, threshold = is_passed_threshold)

        if num_frames > 20:
            status = tracker.excute_drop_off(status)

        # Draw Keypoint results, and status(if moving, failed, passed)
        overlay_image = posenet.draw_skel_and_kp(
            status, display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.01)


        overlay_image = cv2.resize(overlay_image, dsize=(1000, 640), interpolation=cv2.INTER_AREA)
        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        num_frames += 1

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()