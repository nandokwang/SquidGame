import cv2
import numpy as np

import posenet.constants

import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, source_img, scale


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def draw_skel_and_kp(
        status, img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):

    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(0, 0, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(0, 0, 255), thickness=10)
    
    for k, v in status.items():
        x1 = int(v[0][-1][0][1])
        y1 = int(v[0][-1][0][0])
        if v[2] == 0:
            if v[1] == 1:
                cv2.putText(out_img, 'Moving', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 5, [0,0,255], 5)
            else:
                cv2.putText(out_img, 'Stopping', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 5, [255,0,0], 5)
        else:
            cv2.putText(out_img, f'No.{k} Tal-Rak', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 3, [0,0,255], 3)

    return out_img


def make_tracking_bbox(keypoint_coords):
    """
    bbox_for_tracker: 
    keypoint_coords: keypoint coordinates
    """
    bbox_for_tracker = []
    for pi in range(len(keypoint_coords)):

        idxs = [5, 6, 11, 12] # left, right shoulders and hips

        x = keypoint_coords[pi, idxs, 1]
        y = keypoint_coords[pi, idxs, 0] # (10, 17, 2)

        x1 = min(x)
        x2 = max(x)
        y1 = min(y)
        y2 = max(y)
        
        bbox_for_tracker.append([x1, y1, x2, y2, 0.5])

    return bbox_for_tracker


def start_game(bbox_for_tracker, display_image, ocr_thresh):
    ocr_options = "outputbase digits"
    ocr_unique_number = []
    for i, bbox in enumerate(bbox_for_tracker):
        crop_weight = 20

        x1 = int(bbox[0]) - crop_weight
        y1 = int(bbox[1])
        x2 = int(bbox[2]) + crop_weight
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
                cropped_imgs = cv2.rectangle(cropped_imgs, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped_imgs = cv2.putText(cropped_imgs, text, (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                ocr_unique_number.append([x1+x, y1+y, x1+x+w, y1+y+h, int(text)])
    
    return ocr_unique_number


def viz_participants_number(track_bbs_ids, display_image, match_tag, frame_count):
    for track_bbs_id in track_bbs_ids:
        x1 = int(track_bbs_id[0])
        y1 = int(track_bbs_id[1])
        x2 = int(track_bbs_id[2])
        y2 = int(track_bbs_id[3])
        unique_id = str(int(track_bbs_id[4]))

        t_size = cv2.getTextSize(unique_id, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]

        if frame_count == 0:
            cv2.putText(display_image, unique_id, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 5, [0,0,0], 5)
        else:
            cv2.putText(display_image, match_tag[int(unique_id)], (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], 2)

    return display_image


def draw_passing_line(display_image, is_passed_threshold):
    display_image = cv2.line(display_image, 
                    (0, is_passed_threshold), 
                    (display_image.shape[1], is_passed_threshold), (0, 0, 255), 5)
    
    return display_image
