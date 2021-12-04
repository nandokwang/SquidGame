### Input process output로 작성한 psudo code

```python
img_inputs = cv2.VideoCapture('asdf.mp4') # 카메라 직캠
tracker = Sort(params)

for i, img_input in enumerate(img_inputs):
  keypoint_coord = keypoint_model(img_input)
  num_of_participants = len(keypoint_coord)
  
  # 1. 첫프레임
  if i == 0:
     cropped_imgs = crop(어깨좌표 2개, 골반좌표 2개) # 사람수만큼
     for cropped_img in cropped_imgs:
         번호표_text, box_coord_for_text = OCR(cropped_img) # 사람수만큼(합쳐서)
         번호표.append(번호표_text + box_coord_for_text)

  elif i == 1:
      # IoU(번호표bbox, 키포인트후처리bbox)
      # euclidean(번호표중앙좌표, 몸통중앙좌표 or 후처리bbox중앙좌표)
      # 양자 택일
      matched_id_and_number = IoU(det_bboxes_ids, 번호표)
      # [x1, y1, x2, y2, id], [box_coord_for_text, 번호표_text]
      # matched_id_and_number == DICT {sort_id: 번호표}

  bboxes = get_bbox(keypoint_coord) # [[x1, y1, x2, y2], ... [x1, y1, x2, y2]]
  det_bboxes_ids = tracker.update(bboxes) # [[x1, y1, x2, y2, sort_id], ... []]

  """
  miss 타겟 mapping module
  """
  # 한명의 정보만 잃어버렸을 때
  if det_bboxes_ids.shape[0] == num_of_participants-1:
      missing_target = set(matched_id_and_number.keys()) - set(det_bboxes_ids.T[4])
      # sort에서 이전 프레임의 정보를 가져와야서 비교 후 할당

  # 두명이상 정보를 잃어버렸을때
  elif det_bboxes_ids.shape[0] < num_of_participants.shape[0]-1:
      # challenge_1: 잃어버리기 전 최근 좌표를 기록해두고 가까운 값을 채택
      # challenge_2: 잃어버리기 전 최근의 dx, dy 값으로 기울기를 지속적으로 기록해두고 (이동평균) 다시 정보를 얻었을 때 기울기로 예측 

  """
  움직임 여부 check module
  """ 
  if 사망하지 않은 참가자 중에서:
  movement = tracker.movement(keypoint_coord)
      # 전 프레임의 17개의 좌표와의 차이를 구함
      # 17개의 좌표의 차이가 일정치 이상 차이가 난다면 움직임으로 플래그 변경
      # 움직임의 이동평균으로 계산
      # output: [{sort_id: [last_coord, movement, is_alive, goal]}, ..., sort_id: []]

  """
  처형 module
  """
	if 움직임이 있는 참가자의 경우:
      excute(movement)
      sound(f'{matched_id_and_number[sort_id]}번 참가자 탈락')
    	# value가 dead인 참가자의 탈락 sound 재생
      # output: [{sort_id: [last_coord, movement, is_alive, goal]}, ..., sort_id: []]
      
  """
  통과자 module (밑에다가 라인 하나그려놓고, 넘어가면, excute() 안돌림)
  """
  if 사망하지 않은 참가자 중에서:
      if any(foot_coord) < y축라인값:
          cv2.puttext(f'{matched_id_and_number[sort_id]}번 참가자 통과')
  		# output: [{sort_id: [번호표, last_coord, movement, is_alive, goal]}, ..., sort_id: []]
      
      
"""
Memo
Camera는 몸통에 달아야함(지속적으로 촬영 필요)
Movement 모듈과 excute 모듈은 무궁화꽃이피었습니다 음성이 pause 상태일때만 돌리기
Goal: end 라인 설정
"""
```

