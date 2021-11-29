# import easyocr
import cv2
import numpy as np

import pytesseract
from pytesseract import Output


pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'


image_file = './ocr_imgs/ocr_example.png'
# # image_file = './ocr_imgs/apple.jpg'
# # image_file = './ocr_imgs/squid_game_number_tag2.jpg'
# # image_file = './ocr_imgs/456.jpg'
img = cv2.imread(image_file)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

options = "" # This one will detect also character as well
options = "outputbase digits" # This option only detect number

# text = pytesseract.image_to_string(img, config=options)
# print('Output:', text)

# boxes = pytesseract.image_to_boxes(img)
# print(boxes)

# ocr_results = pytesseract.image_to_data(img, output_type=Output.DICT, config=options)
# # print(ocr_results.keys())

# text = ocr_results['text']
# print(text)
# idx = [n for n, i in enumerate(text) if len(i) > 0][0]

# text = text[idx]
# x = ocr_results['left'][idx]
# y = ocr_results['top'][idx]
# w = ocr_results['width'][idx]
# h = ocr_results['height'][idx]

# img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# img = cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)


# 69번 탈락
options = "outputbase digits"
cap = cv2.VideoCapture('number_tag.mp4')


fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (640,480))


while cap.isOpened():
	ret, frame = cap.read()
	ocr_results = pytesseract.image_to_data(frame, output_type=Output.DICT, config=options)
	text = ocr_results['text']
	# imgchar = pytesseract.image_to_string(frame)
	# print('output', imgchar)
	# idx = [n for n, i in enumerate(text) if len(i) > 0][0]

	# text = text[idx]
	# x = ocr_results['left'][idx]
	# y = ocr_results['top'][idx]
	# w = ocr_results['width'][idx]
	# h = ocr_results['height'][idx]

	# frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# frame = cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
	frame = cv2.rectangle(frame, (100, 100), (200, 200), (0, 0, 255), 2)
	out.write(frame)

	cv2.imshow('frame', frame)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()