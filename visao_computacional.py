"""
Exemplo abaixo captura uma imagem
"""
import cv2

capture = cv2.VideoCapture(1)
cv2.namedWindow('Janela', cv2.WINDOW_AUTOSIZE)

ret, frame = capture.read()

cv2.imshow('Janela', frame)

while true:
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
