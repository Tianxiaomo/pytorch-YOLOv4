import cv2

imgcv2 = cv2.imread("./giraffe.jpg")

imgcv2_ = cv2.resize(imgcv2, (512,512))
cv2.imwrite("giraffe_out.jpg",imgcv2_)


