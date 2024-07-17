from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())

FIRST_NUMBER = {
    '3': "American Express",
    '4': "Visa",
    '5': "MasterCard",
    '6': "Discover Card"
}

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread(args["template"])
# 灰度通道
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 阈值处理
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
# 计算轮廓
ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETA_EXTERNAL, cv2.CHAIN_APPORX_SIMPLE)

cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show('img', img)
print(np.array(refCnts).shape)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

# 遍历每一个轮廓
digits = dict()
for i, c in enumerate(refCnts):
    # 计算外接矩形并且resize成合适大小
    x, y, w, h = cv2.boundingRect(c)
    roi = ref[y: y+h, x: x+w]
    roi = cv2.resize(roi, (57, 88))
    # 每个数字对应一个模板
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像 + 预处理
image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
minval, maxval = np.min(gradX), np.max(gradX)
gradX = (255 * ((gradX - minval))) / (maxval - minval)
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
cv_show('gradX', gradX)

# 通过闭操作将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)

# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需要把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)

# 再来一次闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh', thresh)

# 计算轮廓
thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)

locs = list()
for i, c in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)

    # 选择合适的区域
    if ar > 2.5 and ar < 4.0:
        if w > 40 and w < 55 and h > 10 and h < 20:
            locs.append((x, y, w, h))

locs.sort(key=lambda x: x[0])

output = []
for i, gX, gY, gW, gH in enumerate(locs):
    group_output = []
    # 根据坐标提取每一个组
    group = gray[gY - 5: gY + gH + 5, gX - 5: gX + gW + 5] # +/-5 是稍微往外扩一些
    cv_show('group', group)

    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)

    # 计算每一组的轮廓
    group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        x, y, w, h = cv2.boundingRect(c)
        roi = group[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []
        for digit, digitROI in digits.items():
            res = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            _, score, _ = cv2.minMaxLoc(res)
            scores.append(score)

        # 得到最适合的
        group_output.append(str(np.argmax(scores)))

    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(group_output), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.extend(group_output)

# 打印
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv_show("image", image)
