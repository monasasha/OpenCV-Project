import math
import time

import numpy as np
import pytesseract
import cv2
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

MIN_H_LINES = 40
MAX_ANGLE_DEVIATION = 40


def show(img, title=''):
    plt.title(title)
    plt.imshow(img)
    plt.show()


def ocr(img):
    result = np.full(img.shape, 255, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 12, dst=gray)
    extract = cv2.merge([gray, gray, gray])
    # show(img, "extract")
    gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((1, 1), np.uint8), iterations=4)
    gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, np.ones((1, 1), np.uint8), iterations=4)
    contours = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if (area > 130) and h > w:
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 3)
            result[y:y + h, x:x + w] = extract[y:y + h, x:x + w]
    invert = 255 - result
    show(invert, "invert")
    data = pytesseract.image_to_string(invert, lang='eng',
                                       config='--psm 13 --oem 3')
    if data != '':
        data = data.split()
        data = ''.join(data)
        print("number is about", data)
    else:
        print("can't recognize!")
    # time.sleep(2)


def rotate_image(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def sobel(img):
    img_sobel = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    img_sobel = img_sobel.astype('uint8')
    img_sobel = cv2.addWeighted(img, 0.95, img_sobel, 0.05, 0)

    return img_sobel


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def line_analyze(img, lines):
    px_eps, h_lines_count, counter, angle = 10, 0, 0, 0
    main_lines = []
    for i in range(len(lines)):
        for j in range(int(len(lines) / 2)):
            ix1, iy1, ix2, iy2 = lines[i][0]
            jx1, jy1, jx2, jy2 = lines[j][0]
            angle += math.atan2(abs(iy1 - iy2), abs(ix1 - ix2)) * 180 / math.pi
            # lines almost in the same place
            if abs(jy1 - iy1) < px_eps and abs(jy2 - iy2) < px_eps and i != j:
                # exclude image edges
                if iy1 < 10 or iy2 > img.shape[0] - 10:
                    continue
                main_lines.append([ix1, iy1, ix2, iy2])
                cv2.line(img, (ix1, iy1), (ix2, iy2), (0, 0, 250), 5)
            # line that almost horizontal
            if abs(iy1 - iy2) < px_eps:
                cv2.line(img, (ix1, iy1), (ix2, iy2), (0, 255, 250), 5)
                h_lines_count += 1
            counter += 1

    return h_lines_count > MIN_H_LINES, angle / counter, main_lines


def get_mean_line_cord(lines):
    mean_ = [0.0, 0.0, 0.0, 0.0]
    for e in lines:
        mean_ = np.add(mean_, e)
    mean_ /= np.full(4, len(lines))
    xx = int(mean_[0])
    yy = int(0.5 * mean_[1] + mean_[3])

    return xx, yy


def find_number(img, orig):
    h, w = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 250, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 120, 255, apertureSize=3)
    # searching lines on the part of the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=0.2 * w, maxLineGap=30)
    reliable, mean_angle, main_lines = line_analyze(img, lines)
    # if lines wasn't enough - there:
    if not reliable:
        if mean_angle > MAX_ANGLE_DEVIATION:
            # check for rotation
            image_rotated = rotate_image(orig, mean_angle)
            return [image_rotated]
        else:
            # check for another splitting
            xx, yy = get_mean_line_cord(main_lines)
            if xx / w > yy / h:
                img1 = orig[0:h, 0:xx]
                img2 = orig[0:h, xx:w]
            else:
                img1 = orig[0:yy, 0:w]
                img2 = orig[yy:h, 0:w]
            return img1, img2
    else:
        return [orig]


def prepare_image(img):
    prepared = sobel(img)
    prepared = cv2.morphologyEx(prepared, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    prepared = apply_brightness_contrast(prepared, 60, 30)

    return prepared


def do_parts(orig):
    # splitting images along the contours into parts
    parts = []
    prepared = prepare_image(orig)
    gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if contours != 0:
        for c in contours:
            if cv2.contourArea(c) > 5000:
                approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
                cv2.drawContours(prepared, [approx], -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(approx)
                ROI = prepared[y:y + h, x:x + w]
                ROI_orig = orig[y:y + h, x:x + w]
                images = find_number(ROI, ROI_orig)
                parts.extend(images)
    print("numbers on image: {}".format(len(parts)))
    return parts


if __name__ == '__main__':
    image = cv2.imread('1.jpg')
    if image is None:
        exit(0)
    numbers_img = do_parts(image)
    for img_part in numbers_img:
        ocr(img_part)
