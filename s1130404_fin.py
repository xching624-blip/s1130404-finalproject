import cv2
import os
import re
import pytesseract

# Tesseract 安裝路徑
pytesseract.pytesseract.tesseract_cmd = \
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 全域結果
plate_counter = 1
all_results = []

# 車牌格式驗證
def is_valid_plate(text):
    return bool(
        re.fullmatch(
            r'[A-Z]{3}-\d{4}',
            text
        )
    )

# OCR文字修正
def fix_plate_text(text):

    text = text.upper()

    text = text.replace('I', '1')
    text = text.replace('O', '0')
    text = text.replace('Q', '0')

    text = re.sub(
        r'[^A-Z0-9]',
        '',
        text
    )

    letters = ''.join(
        c for c in text
        if c.isalpha()
    )

    digits = ''.join(
        c for c in text
        if c.isdigit()
    )

    if len(letters) >= 3:
        letters = letters[-3:]

    if len(digits) >= 4:
        digits = digits[:4]

    if len(letters) == 3 and len(digits) == 4:
        return f"{letters}-{digits}"

    return ""

# 車牌偵測與OCR
def detect_and_extract_roi(
        image_path,
        cascade_path="haar_carplate.xml"):

    global plate_counter
    global all_results

    if not os.path.exists(image_path):
        print(f"⚠️ 找不到影像：{image_path}")
        return

    if not os.path.exists(cascade_path):
        print(f"⚠️ 找不到模型：{cascade_path}")
        return

    car_cascade = cv2.CascadeClassifier(
        cascade_path
    )

    img = cv2.imread(image_path)

    if img is None:
        print("⚠️ 影像讀取失敗")
        return

    target_width = 800

    h, w = img.shape[:2]

    ratio = target_width / w
    target_height = int(h * ratio)

    img_resized = cv2.resize(
        img,
        (target_width, target_height)
    )

    gray = cv2.cvtColor(
        img_resized,
        cv2.COLOR_BGR2GRAY
    )

    # Haar車牌偵測
    plates = car_cascade.detectMultiScale(
        gray,
        scaleFactor=1.07,
        minNeighbors=7,
        minSize=(60, 20),
        maxSize=(350, 120)
    )

    # 幾何條件過濾
    valid_plates = []

    for (x, y, w, h) in plates:

        aspect_ratio = w / h
        center_y = y + h / 2

        if (
                2.0 <= aspect_ratio <= 4.0
                and center_y >= target_height * 0.35
        ):
            valid_plates.append(
                (x, y, w, h)
            )

    # NMS去除重疊框
    valid_plates = sorted(
        valid_plates,
        key=lambda b: b[2] * b[3],
        reverse=True
    )

    final_plates = []

    for (x, y, w, h) in valid_plates:

        overlap = False

        for (fx, fy, fw, fh) in final_plates:

            inter_x = max(x, fx)
            inter_y = max(y, fy)

            inter_w = min(
                x + w,
                fx + fw
            ) - inter_x

            inter_h = min(
                y + h,
                fy + fh
            ) - inter_y

            if inter_w > 0 and inter_h > 0:

                inter_area = (
                        inter_w *
                        inter_h
                )

                box_area = w * h

                if (
                        inter_area /
                        box_area
                        > 0.3
                ):
                    overlap = True
                    break

        if not overlap:
            final_plates.append(
                (x, y, w, h)
            )

    # 最多保留6面車牌
    final_plates = final_plates[:6]

    # OCR辨識
    if len(final_plates) == 0:

        print(
            f"【{image_path}】偵測車牌失敗"
        )
        return

    print(
        f"成功在 {image_path} "
        f"鎖定 {len(final_plates)} 面車牌！"
    )

    for (x, y, w, h) in final_plates:

        offset_x = int(w * 0.12)
        expand_w = int(w * 0.18)

        adjusted_x = max(
            0,
            x - offset_x
        )

        adjusted_w = w + expand_w

        end_x = min(
            target_width,
            adjusted_x + adjusted_w
        )

        end_y = min(
            target_height,
            y + h
        )

        roi_img = img_resized[
            y:end_y,
            adjusted_x:end_x
        ]

        roi_large = cv2.resize(
            roi_img,
            None,
            fx=2.5,
            fy=2.5,
            interpolation=cv2.INTER_CUBIC
        )

        roi_gray = cv2.cvtColor(
            roi_large,
            cv2.COLOR_BGR2GRAY
        )

        roi_blur = cv2.GaussianBlur(
            roi_gray,
            (5, 5),
            0
        )

        _, roi_thresh = cv2.threshold(
            roi_blur,
            0,
            255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )

        try:

            custom_config = (
                r'--psm 7 '
                r'-c tessedit_char_whitelist='
                r'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            )

            plate_text = (
                pytesseract.image_to_string(
                    roi_thresh,
                    config=custom_config
                )
            )

            plate_text = (
                plate_text
                .strip()
                .replace(" ", "")
                .replace("\n", "")
            )

            fixed_plate = (
                fix_plate_text(
                    plate_text
                )
            )

            print(
                f"\n車牌 #{plate_counter}"
            )

            print(
                f"OCR原始結果 : "
                f"【{plate_text}】"
            )

            print(
                f"修正後結果 : "
                f"【{fixed_plate}】"
            )

            if is_valid_plate(
                    fixed_plate
            ):

                print(
                    "格式正確"
                )

                all_results.append({
                    "id": plate_counter,
                    "image": image_path,
                    "plate": fixed_plate
                })

                plate_counter += 1

            else:

                print(
                    "格式不符合"
                )

        except Exception as e:

            print(
                f"OCR辨識失敗：{e}"
            )

        roi_filename = (
            f"roi_plate_"
            f"{plate_counter}_"
            f"{os.path.basename(image_path)}"
        )

        cv2.imwrite(
            roi_filename,
            roi_img
        )

        cv2.rectangle(
            img_resized,
            (adjusted_x, y),
            (end_x, end_y),
            (0, 255, 0),
            3
        )

    cv2.imshow(
        f"Final Result: {image_path}",
        img_resized
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 主程式
if __name__ == '__main__':

    test_images = [
        "im1.jpg",
        "im2.jpg",
        "im3.jpg",
        "im4.jpg",
        "im5.jpg",
        "im6.jpg"
    ]

    for img_name in test_images:

        print("\n" + "=" * 60)

        print(
            f"正在測試: {img_name}"
        )

        detect_and_extract_roi(
            img_name
        )

    print("\n")
    print("最終辨識結果")

    if len(all_results) == 0:

        print(
            "沒有成功辨識任何車牌"
        )

    else:

        for result in all_results:

            print(
                f"車牌 #{result['id']} "
                f"({result['image']}) "
                f"→ {result['plate']}"
            )

        print(
            f"\n共成功辨識："
            f"{len(all_results)} 個車牌"
        )