from fastapi import FastAPI, HTTPException,Request,Form
from starlette.middleware.cors import CORSMiddleware # ブラウザからクロスオリジンリクエスト(CORS)を許可
import base64
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import math
from io import BytesIO
import re
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from reedsolo import RSCodec
from cryptography.fernet import Fernet
import pyocr
import pyocr.builders

import pyocr.tesseract
import cv2
import ast
import numpy as np

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

# スクリプトのフォルダパス
FILE_FOLDER_PATH = os.path.dirname(__file__)
# 画像ファイル名
SAVE_PICTURE_NAME = './3D-Code.png'
# 保存するファイルパス
IMAGE_FILE_PATH = FILE_FOLDER_PATH + SAVE_PICTURE_NAME
# 余白
PADDING = 20
# フォントサイズ
FONTSIZE = 20
# メタ情報の最小のサイズ
META_MIN = 8
# メタ情報の最大のサイズ
META_MAX = 60
# AESのKey
AES_KEY = b"AES_KEY"

def Crypto_AES(input_text):
    print('Crypto_AESの始まり')
    print('------------------------------------------------')
    cipher_suite = Fernet(AES_KEY)
    print("Encryption Key:", AES_KEY)
    print("cipher_suite Key:", cipher_suite)

    encrypted_text = cipher_suite.encrypt(input_text.encode())
    print("encrypted_text:", encrypted_text)
    encrypted_text_length = len(encrypted_text)
    print('------------------------------------------------')
    return encrypted_text,encrypted_text_length


def Reed_solo(encrypted_text,spl_num,level):
    print('Reed_soloの始まり')
    print('------------------------------------------------')
    # 暗号化文を分割したい文字数で分割するため何等分できるか計算
    count = int(len(encrypted_text) / spl_num)
    # 余りが出た場合は１回増やす
    if len(encrypted_text) % spl_num != 0:
        count += 1

    # 分割したもの格納用
    split_text = []

    try:
        # 分割して格納、ここで暗号化文をそのままいじったのでめんどいことになった
        for i in range(count):
            split_text.append(encrypted_text[:120])
            encrypted_text = encrypted_text[120:]

        division_reed = []

        print('split_text:',split_text)
        
        # 分割した物に順にリーソロ付与。格納場所は上書きしている。
        for num,i in enumerate(split_text):
            # リードソロモン符号の付与
            if level == "L":
                rsc = RSCodec(int(len(i)/4))
            elif level == "M":
                rsc = RSCodec(int(len(i)/2))
            elif level == "Q":
                rsc = RSCodec(int(len(i)))
            elif level == "H":
                rsc = RSCodec(int(len(i)*2))
            else:
                print("レベル指定エラー")
            
            e1 = rsc.encode(i)
            split_text[num] = e1

        # 符号付与後を纏めるための引数
        reed_text = ""

        # 16進数化してreed_textに入れる
        for i in split_text:
            Hex_text = i.hex()
            print('Hex_text:',Hex_text)
            reed_text += Hex_text

            division_reed.append(len(Hex_text))

        reed_text_length = len(reed_text)
        print('reed_text:',reed_text)

        # 結果の表示
        # print("分割した場合\t入力した文字列の長さ：",len(input_text),"\t付与レベル：",level,"\t付与後の文字数：",reed_text_length,"\t\t分割数:",count)
        print('0付与前のreed_text:',reed_text)
        print('------------------------------------------------')
        return reed_text,reed_text_length,division_reed
    except:
        pass

def Adjust(reed_text):
    print('ajustの始まり')
    print('------------------------------------------------')
    
    print("Root:", math.ceil(math.sqrt(len(reed_text))))
    maxmoji = math.ceil(math.sqrt(len(reed_text))) ** 2

    while maxmoji > len(reed_text):
        reed_text += '0'

    print('reed_text',reed_text)

    print("リードソロモンした文字列の長さ：",len(reed_text))
    
    print('------------------------------------------------')
    
    return reed_text

def MetaDataCreate(encrypted_text_length,split_length,level,division_reed):
    print('MetaDataCreateの始まり')
    print('------------------------------------------------')

    MetaData = [encrypted_text_length,division_reed]

    print('MetaData:',MetaData)
    print('MetaData:',type(MetaData))
    MetaData = str(MetaData)
    print('MetaData:',MetaData)
    print('MetaData:',type(MetaData))

    rsc_meta = RSCodec(int(len(MetaData)*2.75))
    Meta_reed = rsc_meta.encode(MetaData.encode())
    print('Meta_reed:',Meta_reed)
    print('Meta_reedの長さ:',len(Meta_reed))
    Meta_bit = "".join(format(byte, "08b") for byte in Meta_reed)
    print('------------------------------------------------')
    return Meta_bit


def CodeCreate(reed_text, Meta_bit):
    print('CodeCreateの始まり')
    print('------------------------------------------------')
    # 縦と横の間隔を設定
    spacing = 5  # 5pxの間隔
    font_with_spacing = FONTSIZE + spacing  # フォントサイズに間隔を加える
    
    # 画像の幅と高さを計算
    width = height = math.ceil(math.sqrt(len(reed_text))) * font_with_spacing + (PADDING * 2)
    print('width,height', width, height)
    
    # イメージオブジェクト作成
    display_view = Image.new('RGBA', [width, height], (255, 255, 255, 255))
    draw = ImageDraw.Draw(display_view)
    font = ImageFont.truetype("./app/PTSerif-Bold.ttf", FONTSIZE)
    
    index = 0

    # 文字の描画
    for y in range(PADDING, height - PADDING, font_with_spacing):
        for x in range(PADDING, width - PADDING, font_with_spacing):
            if index < len(reed_text):
                if index >= len(Meta_bit) or Meta_bit[index] == '0':
                    color = (0, 0, 0, 255)  # 黒
                else:
                    color = (0, 255, 0, 255)  # 緑
                
                if index < len(reed_text):
                    # 擬似的に太字にするために文字を少しずつずらして描画
                    offsets = [(-0.2, -0.2), (0, -0.2), (0.2, -0.2), 
                            (-0.2, 0), (0, 0), (0.2, 0),
                            (-0.2, 0.2), (0, 0.2), (0.2, 0.2)]
                    for dx, dy in offsets:
                        draw.text((x + dx, y + dy), reed_text[index], fill=color, font=font)

                index += 1
    
    # 画像の周りに黒い太線を描画
    border_thickness = 5  # 太線の幅
    
    # 上の線
    draw.rectangle([0, 0, width, border_thickness], fill=(0, 0, 0, 255))
    # 下の線
    draw.rectangle([0, height - border_thickness, width, height], fill=(0, 0, 0, 255))
    # 左の線
    draw.rectangle([0, 0, border_thickness, height], fill=(0, 0, 0, 255))
    # 右の線
    draw.rectangle([width - border_thickness, 0, width, height], fill=(0, 0, 0, 255))
    
    # 画像保存
    print('------------------------------------------------')
    return display_view



def Correction(TDCode, output_size=(640, 640)):

    TDCode = np.array(TDCode)

    if TDCode.shape[-1] == 4:
        TDCode = cv2.cvtColor(TDCode, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(TDCode, cv2.COLOR_BGR2GRAY)

    # 大津の二値化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    if len(approx) != 4:
        raise HTTPException(status_code=400, detail="正方形の頂点が見つかりませんでした。")

    points = approx.reshape(4, 2)
    sorted_points = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    sorted_points[0] = points[np.argmin(s)]
    sorted_points[2] = points[np.argmax(s)]
    sorted_points[1] = points[np.argmin(diff)]
    sorted_points[3] = points[np.argmax(diff)]

    dst_points = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(sorted_points, dst_points)
    corrected_image = cv2.warpPerspective(TDCode, matrix, output_size)


    cv2_conv = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
    TDCode_white = Image.fromarray(cv2_conv)

    
    # 輝度調整処理を追加
    hsv = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 30)  # Sチャンネルの値を上げる
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 50)  # Vチャンネルの値を上げる
    corrected_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    TDCode = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
    TDCode = Image.fromarray(TDCode)
    
    # 彩度の調整
    enhancer_color = ImageEnhance.Color(TDCode_white)
    TDCode_white = enhancer_color.enhance(1.0)

    # 輝度の調整
    enhancer_brightness = ImageEnhance.Brightness(TDCode_white)
    TDCode_white = enhancer_brightness.enhance(1.0)

    # 均一化フィルタを適用して端のムラを軽減
    image = TDCode_white.filter(ImageFilter.SMOOTH)
    TDCode = image.filter(ImageFilter.EDGE_ENHANCE)
    

    return TDCode,TDCode_white


def MetaDecode(TDCode,print_num):
    print('MetaDecodeの始まり')
    print('------------------------------------------------')

    print(type(TDCode))

    output_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    # tesseractの登録
    pyocr.tesseract.TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
    # ツールの取得
    tools = pyocr.get_available_tools()

    if len(tools) == 0:
        print("OCRツールが見つかりません。Tesseractがインストールされているか確認してください。")
        exit(1)

    # Tesseractを使用
    tool = tools[0]
    if print_num == 1:
        print(f"使用中のOCRツール: {tool.get_name()}")

    # 言語設定
    langs = tool.get_available_languages()
    if print_num == 1:
        print(f"使用可能な言語: {langs}")

    # 画像をRGBAに変換
    TDCode.putalpha(alpha=255)

    Meta_result = ""
    imageList = []

    # OCR実行
    first_ocr_results = tool.image_to_string(
        TDCode,
        lang='eng',
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=5)
    )

    print()
    print("取得列：",len(first_ocr_results))

    vertical_point_list = []
    for numb,i in enumerate(first_ocr_results):
        #print(numb,"番目の座標：",i.position)
        if i.content == "":
            print("中身がない")
        else:
            left,top = i.position[0]
            right,bottom = i.position[1]

            point = left,right
            vertical_point_list.append(point)


    # OCR実行
    second_ocr_results = tool.image_to_string(
        TDCode,
        lang='eng',
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=6)
    )

    print()
    print("取得行数：",len(second_ocr_results))

    holizontal_point_list = []
    for numb,i in enumerate(second_ocr_results):
        #print(numb,"番目の座標：",i.position)
        if i.content == "":
            print("中身がない")
        else:
            left,top = i.position[0]
            right,bottom = i.position[1]

            point = top,bottom
            holizontal_point_list.append(point)

    if len(holizontal_point_list) != len(vertical_point_list):
        raise HTTPException(status_code=400, detail="OCRの数が合いません")

    print()
    print("横座標リスト：",holizontal_point_list)
    print("縦方向リスト：",vertical_point_list)
    print()
    try:
        vertical_point_list.reverse()
        rect_image = TDCode.copy()
        # Pillow 画像を NumPy 配列に変換
        rect_image_np =  np.array(rect_image)

        for verti_num,i in enumerate(holizontal_point_list):
            # print("縦方向に",verti_num,"番目の座標：")
            for holi_num,j in enumerate(vertical_point_list):

                top,bottom = i[0],i[1]
                left,right = j[0],j[1]

                rect_posi1 = left,top
                rect_posi2 = right,bottom

                # print("\t横方向に",holi_num,"番目の座標：",rect_posi1,rect_posi2)

                # 元画像を切り取り
                cropped_img = TDCode.crop((left, top, right, bottom))
                imageList.append(cropped_img)
                output_path = os.path.join(output_dir, f"text_{verti_num+1}_{holi_num+1}.png")
                cropped_img.save(output_path)

                cv2.rectangle(rect_image_np, rect_posi1, rect_posi2, (255,255,255), 1)
    except Exception as e:
        print("eroor:",e)

    # cv2からimageに変換
    new_image = cv2.cvtColor(rect_image_np, cv2.COLOR_BGR2RGB)
    rect_image_reconv = Image.fromarray(new_image)
    rect_image_reconv.show()

    pixel_list = []
    
    for image in imageList:
        # NumPy配列に変換
        cropped_image_image = image.resize((100, 100))
        pixels = np.array(cropped_image_image)
    
        color_result = np.mean(np.mean(pixels,axis=1),axis=0)
        pixel_list.append(color_result)
        result_difference = color_result[1] - np.min(color_result)

        if int(np.abs(result_difference)) < 15:
            Meta_result += "0"
        else:
            Meta_result += "1"

    print('Meta_result:',Meta_result)


    target = "0" * 16  # 0が24個連続のパターンを定義

    index = -1

    for i in range(0,len(Meta_result),8):
        index = Meta_result.find(target, i)
        if index % 8 == 0:
            break
    
    if index == -1:
        raise HTTPException(status_code=400, detail="色検知の部分が上手くいっていません。")
    
    print("index:",index)
    
    byte_result = Meta_result[0:index]

    print('len,byte_result:',len(byte_result)) 
    print('byte_result:',byte_result) 

    byte_result = bytes(int(byte_result[i:i+8], 2) for i in range(0, len(byte_result), 8))

    print('len,byte_result:',len(byte_result)) 
    print('byte_result:',byte_result) 

    decoded_meta = ""

    # 最低34文字、最高100文字のメタ情報であるため
    for n in range(META_MIN, META_MAX):
        try:
            rsc = RSCodec(int(n * 2.75))
            decoded_meta = rsc.decode(byte_result)[0]
            decoded_meta = decoded_meta.decode()
            if len(decoded_meta) == n:
                print("メタ情報のデコードに成功")
                break

        except Exception:
            pass
    print('------------------------------------------------')

    print(type(decoded_meta))
    if decoded_meta != "":
        decoded_meta = ast.literal_eval(decoded_meta)
        print(decoded_meta)  # [100, [400]]
        print(type(decoded_meta))

    return decoded_meta


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def image_crop(TDCode,print_num):
        # tesseractの登録
    pyocr.tesseract.TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
    # ツールの取得
    tools = pyocr.get_available_tools()

    # tesseractが見つからなかった場合
    if len(tools) == 0:
        print("OCRツールが見つかりません。Tesseractがインストールされているか確認してください。")
        exit(1)

    # Tesseractを使用
    tool = tools[0]
    print(f"使用中のOCRツール: {tool.get_name()}")
    # 使用可能な言語設定の表示
    langs = tool.get_available_languages()
    print(f"使用可能な言語: {langs}")

    # 画像をRGBAに変換
    TDCode.putalpha(alpha=255)
    # OCR実行
    results = tool.image_to_string(
        TDCode,
        lang='eng',
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=6)
    )


    # 画像の横幅の統一のためにxの最大値、最小値を格納する変数
    # 初期値として1番目の行の起点のx座標をぶち込んである
    max_x = results[0].position[0][0]
    min_x = results[0].position[0][0]

    # 上記で実行したOCR結果からx座標の最大値、最小値を探す
    for box in results:
        # 現状の最小x座標より起点のx座標の方が小さかった場合入れ替え
        if min_x > box.position[0][0]:
            min_x = box.position[0][0]
        # 現状の最大x座標より終点のx座標の方が大きかった場合入れ替え
        if max_x < box.position[1][0]:
            max_x = box.position[1][0]


    # 過去に文字数の数え方云々で議論がわき上がった際に規定化するために変数にぶち込んだもの
    # 現状となっては必要がないが、この変数で下記を作ってしまっているため変更は保留
    result_row_length = len(results)

    # 全体の縦の長さ = 最後の行の終点のy座標 - 最初の行の起点のy座標
    y_long = results[result_row_length-1].position[1][1] - results[0].position[0][1]
    # 1文字のyの長さ = 全体の縦の長さ ÷ 行の総数
    tiny = y_long/result_row_length
    # 四捨五入を行ってOCRによる微変動を抑制している
    tiny = int(Decimal(str(tiny)).quantize(Decimal('0'), ROUND_HALF_UP))

    # 全体の横の長さ = x座標の最大値 - (最初の行の起点のx座標 - 10)
    # ここはx座標の最小値を取得する前に手作業でいい感じに取得できるようにしていた場所で、今何も考えずに治すと大変なことになるのでいったん保留
    x_long = max_x - (results[0].position[0][0] - 10)
    # 1文字のxの長さ = 全体の横の長さ ÷ 行の総数
    tiny_x = x_long/result_row_length
    # 四捨五入を行ってOCRによる微変動を抑制している
    tiny_x = int(Decimal(str(tiny_x)).quantize(Decimal('0'), ROUND_HALF_UP))

    # OCR情報の表示
    print()
    print("取得した行の数：",result_row_length)
    print("yの長さ：",y_long,"\txの長さ：",max_x)
    print("1つ分のyの長さ：",tiny,"\t1つ分のxの長さ：",tiny_x)
    print()

    # 分割した画像を格納するリスト
    image_list = []

    # 1文字ずつ分割する際の初期座標の算出

    # 初期x座標 = x座標の最小値 - (1文字の横の長さ * 0.4)
    # 文字を認識しやすいように初期値を少し左にずらして文字認識精度を上げている処理、下手にいじりにくい場所その２
    one_x = min_x - int((tiny_x*0.4))
    # 初期y座標 = 各行の起点y座標 - (1文字の縦の長さ * 0.2)
    # 文字を認識しやすいように初期値を少し上にずらして文字認識精度を上げている処理、下手にいじりにくい場所その３
    one_y = results[0].position[0][1] - int((tiny*0.2))

    # 1文字ずつ分割する際の結論座標タプル

    # 起点 = 初期x座標 -2 , 初期y座標 -2
    # 文字を認識しやすい（省略）下手にいじりにくい場所その４
    posi1 = one_x-2,one_y-2
    # 終点 = 起点のx座標 +1文字分の横の長さ +8 , 起点のy座標 +1文字分の縦の長さ +8
    # 文字を認識しやすい（省略）下手にいじりにくい場所その５
    posi2 = posi1[0]+tiny_x+8,posi1[1]+(tiny*result_row_length)+8


    # OCRした結果を1行ずつ
    for i in range(int(result_row_length)):
        # 最初の場合はそのまま描画、2回目以降は1列ずらす
        if i == 0:
            # 切り出し部分の起点のx,yと終点のx,yの代入
            left, top = posi1
            right, bottom = posi2

            # 元画像を縦長に切り取り
            cropped_img = TDCode.crop((left, top, right, bottom))

        else:
            # 起点終点のx座標をすべて1文字分の横の長さ分ずらす
            posi1 = posi1[0]+tiny_x,posi1[1]
            posi2 = posi2[0]+tiny_x,posi2[1]

            # 1文字だけ画像として切り出し
            left, top = posi1
            right, bottom = posi2

            # 元画像を縦長に切り取り
            cropped_img = TDCode.crop((left, top, right, bottom))

        # 画像の引き渡し
        image_list.append(cropped_img)

    return image_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def do_OCR(TDcode):
    result = TDcode
    result_copy = result.copy()
    image = TDcode

    gray_image = image.convert("L")  # グレースケールに変換
    
    initial_brightness = np.mean(np.array(gray_image))  # ピクセル値の平均を計算
    print(f"調整前の輝度: {initial_brightness:.2f}")

    while initial_brightness <= 230:
        # 輝度を調整
        image_array = np.array(image, dtype=np.float32)  # ピクセル値をfloat32に変換
        adjusted_array = image_array + 1  # 差分を加算
        adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)  # 範囲をクリップしてuint8に変換

        # 調整後の画像を作成
        image = Image.fromarray(adjusted_array)

        # 調整後の輝度を計算
        initial_brightness = np.mean(np.array(image))  # ピクセル値の平均を計算

    print(f"調整後の輝度: {initial_brightness:.2f}")


    # 取得した画像をImageからcv2に変換
    img_of_cv2 = np.array(image, dtype=np.uint8)
    crop_cv2 = cv2.cvtColor(img_of_cv2, cv2.COLOR_RGB2BGR)

    # cv2からimageに変換
    new_image = cv2.cvtColor(img_of_cv2, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(new_image)

    # 保存用のディレクトリを作成（実行時の日時を使用）
    output_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    # tesseractの登録
    pyocr.tesseract.TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
    # ツールの取得
    tools = pyocr.get_available_tools()

    # print(tools)
    if len(tools) == 0:
        print("OCRツールが見つかりません。Tesseractがインストールされているか確認してください。")
        exit(1)

    # Tesseractを使用
    tool = tools[0]
    print(f"使用中のOCRツール: {tool.get_name()}")

    # 画像をRGBAに変換
    image.putalpha(alpha=255)
    # OCR実行
    first_ocr_results = tool.image_to_string(
        image,
        lang='eng',
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=5)
    )

    print("取得行数：",len(first_ocr_results))

    # 検知した情報を格納する辞書型の宣言
    dics = {}
    # 作製した辞書にOCRで検知した行数分配列を作製
    for n in range(len(first_ocr_results)):
        dics[f"array_{n}"] = []

    # 取得した画像をImageからcv2に変換
    img_of_cv2 = np.array(result, dtype=np.uint8)
    rect_image = cv2.cvtColor(img_of_cv2, cv2.COLOR_RGB2BGR)

    for number,i in enumerate(first_ocr_results):

        left,top = i.position[0]
        right,bottom = i.position[1]

        if number == 0:
            right = result_copy.size[0]
            left = first_ocr_results[number+1].position[1][0]
            print("最初\t右辺：",right,"\t左辺：",left)
        
        elif len(first_ocr_results) == number+1:
            right = first_ocr_results[number-1].position[0][0]
            left = 0
            print("最後\t右辺：",right,"\t左辺：",left)
        
        else:
            right = first_ocr_results[number-1].position[0][0]
            left = first_ocr_results[number+1].position[1][0]
            print("途中\t右辺：",right,"\t左辺：",left)

        top = 0
        bottom = result_copy.size[1]

        rect_posi1 = left,top
        rect_posi2 = right,bottom

        color = list(np.random.choice(range(256), size=3))
        color_tapl = int(color[0]),int(color[1]),int(color[2])
        cv2.rectangle(rect_image, rect_posi1, rect_posi2, color_tapl, 1)


        # 元画像を縦長に切り取り
        cropped_img = result_copy.crop((left, top, right, bottom))

        gray_image = cropped_img.convert("L")  # グレースケールに変換
        initial_brightness = np.mean(np.array(gray_image))  # ピクセル値の平均を計算
       
        # 取得した画像をImageからcv2に変換
        img_of_cv2 = np.array(cropped_img, dtype=np.uint8)
        crop_cv2 = cv2.cvtColor(img_of_cv2, cv2.COLOR_RGB2BGR)
        cv2.imwrite('croped.jpg', crop_cv2)


        # 画像をRGBAに変換
        cropped_img.putalpha(alpha=255)

        # OCR実行
        second_ocr_results = tool.image_to_string(
            cropped_img,
            lang='eng',
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=6)
        )

        print(number,"番目のocr：",len(second_ocr_results))

        # 取得した画像をImageからcv2に変換
        img_of_cv2 = np.array(cropped_img, dtype=np.uint8)
        crop_cv2 = cv2.cvtColor(img_of_cv2, cv2.COLOR_RGB2BGR)

        if not second_ocr_results:
            print("text is none")
        else:
            for num in range(len(first_ocr_results)):
                try:
                    j = second_ocr_results[num]
                    text = j.content
                    if not text:
                        text = "0"
                    elif len(text) > 1:
                        text = text[0]
                    
                    if not(re.fullmatch(r'[a-f0-9]*', text)):  # ASCII範囲外 (127を超える値)
                        text = '0' # '0'のASCII値に置き換える

                    # 辞書のしかるべき場所に挿入
                    dics[f"array_{num}"].append(text)
                    
                    cv2.rectangle(crop_cv2, j.position[0], j.position[1], (255, 0, 0), 1)
                except Exception as e:
                    print("eroor:",e)
                    dics[f"array_{num}"].append("0")

        # cv2からimageに変換
        result_copy = cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2RGB)
        result_copy = Image.fromarray(result_copy)


    # cv2からimageに変換
    result_copy = cv2.cvtColor(rect_image, cv2.COLOR_BGR2RGB)
    result_copy = Image.fromarray(result_copy)


    ocr_result = ""
    # 検知行数分回して結果の結合
    for n in range(len(dics)):
        if not dics[f"array_{n}"]:
            print("dics[array_{n}] is none")
        else:
            result_list = dics[f"array_{n}"]
            print(n,"番目のリスト：\t",result_list,"\t文字数：",len(result_list))
            result_list.reverse()
            print(n,"番目のリスト：\t",result_list,"\t文字数：",len(result_list),"\n")
            # 1行分の結果を結合
            result = "".join(dics[f"array_{n}"])

            # 1行分の結果を結果に結合
            ocr_result = ocr_result + result

    print('ocr_result\t：',ocr_result,"\t文字数：",len(ocr_result))

    return ocr_result

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ReedDecode(decoded_meta,ocr_result):

    # 本来のリードソロモン符号付き合計文字数を求める
    Reed_total_len = sum(decoded_meta[1])

    ocr_result = ocr_result[0:Reed_total_len]

    print("0除去後のocr_result:",ocr_result,"\n")
    
    ocr_result_division = []

    start = 0

    for i in decoded_meta[1]:
        ocr_result_division.append(ocr_result[start:i + start])
        start += i

    
    print('ocr_result_division')
    for i in ocr_result_division:
        print(i,"文字数：",len(i))
    print()


    decoded_text = ""

    # リードソロモン符号で暗号文を復号
    for i,text in enumerate(ocr_result_division):
        if i != len(ocr_result_division) - 1:
            try:
                rsc = RSCodec(120)
                decode_byte = bytes.fromhex(text)
                print('decode_byte:',decode_byte)
                decoded_message = rsc.decode(decode_byte)[0]
                decoded_text += decoded_message.decode()

            except Exception as e:
                print(i + 1,"番目のdecodeに失敗しました。")
                print(e)
                print('text:',text)
                print(type(text))
                pass
        else:
            print(int((decoded_meta[0] - (120 * (len(decoded_meta[1]) - 1)))))
            try:
                rsc = RSCodec(int((decoded_meta[0] - (120 * (len(decoded_meta[1]) - 1)))))
                decode_byte = bytes.fromhex(text)
                print('decode_byte:',decode_byte)
                decoded_message = rsc.decode(decode_byte)[0]
                decoded_text += decoded_message.decode()

            except Exception as e:
                print("末端のdecodeに失敗しました。")
                print(e)
                print('text:',text)
                print(type(text))
                pass

    print('decode_text:',decoded_text)

    # 共通かぎで元文を取り出す
    if decoded_text != '':
        cipher_suite = Fernet(AES_KEY)
        decrypted_text = cipher_suite.decrypt(decoded_text).decode()
        print("\n\t復号した文章：\t", decrypted_text, "\n")
    else:
        decrypted_text = '読み取りに失敗しました。'

    print('------------------------------------------------')
    return decrypted_text



app = FastAPI()

# CORSのミドルウェアの設定
# 別ドメイン/ポート間でのリクエストを許可する仕組み。
app.add_middleware(
    CORSMiddleware,
    # すべてのオリジンからのリクエストを許可:*,許可しない:空
    allow_origins=["*"],
    # 認証情報(クッキーなど)の送信を許可
    allow_credentials=True,
    # すべてのHTTPメソッド(GET,POST)を許可
    allow_methods=["*"],
    # 任意のHTTPヘッダーを許可
    allow_headers=["*"]
)


# エンドポイントにアクセスした時に指定したHTMLファイルを返す
@app.get("/", response_class=HTMLResponse)
async def server_html():
    # 特定のHTMLを指定
    with open(r"/var/www/html", "r", encoding="utf-8") as file:
        # 指定したHTMLを読み込んで返す
        html_content = file.read()
    return HTMLResponse(content=html_content)



# Write.html側の処理
@app.post("/api/data")
async def get_data(request: Request):
    # クライアントから送信されたデータを取得
    request_data = await request.json()
    print('request_data:',request_data)
    input_text = request_data.get("text")
    print('input_text:',input_text)

    split_length = 120
    level = "Q"

    # 共通鍵で本文を暗号化
    encrypted_text,encrypted_text_length = Crypto_AES(input_text)
    # 分割してリードソロモン符号を付与
    reed_text,reed_text_length,division_reed = Reed_solo(encrypted_text,split_length,level)
    # 平方数になるまで文字数を増加させる
    reed_text = Adjust(reed_text)
    # メタ情報を生成
    Meta_bit = MetaDataCreate(encrypted_text_length,split_length,level,division_reed)

    print('mainのログが挟まります')
    print('------------------------------------------------')

    print('reed_text:',reed_text)
    print('reed_textの長さ:',len(reed_text))
    print('Meta_bit:',Meta_bit)
    print('Meta_bitの長さ:',len(Meta_bit))

    print('------------------------------------------------')
    
    # 多次元コード生成
    TDCode = CodeCreate(reed_text,Meta_bit)

    # TDCodeはPillowのImageオブジェクト
    buffer = BytesIO()
    TDCode.save(buffer, format="PNG")  # 画像フォーマットを指定 (例: PNG)
    buffer.seek(0)  # バッファの先頭に移動
    base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # print("Base64 Data:", base64_data)
    return {"img":base64_data}


# 受信するデータのスキーマ(構造)を定義
class ImageData(BaseModel):
    # Base64でエンコードされたデータを受信
    image_data: str



# Read.html側の処理
@app.post("/api/upload_image")
async def upload_image(TDCode: ImageData):
    try:

        # Base64デコード
        TDCode = base64.b64decode(TDCode.image_data)
        TDCode = Image.open(BytesIO(TDCode))

        print('TDCode:',type(TDCode))

        try:
            TDCode= Correction(TDCode)
        except HTTPException as e:
            return {"message": e.detail}
        

        try:
            ocr_result = do_OCR(TDCode)
        except Exception as e:
            print("分割error！",e)

        
        # ログ出力:1 しない:0
        # メタデータの読み取り、メタ情報のデコード
        try:
            decoded_meta = MetaDecode(TDCode,1)
        except HTTPException as e:
            return {"message": e.detail}


        print('decoded_meta:',decoded_meta)
        print('type(decoded_meta):',type(decoded_meta))
        print('len(decoded_meta):',len(decoded_meta))

        if type(decoded_meta) == bytes or decoded_meta == "":
            print("メタ情報の読み取りに失敗しました。再度、3D-Codeを読み取ってください。")
            return {"message": "メタ情報の読み取りに失敗しました。再度、3D-Codeを読み取ってください。"}

        # リードソロモン、共通かぎのデコード処理
        decrypted_text = ReedDecode(decoded_meta,ocr_result)


        return {"message": decrypted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")
