import uvicorn
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

## Import for SER prediction BEGIN
from ppstructure.kie.predict_kie_token_ser import SerPredictor

import os
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json

from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args

logger = get_logger()
## Import for SER prediction END

## Additional import for RE prediction BEGIN
from ppstructure.kie.predict_kie_token_ser_re import SerRePredictor
from ppocr.utils.visual import draw_re_results
## Additional import for RE prediction END

@app.post("/ser_predict")
async def ser_predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Process the file contents as needed
    args = parse_args()
    ## Include arguments here (must be before initializing the predictor)
    args.kie_algorithm="LayoutXLM"
    args.ser_model_dir="./inference/ser_vi_layoutxlm_xfund_infer"
    args.image_dir="./ppstructure/docs/kie/input/zh_val_42.jpg"
    args.ser_dict_path="./train_data/XFUND/class_list_xfun.txt"
    args.vis_font_path="./doc/fonts/simfang.ttf"
    args.ocr_order_method="tb-yx"

    image_file_list = get_image_file_list(args.image_dir)
    ser_predictor = SerPredictor(args)
    count = 0
    total_time = 0

    os.makedirs(args.output, exist_ok=True)
    with open(
            os.path.join(args.output, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
                img = img[:, :, ::-1]
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            ser_res, _, elapse = ser_predictor(img)
            ser_res = ser_res[0]

            res_str = '{}\t{}\n'.format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": ser_res,
                    }, ensure_ascii=False))
            f_w.write(res_str)

            img_res = draw_ser_results(
                image_file,
                ser_res,
                font_path=args.vis_font_path, )

            img_save_path = os.path.join(args.output,
                                         os.path.basename(image_file))
            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))

    return {"filename": file.filename}

@app.post("/re_predict")
async def re_predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Process the file contents as needed
    args = parse_args()
    ## Include arguments here (must be before initializing the predictor)
    args.kie_algorithm="LayoutXLM"
    args.re_model_dir="./inference/re_vi_layoutxlm_xfund_infer"
    args.ser_model_dir="./inference/ser_vi_layoutxlm_xfund_infer"
    args.use_visual_backbone=False
    args.image_dir="./ppstructure/docs/kie/input/zh_val_42.jpg"
    args.ser_dict_path="./train_data/XFUND/class_list_xfun.txt"
    args.vis_font_path="./doc/fonts/simfang.ttf"
    args.ocr_order_method="tb-yx"

    image_file_list = get_image_file_list(args.image_dir)
    ser_re_predictor = SerRePredictor(args)
    count = 0
    total_time = 0

    os.makedirs(args.output, exist_ok=True)
    with open(
            os.path.join(args.output, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
                img = img[:, :, ::-1]
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            re_res, elapse = ser_re_predictor(img)
            re_res = re_res[0]

            res_str = '{}\t{}\n'.format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": re_res,
                    }, ensure_ascii=False))
            f_w.write(res_str)
            if ser_re_predictor.predictor is not None:
                img_res = draw_re_results(
                    image_file, re_res, font_path=args.vis_font_path)
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] +
                    "_ser_re.jpg")
            else:
                img_res = draw_ser_results(
                    image_file, re_res, font_path=args.vis_font_path)
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] +
                    "_ser.jpg")

            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))

    return {"filename": file.filename}

@app.get("/list_files")
async def list_files():
    directory_path = 'POC data'
    extensions = ['.png', '.jpg', '.jpeg', '.txt', '.pdf', '.docx', '.xlsx']
    files = find_files_with_extensions(directory_path, extensions)
    return {"files": files}

def find_files_with_extensions(directory, extensions):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in extensions:
                file_list.append(os.path.join(root, file))
    return file_list

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)