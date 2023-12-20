import uvicorn
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

## Import for SER prediction BEGIN
from ppstructure.kie.predict_kie_token_ser import SerPredictor

import os

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

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

## Import for multithreading BEGIN
from multiprocessing.pool import ThreadPool

## Import for multithreading END

## Import for converting files into images BEGIN
from pdf2image import convert_from_path
import shutil
import subprocess

## Import for converting files into images END

## Import for layout analysis BEGIN
from paddleocr import PPStructure, draw_structure_result, save_structure_res
from PIL import Image

## Import for layout analysis END


@app.post("/ser_predict")
async def ser_predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Process the file contents as needed
    args = parse_args()
    ## Include arguments here (must be before initializing the predictor)
    args.kie_algorithm = "LayoutXLM"
    args.ser_model_dir = "./inference/ser_vi_layoutxlm_xfund_infer"
    args.image_dir = "./ppstructure/docs/kie/input/zh_val_42.jpg"
    args.ser_dict_path = "./train_data/XFUND/class_list_xfun.txt"
    args.vis_font_path = "./doc/fonts/simfang.ttf"
    args.ocr_order_method = "tb-yx"

    image_file_list = get_image_file_list(args.image_dir)
    ser_predictor = SerPredictor(args)
    count = 0
    total_time = 0

    os.makedirs(args.output, exist_ok=True)
    with open(
        os.path.join(args.output, "infer.txt"), mode="w", encoding="utf-8"
    ) as f_w:
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

            res_str = "{}\t{}\n".format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": ser_res,
                    },
                    ensure_ascii=False,
                ),
            )
            f_w.write(res_str)

            img_res = draw_ser_results(
                image_file,
                ser_res,
                font_path=args.vis_font_path,
            )

            img_save_path = os.path.join(args.output, os.path.basename(image_file))
            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))

    return {"filename": file.filename}


@app.post("/ser_re_predict")
async def ser_re_predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Process the file contents as needed
    args = parse_args()
    ## Include arguments here (must be before initializing the predictor)
    args.kie_algorithm = "LayoutXLM"
    args.re_model_dir = "./inference/re_vi_layoutxlm_xfund_infer"
    args.ser_model_dir = "./inference/ser_vi_layoutxlm_xfund_infer"
    args.use_visual_backbone = False
    args.image_dir = "output/POC data/(Cleaned)5.1.1/(Cleaned)Guidelines for Incident Response and Management for Macau Branch/input_imgs/page_0009.png"
    args.ser_dict_path = "./train_data/XFUND/class_list_xfun.txt"
    args.vis_font_path = "./doc/fonts/simfang.ttf"
    args.ocr_order_method = "tb-yx"

    image_file_list = get_image_file_list(args.image_dir)
    ser_re_predictor = SerRePredictor(args)
    count = 0
    total_time = 0

    os.makedirs(args.output, exist_ok=True)
    with open(
        os.path.join(args.output, "infer.txt"), mode="w", encoding="utf-8"
    ) as f_w:
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

            res_str = "{}\t{}\n".format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": re_res,
                    },
                    ensure_ascii=False,
                ),
            )
            f_w.write(res_str)
            if ser_re_predictor.predictor is not None:
                img_res = draw_re_results(
                    image_file, re_res, font_path=args.vis_font_path
                )
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] + "_ser_re.jpg",
                )
            else:
                img_res = draw_ser_results(
                    image_file, re_res, font_path=args.vis_font_path
                )
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] + "_ser.jpg",
                )

            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))

    return {"filename": file.filename}


@app.get("/process_poc")
async def process_poc():
    directory_path = "POC data"
    extensions = [".png", ".jpg", ".jpeg", ".txt", ".pdf", ".docx", ".doc", ".xlsx"]
    files = find_files_with_extensions(directory_path, extensions)

    OUTPUT_PREFIX = "output"

    ## Save all pdf, png, jpg/jpeg and docx/doc as input_imgs BEGIN
    pool = ThreadPool(20)
    results = []
    for file in files:
        base = os.path.basename(file)
        filename_without_ext = os.path.splitext(base)[0]
        input_dir = os.path.dirname(file)
        output_dir = os.path.join(OUTPUT_PREFIX, input_dir, filename_without_ext)
        os.makedirs(output_dir, exist_ok=True)

        result = pool.apply_async(
            save_as_input_imgs,
            args=(
                file,
                output_dir,
            ),
        )

        image_paths = result.get()
        results.append(result)

    pool.close()
    pool.join()
    results = [r.get() for r in results]
    print(results)
    ## Save all pdf, png, jpg/jpeg and docx/doc as input_imgs END

    ## layout analysis + table recognition BEGIN
    image_files = find_image_files(os.path.join(OUTPUT_PREFIX, directory_path))
    print(image_files)
    pool = ThreadPool(20)
    results = []
    for image_file in image_files:
        input_imgs_dir = os.path.dirname(image_file)
        output_dir = os.path.dirname(input_imgs_dir)
        assert os.path.exists(output_dir) and os.path.isdir(output_dir) == True

        result = pool.apply_async(
            layout_analysis,
            args=(
                image_file,
                output_dir,
                True,
            ),
        )

        _ = result.get()
        results.append(result)

    pool.close()
    pool.join()
    results = [r.get() for r in results]
    print(results)
    ## layout analysis + table recognition END

    ## ser_re analysis BEGIN
    pool = ThreadPool(20)
    results = []
    for image_file in image_files:
        input_imgs_dir = os.path.dirname(image_file)
        output_dir = os.path.dirname(input_imgs_dir)
        assert os.path.exists(output_dir) and os.path.isdir(output_dir) == True

        result = pool.apply_async(
            ser_re_analysis,
            args=(
                image_file,
                output_dir,
                True,
            ),
        )

        _ = result.get()
        results.append(result)

    pool.close()
    pool.join()
    results = [r.get() for r in results]
    print(results)
    ## ser_re analysis END

@app.get("/list_files")
async def list_files():
    directory_path = "POC data"
    extensions = [".png", ".jpg", ".jpeg", ".txt", ".pdf", ".docx", ".doc", ".xlsx"]
    files = find_files_with_extensions(directory_path, extensions)
    return {"files": files}

'''
Return:
    None -> file format is not one of png/jpg/jpeg
    directory path -> ser_re analysis has been run before for this image
    infer.txt path -> ser_re analysis is successful
'''
def ser_re_analysis(file: str, output_dir, is_english):
    if (
        file.lower().endswith(".png")
        or file.lower().endswith(".jpg")
        or file.lower().endswith(".jpeg")
    ):
        ser_re_dir = os.path.join(output_dir, "ser_re")
        os.makedirs(ser_re_dir, exist_ok=True)

        base = os.path.basename(file)
        filename_without_ext = os.path.splitext(base)[0]
        img_ser_re_dir = os.path.join(ser_re_dir, filename_without_ext)
        os.makedirs(img_ser_re_dir, exist_ok=True)
        
        if not len(os.listdir(img_ser_re_dir)) == 0:
            return img_ser_re_dir
        
        subprocess.call(["python", "modified_predict_kie_token_ser_re.py", "--image_dir", file])

        return os.path.join(img_ser_re_dir, "infer.txt")

'''
Return:
    None -> file format is not one of png/jpg/jpeg
    directory path -> layout analysis has been run before for this image
    result.png path -> layout analysis is successful
'''
def layout_analysis(file: str, output_dir, is_english):
    if (
        file.lower().endswith(".png")
        or file.lower().endswith(".jpg")
        or file.lower().endswith(".jpeg")
    ):
        layout_dir = os.path.join(output_dir, "layout")
        os.makedirs(layout_dir, exist_ok=True)

        base = os.path.basename(file)
        filename_without_ext = os.path.splitext(base)[0]
        img_layout_dir = os.path.join(layout_dir, filename_without_ext)
        os.makedirs(img_layout_dir, exist_ok=True)
        
        if not len(os.listdir(img_layout_dir)) == 0:
            return img_layout_dir

        if is_english:
            table_engine = PPStructure(show_log=True, lang='en')
        else:
            table_engine = PPStructure(show_log=True)

        img = cv2.imread(file)
        result = table_engine(img)

        save_structure_res(result, img_layout_dir, filename_without_ext)

        for line in result:
            line.pop("img")
            print(line)

        font_path = "doc/fonts/simfang.ttf"  # font provided in PaddleOCR
        image = Image.open(file).convert("RGB")
        im_show = draw_structure_result(image, result, font_path=font_path)
        im_show = Image.fromarray(im_show)
        im_show.save(os.path.join(img_layout_dir, "result.png"))
        
        return os.path.join(img_layout_dir, "result.png")

'''
Return:
    None -> file format is not one of pdf/png/jpg/jpeg/docx/doc
    [] -> file has been converted into images before
    list of image paths -> conversion is successful
'''
def save_as_input_imgs(file: str, output_dir):
    if (
        file.lower().endswith(".pdf")
        or file.lower().endswith(".png")
        or file.lower().endswith(".jpg")
        or file.lower().endswith(".jpeg")
        or file.lower().endswith(".docx")
        or file.lower().endswith(".doc")
    ):
        input_imgs_dir = os.path.join(output_dir, "input_imgs")
        os.makedirs(input_imgs_dir, exist_ok=True)

        if not len(os.listdir(input_imgs_dir)) == 0:
            return []

    if file.lower().endswith(".pdf"):
        images = convert_from_path(file)

        image_paths = []
        for i, image in enumerate(images):
            image_path = f"{input_imgs_dir}/page_{str(i+1).zfill(4)}.png"
            image.save(image_path, "PNG")
            print(f"Saved page {i+1} as {image_path}")
            image_paths.append(image_path)

        return image_paths

    elif file.lower().endswith(".png"):
        i = 0
        image_path = f"{input_imgs_dir}/page_{str(i+1).zfill(4)}.png"
        shutil.copy(file, image_path)
        print(f"Saved page {i+1} as {image_path}")
        image_paths = image_path
        return image_paths

    elif file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
        i = 0
        image_path = f"{input_imgs_dir}/page_{str(i+1).zfill(4)}.jpg"
        shutil.copy(file, image_path)
        print(f"Saved page {i+1} as {image_path}")
        images_paths = image_path
        return images_paths

    elif file.lower().endswith(".docx") or file.lower().endswith(".doc"):
        base = os.path.basename(file)
        pdf_path = f"{input_imgs_dir}/{os.path.splitext(base)[0]}.pdf"
        subprocess.call(["unoconv", "-f", "pdf", "-o", pdf_path, file])

        images = convert_from_path(pdf_path)
        image_paths = []
        for i, image in enumerate(images):
            image_path = f"{input_imgs_dir}/page_{str(i+1).zfill(4)}.png"
            image.save(image_path, "PNG")
            print(f"Saved page {i+1} as {image_path}")
            image_paths.append(image_path)

        os.remove(pdf_path)  # Remove the temporary PDF file

        return image_paths

    elif file.lower().endswith(".xlsx") or file.lower().endswith(".txt"):
        return None

def find_image_files(directory_path):
    image_files = []
    if not os.path.isdir(directory_path):
        raise ValueError(f"{directory_path} is not a valid directory.")

    for root, dirs, files in os.walk(directory_path):
        if os.path.basename(root) == "input_imgs":
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    image_files.append(os.path.join(root, file))

    return image_files

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
