# Prepare for the venv environment
Create and activate the venv environment.
```bash
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
```

Use the following command to install KIE dependencies.
```bash
pip install -r requirements.txt
pip install -r ppstructure/kie/requirements.txt
pip install paddleocr -U
```

Install PaddlePaddle.
```bash
# If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# If you have no available GPU on your machine, please run the following command to install the CPU version
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

## Prepare for the dataset
```bash
mkdir train_data
cd train_data
# download and uncompress the dataset
wget https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar && tar -xf XFUND.tar
cd ..
```
- `train_data` directory is then created.

## Download models for `Inference using PaddleInference`
Firstly, download the inference SER inference model.
```bash
mkdir inference
cd inference
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_infer.tar && tar -xf ser_vi_layoutxlm_xfund_infer.tar
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_infer.tar && tar -xf re_vi_layoutxlm_xfund_infer.tar
cd ..
```
- `inference` directory is then created.

## SER inference
```bash
cd ppstructure
python3 kie/predict_kie_token_ser.py \
  --kie_algorithm=LayoutXLM \
  --ser_model_dir=../inference/ser_vi_layoutxlm_xfund_infer \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../train_data/XFUND/class_list_xfun.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx"
cd ..
```
- The visual results and text file will be saved in directory `ppstructure/output`.

## RE inference
```bash
cd ppstructure
python3 kie/predict_kie_token_ser_re.py \
  --kie_algorithm=LayoutXLM \
  --re_model_dir=../inference/re_vi_layoutxlm_xfund_infer \
  --ser_model_dir=../inference/ser_vi_layoutxlm_xfund_infer \
  --use_visual_backbone=False \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../train_data/XFUND/class_list_xfun.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx"
cd ..
```
- The visual results and text file will be saved in directory `ppstructure/output`.

If you want to use a custom ocr model, you can set it through the following fields.
* --det_model_dir: the detection inference model path
* --rec_model_dir: the recognition inference model path

## layout analysis + table recognition + layout recovery (PDF to Word)
* Note that `--image_orientation=true` is not used because image orientation is not recommended. The reason is that the image orientation feature easily rotates the image by 180 degree incorrectly.

1. Install the image direction classification dependency package paddleclas (if you do not use the image direction classification, you can skip it).
```bash
pip install paddleclas
```

2. Fix the error.
```bash
user_site_folder=$(python -c 'import site; print(site.getsitepackages()[0])')
faiss_folder="${user_site_folder}/faiss"
cd "${faiss_folder}"
ln -s swigfaiss.py swigfaiss_avx2.py
```

3. cd back to the project root directory.

4. Test.
- layout analysis + table recognition + layout recovery (PDF to Word):
```bash
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --recovery=true --lang='en'
```
- `--lang='en'` is for English image (rather than Chinese image) and is not needed in the above case.

- layout analysis + table recognition:
```bash
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --lang='en'
```
- `--lang='en'` is for English image (rather than Chinese image) and is needed in the above case, or otherwise the quality is worse.

# API development
## Install dependencies.
```bash
pip install fastapi
pip install python-multipart
```

```bash
sudo apt update
sudo apt install poppler-utils
pip install pdf2image

sudo apt install libreoffice
sudo apt install unoconv
sudo sed -i 's|#!/usr/bin/env python3|#!/usr/bin/python3|' /usr/bin/unoconv
```

## Put `POC data` in the project root directory
